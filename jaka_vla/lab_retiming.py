"""Collision-checked geometric fallback with bounded TOPPRA timing for simulation."""
import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint


def retime_and_validate(planner,result,velocity=.6,acceleration=1.,dt=.004,goal=None,start=None,hold_orientation=False):
    """The interpolated path is rechecked; spline smoothing cannot bypass collision."""
    from curobo.types.robot import JointState
    points=np.asarray(result.interpolated_plan.position.cpu(),dtype=np.float64)
    points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-7]]
    if len(points)<2:raise ValueError('Degenerate geometric path')
    # Preserve optimizer knot geometry. TOPPRA solves velocity/acceleration limits
    # separately; no trajectory is executed until the sampled spline passes checks.
    progress=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    progress/=progress[-1]
    path=ta.SplineInterpolator(progress,points)
    retimer=algo.TOPPRA([constraint.JointVelocityConstraint(np.full(6,velocity)),
                        constraint.JointAccelerationConstraint(np.full(6,acceleration))],path,
                        solver_wrapper='seidel')
    trajectory=retimer.compute_trajectory(0.,0.)
    if trajectory is None:raise RuntimeError('TOPPRA timing infeasible')
    # TOPPRA's discretized bounds can overshoot between grid points. Resample
    # and stretch time until the actual 250 Hz command respects both limits.
    scale=1.15
    for _ in range(4):
        count=int(np.ceil(trajectory.duration*scale/dt))+1
        if count>15000:raise ValueError('Retimed path exceeds 60 s budget')
        times=np.minimum(np.arange(count)*dt/scale,trajectory.duration)
        q=trajectory(times);v=trajectory(times,1)/scale;a=trajectory(times,2)/scale**2
        if not all(np.isfinite(x).all() for x in (q,v,a)):raise ValueError('Nonfinite retimed path')
        ratio=max(float(np.abs(v).max())/velocity,float(np.sqrt(np.abs(a).max()/acceleration)))
        if ratio<=1.:break
        scale*=ratio*1.1
    else:raise ValueError('Retimed sampled limits exceeded after time stretching')
    mg=planner.motion_gen
    if goal is None or start is None:raise ValueError('Explicit start and goal required')
    if np.max(np.abs(q[0]-start.detach().cpu().numpy().reshape(-1)))>.002:raise ValueError('Path start mismatch')
    endpoint=mg.compute_kinematics(JointState.from_position(mg.tensor_args.to_device(q[-1:]),joint_names=planner.active_joints_name))
    pos_error=float((endpoint.ee_pos_seq-goal.position).norm().item())
    import torch
    rot_error=float((2*torch.acos(torch.sum(endpoint.ee_quat_seq*goal.quaternion,dim=-1).abs().clamp(0,1))).item())
    if pos_error>.005 or rot_error>.05:raise ValueError(f'Endpoint mismatch: {pos_error:.5f} m / {rot_error:.5f} rad')
    for start in range(0,len(q),256):
        positions=mg.tensor_args.to_device(q[start:start+256]).unsqueeze(0)
        mg.rollout_fn.robot_self_collision_constraint.enable_cost()
        js=JointState(position=positions,velocity=mg.tensor_args.to_device(v[start:start+256]).unsqueeze(0),
                      acceleration=mg.tensor_args.to_device(a[start:start+256]).unsqueeze(0),
                      jerk=torch.zeros_like(positions),joint_names=planner.active_joints_name)
        state=mg.compute_kinematics(js)
        metrics=mg.rollout_fn.constraint_fn(state,use_batch_env=False)
        if not bool(metrics.feasible.all().item()):
            terms={'bounds':mg.rollout_fn.bound_constraint(state.state_seq),
                   'world':mg.rollout_fn.primitive_collision_constraint(state.robot_spheres),
                   'self':mg.rollout_fn.robot_self_collision_constraint(state.robot_spheres)}
            costs={k:float(value.max().item()) for k,value in terms.items()}
            raise RuntimeError(f'Retimed spline rejected at samples {start}:{start+len(positions[0])}: {costs}')
        if hold_orientation:
            state=mg.compute_kinematics(JointState.from_position(positions.squeeze(0),joint_names=planner.active_joints_name))
            errors=2*torch.acos(torch.sum(state.ee_quat_seq*goal.quaternion,dim=-1).abs().clamp(0,1))
            if float(errors.max().item())>.05:raise ValueError('Constrained insertion rotates tool by >0.05 rad')
    return {'status':'Success','position':q,'velocity':v,'retiming':{
        'method':'TOPPRA_geometry_then_timing_with_full_250Hz_collision_recheck',
        'velocity_limit_rad_s':velocity,'acceleration_limit_rad_s2':acceleration,
        'duration_s':(len(q)-1)*dt,'time_scale':scale,'peak_velocity_rad_s':float(np.abs(v).max()),
        'peak_acceleration_rad_s2':float(np.abs(a).max()),'collision_checked_samples':len(q),
        'endpoint_position_error_m':pos_error,'endpoint_rotation_error_rad':rot_error,'jerk_limit_certified':False,'real_hardware_execution':False}}
