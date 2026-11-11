"""Language-conditioned multi-object scenes for the archived Zu7 rig.

Object truth is used only by the scripted expert and evaluator, never the policy.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import MethodType
import numpy as np
import sapien
import transforms3d as t3d
from jaka_vla.identity import rig_urdf_hash


COLORS={'red':[.8,.04,.04],'blue':[.04,.12,.8],'yellow':[.85,.65,.02]}


def scene_spec(layout_id,target_index,split='train'):
    rng=np.random.default_rng(60000+layout_id)
    colors=list(COLORS);rng.shuffle(colors)
    positions=[[-.58,.18],[-.42,.32],[-.10,.32]]
    objects=[]
    for color,(x,y) in zip(colors,positions):
        half=[float(rng.uniform(.024,.030)),float(rng.uniform(.024,.030)),float(rng.uniform(.045,.060))]
        objects.append({'color':color,'half_size':half,'xyz':[float(x+rng.uniform(-.02,.02)),
                        float(y+rng.uniform(-.025,.025)),.74+half[2]+.002],
                        'yaw':float(rng.uniform(-.4,.4))})
    return {'version':2,'layout_id':layout_id,'target_index':target_index,'split':split,'objects':objects,
            'instruction':f'Use the left arm to grasp the {objects[target_index]["color"]} block, lift it, and hold it.',
            'scope':'left_grasp_lift_hold','camera_assumptions':'three simulated RGB cameras; wrist cameras not evidenced in archive'}


def install_scene(task,spec):
    from envs.utils import create_box, create_table
    # The planner uses the archive table. Use the identical tabletop in physics;
    # RoboTwin's default table has a different centre and width.
    table=json.loads(Path('/data/jaka-vla/manifests/lab_hardware_20261111.json').read_text())['world_table']
    def load_table(self,table_xy_bias=None,table_height=.74):
        self.table_xy_bias=table['position'][:2]
        self.wall_texture=self.table_texture=None
        self.wall=create_box(self.scene,sapien.Pose([0,2,1.5]),half_size=[3,.1,1.5],
                             color=[.55,.55,.55],name='wall',is_static=True)
        top=table['position'][2]+table['dims'][2]/2
        self.table=create_table(self.scene,sapien.Pose([*table['position'][:2],top],table['quaternion']),
                               length=table['dims'][0],width=table['dims'][1],height=top,
                               thickness=table['dims'][2],color=[.52,.52,.52],is_static=True)
    task.create_table_and_wall=MethodType(load_table,task)
    def load(self):
        self.scene.set_ambient_light([.2,.2,.2])
        for light in self.point_light_lst:light.set_color([.45,.45,.45])
        for light in self.direction_light_lst:light.set_color([.35,.35,.35])
        self.lab_objects=[]
        for i,obj in enumerate(spec['objects']):
            actor=create_box(scene=self,pose=sapien.Pose(obj['xyz'],t3d.euler.euler2quat(0,0,obj['yaw'])),
                half_size=obj['half_size'],color=COLORS[obj['color']],name='box' if i==spec['target_index'] else f'distractor_{i}')
            self.lab_objects.append(actor)
        self.box=self.lab_objects[spec['target_index']]
        self.target_box=create_box(scene=self,pose=sapien.Pose([.30,.30,.745]),half_size=[.05,.05,.005],
                                    color=[.2,.65,.25],name='target_box',is_static=True)
        self.block_middle_pose=[0,.2,1.,0,1,0,0]
    task.load_actors=MethodType(load,task)
    task.lab_scene_spec=deepcopy(spec)


def install_lab_grasp(task):
    from jaka_vla.short_grasp import joint_arc
    robot=task.robot;entity=robot.left_entity;active=entity.get_active_joints()
    indices=[active.index(j) for j in robot.left_arm_joints]
    old_choose=task.choose_grasp_pose;old_move=task.left_move_to_pose;queue=[]
    audit={'method':'lab_grasp_ik_then_reverse_cartesian_4_yaws','candidates':[],'executed_cached_paths':0}
    def choose(self,actor,arm_tag,pre_dis=.1,target_dis=0,contact_point_id=None):
        if str(arm_tag)!='left' or audit['candidates']:return old_choose(actor,arm_tag,pre_dis,target_dis,contact_point_id)
        initial=entity.get_qpos().copy();best=None
        center=actor.get_pose().p.copy();center[2]+=.035
        planner=robot.left_planner
        for yaw in (0,np.pi/2,np.pi,3*np.pi/2):
            x=np.array([0.,0.,-1.]);y=np.array([np.cos(yaw),np.sin(yaw),0.]);z=np.cross(x,y)
            rotation=np.stack([x,y,z],axis=1);quat=t3d.quaternions.mat2quat(rotation)
            grasp=np.r_[center-x*.12,quat].tolist();pre=grasp.copy();pre[2]+=.12
            row={'yaw':yaw,'pre_pose':pre,'grasp_pose':grasp};audit['candidates'].append(row)
            # All non-target objects remain obstacles. Target is excluded only during final insertion.
            target_local=planner.robot_origion_pose.inv()*actor.get_pose()
            size=np.array(task.lab_scene_spec['objects'][task.lab_scene_spec['target_index']]['half_size'])*2
            keepout={'dims':(size+np.array([.025,.025,.02])).tolist(),'pose':list(target_local.p)+list(target_local.q)}
            from jaka_vla.lab_cartesian import plan_grasp_pair
            try:
                first,second=plan_grasp_pair(planner,initial[indices],
                    robot._trans_from_gripper_to_endlink(pre,arm_tag='left'),
                    robot._trans_from_gripper_to_endlink(grasp,arm_tag='left'),keepout)
            except (ValueError,RuntimeError) as error:
                row.update(pre_status='Fail',grasp_status='Fail',diagnostic=str(error))
                print('GRASP_BRANCH_REJECTED',yaw,str(error),flush=True)
                continue
            after=initial.copy();after[indices]=first['position'][-1]
            row.update(pre_status='Success',grasp_status='Success',pre_retiming=first['retiming'],
                       grasp_retiming=second['retiming'],cartesian=second['cartesian'])
            cost=joint_arc(first['position'],initial[indices])+joint_arc(second['position'],after[indices])
            row['joint_arc_rad']=cost
            if best is None or cost<best[0]:best=(cost,pre,grasp,first,second,initial,after,len(audit['candidates'])-1)
        if best is None:raise RuntimeError('No collision-checked top-grasp candidate')
        _,pre,grasp,first,second,initial,after,selected=best;audit['selected_index']=selected
        queue.extend([(pre,None,first,initial),(grasp,[1,1,1,0,0,0],second,after)])
        return pre,grasp
    def move(self,pose,constraint_pose=None,**kwargs):
        if not queue:return old_move(pose,constraint_pose=constraint_pose,**kwargs)
        expected,constraint,result,start=queue.pop(0)
        error=float(np.abs(entity.get_qpos()[indices]-start[indices]).max())
        if not np.allclose(pose,expected,atol=1e-7) or constraint_pose!=constraint or error>.05:
            raise RuntimeError(f'Cached path start mismatch {error}')
        self.left_joint_path.append(deepcopy(result));audit['executed_cached_paths']+=1
        return deepcopy(result)
    task.choose_grasp_pose=MethodType(choose,task);task.left_move_to_pose=MethodType(move,task)
    # A distant visual background is also represented in the planning world.
    wall=robot.left_planner.robot_origion_pose.inv()*sapien.Pose([0,2,1.5])
    robot.left_planner.project_world['cuboid']['background_wall']={'dims':[6,.2,3],'pose':list(wall.p)+list(wall.q)}
    # Live poses captured once before execution; distractors are stationary expert obstacles.
    for i,actor in enumerate(task.lab_objects):
        if i==task.lab_scene_spec['target_index']:continue
        local=robot.left_planner.robot_origion_pose.inv()*actor.get_pose()
        dims=(np.array(task.lab_scene_spec['objects'][i]['half_size'])*2+.012).tolist()
        robot.left_planner.project_world['cuboid'][f'distractor_{i}']={'dims':dims,'pose':list(local.p)+list(local.q)}
    return audit
