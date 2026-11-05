"""Report command bounds and measured motion separately; generate inspectable plots."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    joints=np.array([0,1,2,3,4,5,7,8,9,10,11,12]);reports=[]
    for file in sorted(args.results.glob('seed_*_trace.npz')):
        arrays=np.load(file);physical=arrays['physics_trace'];control=arrays['control_trace']
        dt=np.median(np.diff(physical[:,0]));command_v=physical[:,15:29][:,joints];measured_v=physical[:,43:57][:,joints]
        row={'file':file.name,'max_command_joint_speed_rad_s':float(np.abs(command_v).max()),
             'max_command_joint_acceleration_rad_s2':float(np.abs(np.diff(command_v,axis=0)/dt).max()),
             'max_measured_joint_speed_rad_s':float(np.abs(measured_v).max()),
             'p99_measured_joint_speed_rad_s':float(np.quantile(np.abs(measured_v),.99)),
             'max_raw_target_step_rad':float(np.abs(np.diff(control[:,29:43][:,joints],axis=0)).max())}
        reports.append(row)
        if file.name!='seed_0_trace.npz':continue
        fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        t=physical[:,0];ct=control[:,0]/25
        ax=axes[0,0];ax.plot(t,np.max(np.abs(command_v),axis=1),label='Command');ax.plot(t,np.max(np.abs(measured_v),axis=1),label='Measured',alpha=.7)
        ax.set(title='Maximum joint speed across both arms',ylabel='rad/s');ax.legend()
        ax=axes[0,1];ax.plot(t,np.max(np.abs(physical[:,1:15][:,joints]-physical[:,29:43][:,joints]),axis=1))
        ax.set(title='Maximum command tracking error',ylabel='rad')
        ax=axes[1,0]
        for label,i in [('Left',6),('Right',13)]:
            ax.plot(ct,control[:,1+i],label=label+' measured');ax.plot(ct,control[:,29+i],'--',label=label+' requested',alpha=.6)
        ax.set(title='Gripper aperture (0=closed, 1=open)',ylabel='Normalized aperture');ax.legend(ncol=2,fontsize=8)
        ax=axes[1,1];ax.plot(ct,control[:,59]);ax.set(title='Object height (evaluation only)',ylabel='World z (m)')
        for ax in axes.flat:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
        fig.savefig(args.results/'seed_0_control.png',dpi=160);plt.close(fig)
    (args.results/'control_metrics.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
