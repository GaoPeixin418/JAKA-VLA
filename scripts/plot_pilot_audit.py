"""Render the saved offline audit with envs/robotwin/bin/python; no model inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/data/jaka-vla/results/offline_audit_20261105_v2')
a = np.load(root/'offline_arrays.npz')
r = json.loads((root/'report.json').read_text())
joints = [0,1,2,3,4,5,7,8,9,10,11,12]
fig, ax = plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
fig.suptitle('Pilot diagnosis: offline inference on training observations', fontsize=17, fontweight='bold')
std = np.rad2deg(a['repeats'][:,0,joints].std(axis=0))
ax[0,0].bar(range(12),std,color=['#3e6da8']*6+['#cf8a43']*6)
ax[0,0].set_xticks(range(12),[f'L{i}' for i in range(1,7)]+[f'R{i}' for i in range(1,7)])
ax[0,0].set_ylabel('First-target standard deviation (degrees)')
ax[0,0].set_title('Identical observation, 32 independent noise samples')
t=(np.arange(50)+1)/25
mu=np.rad2deg(a['repeats'][:,:,1].mean(axis=0)); sigma=np.rad2deg(a['repeats'][:,:,1].std(axis=0))
ax[0,1].plot(t,np.rad2deg(a['expert_initial'][:,1]),color='#234b33',label='Expert target')
ax[0,1].plot(t,mu,color='#3e6da8',label='Model mean')
ax[0,1].fill_between(t,mu-2*sigma,mu+2*sigma,alpha=.2,color='#3e6da8',label='Mean +/- 2 std (not CI)')
ax[0,1].set(xlabel='Future target time (s)',ylabel='Left joint 2 target (degrees)',title='One predicted chunk: training episode 0, frame 0')
ax[0,1].legend(fontsize=9)
pred=np.clip(a['predictions'][:,:,0,6],0,1); gt=a['targets'][:,0,6]; x=np.arange(16)
ax[1,0].plot(x,gt,'o-',color='#234b33',label='Expert first target')
ax[1,0].errorbar(x,pred.mean(axis=1),yerr=pred.std(axis=1),fmt='o',color='#3e6da8',capsize=3,label='Model mean +/- 1 std')
ax[1,0].set_xticks(x,[f'{ep}/{frame}' for ep,frame in a['indices']],rotation=60,ha='right',fontsize=8)
ax[1,0].set(xlabel='Training episode / frame',ylabel='Left gripper target (0 closed, 1 open)',title='Gripper timing: 16 observations, 4 samples each',ylim=(-.1,1.1))
ax[1,0].legend(fontsize=8)
ks=[4,8,16]; x=np.arange(3)
model=[np.rad2deg(r['expert_observation_chunk_boundaries'][str(k)]['predicted_boundary_joint_delta_rad']['mean']) for k in ks]
expert=[np.rad2deg(r['expert_observation_chunk_boundaries'][str(k)]['expert_boundary_joint_delta_rad']['mean']) for k in ks]
ax[1,1].bar(x-.18,expert,.36,label='Expert adjacent target',color='#43805c')
ax[1,1].bar(x+.18,model,.36,label='New / old model chunk boundary',color='#3e6da8')
ax[1,1].set_xticks(x,[f'K={k}' for k in ks])
ax[1,1].set(ylabel='Mean absolute joint-target jump (degrees)',title='Chunk boundaries on expert observation sequence')
ax[1,1].legend(fontsize=8)
for panel in ax.flat:
    panel.spines[['right','top']].set_visible(False)
    panel.grid(axis='y',alpha=.2)
fig.text(.5,-.015,'Diagnostic samples only. These are action targets, not measured closed-loop robot jitter. No new training or rollouts.',ha='center',fontsize=10)
fig.savefig(root/'diagnostic_summary.png',dpi=160,bbox_inches='tight')
fig.savefig(root/'diagnostic_summary.pdf',bbox_inches='tight')
print(root/'diagnostic_summary.png')
