from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=Path('/home/lightwheel/Documents/Codex/2026-11-11/du-q/outputs/20261116/review'));root=p.parse_args().review
s=json.loads((root/'summary.json').read_text())
f=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':f.get_name(),'axes.unicode_minus':False,'font.size':11})
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
entries=[('development','开发集'),('heldout','原定留出'),('heldout_extended','追加留出'),('heldout_all','全部留出')]
labels=[];values=[];lower=[];upper=[];counts=[]
for key,label in entries:
 g=s['groups'][key];n=g['episodes'];k=g['successes']
 if not n:continue
 rate=k/n;ci=g['layout_cluster_bootstrap_95'] or [rate,rate]
 labels.append(label);values.append(100*rate);lower.append(max(0,100*(rate-ci[0])));upper.append(max(0,100*(ci[1]-rate)));counts.append(f'{k}/{n}')
ax=axes[0];bars=ax.bar(labels,values,color=['#4d7898','#419c9a','#88af8b','#23676b'][:len(labels)],yerr=[lower,upper],capsize=4)
for bar,cnt in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+4,cnt,ha='center')
ax.set(title='按布局统计的抓取效果',ylabel='合格成功比例（%）',ylim=(0,100));ax.tick_params(axis='x',rotation=15)
g=s['groups']['heldout_all']
for ax,field,keys,names,colors in [(axes[1],'by_color',['blue','red','yellow'],['蓝','红','黄'],['#316f9e','#bf5c4d','#cba64d']),(axes[2],'by_position_slot',['0','1','2'],['位置0','位置1','位置2'],['#476e83']*3)]:
 rates=[];texts=[]
 for key in keys:
  v=g[field][key];n=v['episodes'];k=v['successes'];rates.append(100*k/n if n else 0);texts.append(f'{k}/{n}')
 bars=ax.bar(names,rates,color=colors)
 for bar,t in zip(bars,texts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,t,ha='center')
 ax.set(title='独立留出：按颜色' if field=='by_color' else '独立留出：按目标位置',ylim=(0,100))
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle(f"π0.5 固定模型评测 · 已完成 {s['completed']}/{s['planned']} 回合",fontproperties=f,fontsize=16)
fig.supxlabel('仿真左臂抓取/抬升/0.6秒持物 · K25/H50/noise42 · 第一图误差线为完整布局聚类bootstrap 95%区间',fontproperties=f,fontsize=10)
fig.savefig(root/'performance.png',dpi=180)
fig.savefig(root/'performance.svg')
