"""Audit saved lab-rig evidence; build a Chinese report and offline video viewer."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np

LABELS={'reproduction':'训练布局复现','development':'开发集','heldout':'原定独立留出集','heldout_extended':'追加独立留出集','heldout_all':'全部独立留出集'}
COLORS={'blue':'蓝','red':'红','yellow':'黄'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    return [max(0,c-r),min(1,c+r)]


def audit(directory,case):
    seed=case['seed'];result=json.loads((directory/'results.json').read_text())[0]
    assert result['seed']==seed and result['k']==25
    assert result['robot_id']=='dual_jaka_zu7_lab_archive' and not result['complete_handover_tested']
    assert result['sampling']['noise_seed']==42
    with np.load(directory/f'seed_{seed}_trace.npz') as trace:
        physics=trace['physics_trace'];e=trace['grasp_evidence'];c=trace['control_trace']
        assert len(e)==len(physics)>1
        dt=float(np.median(np.diff(physics[:,0])))
        np.testing.assert_allclose(np.diff(physics[:,0]),.004,rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(physics[:,0],e[:,0],atol=1e-8)
        held=(e[:,1]>.05)&(e[:,2]>.08)&(e[:,2]<.90)&(e[:,3]==1)&(e[:,4]==0)
        np.testing.assert_array_equal(held,e[:,5].astype(bool))
        streak=longest=0
        for v in held:
            streak=streak+1 if v else 0;longest=max(longest,streak)
        success=longest*dt>=.6
        assert success==result['grasp_success']
        np.testing.assert_allclose(longest*dt,result['longest_verified_left_grasp_s'],atol=1e-7)
        close=np.flatnonzero(c[:,35]<.15)
        align_cm=float(np.linalg.norm(c[close[0],60:62]-c[close[0],57:59])*100) if len(close) else None
        moved_cm=float(np.max(np.linalg.norm(c[:,57:59]-c[0,57:59],axis=1))*100)
        before_close_move=float(np.linalg.norm(c[close[0],57:59]-c[0,57:59])*100) if len(close) else None
        joint_indices=np.array([0,1,2,3,4,5,7,8,9,10,11,12])
        speed=float(abs(physics[:,43:57][:,joint_indices]).max())
        lift_cm=float(e[:,1].max()*100)
    contacts=json.loads((directory/f'seed_{seed}_contacts.json').read_text())
    def forbidden(pair):
        arms=[n for n in pair if n.startswith(('left:','right:'))]
        if not arms:return False
        if len(arms)==2:return True
        other=next(n for n in pair if n not in arms)
        if arms[0].endswith(':Link_0') and other=='scene:table':return False
        return not (other=='scene:box' and arms[0] in ('left:finger_left','left:finger_right','left:finger_third'))
    bad=[x['pair'] for x in contacts if forbidden(x['pair'])]
    qualified=bool(success and result['failure_reason'] is None and not bad)
    failure=(result['failure_reason'] or {}).get('type')
    if not qualified and failure is None:failure='forbidden_contact' if bad else 'grasp_not_verified'
    return {'key':case['key'],'split':case['split'],'layout':case['layout'],'target':case['target'],'color':case['color'],'seed':seed,
            'qualified_success':qualified,'physical_grasp_success':bool(success),'failure':failure,'forbidden_contacts':bad,
            'max_lift_cm':lift_cm,'hold_seconds':longest*dt,'first_close_horizontal_error_cm':align_cm,
            'max_target_horizontal_movement_cm':moved_cm,'target_movement_before_first_close_cm':before_close_move,'non_target_contact':any(any('distractor_' in name for name in x['pair']) for x in contacts),'max_measured_joint_speed_rad_s':speed,
            'steps':result['steps'],'wall_seconds':result['wall_seconds'],'sim_seconds':result['sim_seconds'],
            'trace_sha256':sha(directory/f'seed_{seed}_trace.npz'),'review_video':bool(case['review_video']),
            'history_reused_layout':case['layout'] in (0,1000),'audit_passed':True}


def summarize(rows):
    groups={}
    for split in LABELS:
        group=[r for r in rows if r['split']==split or (split=='heldout_all' and r['split'].startswith('heldout'))]
        layouts=defaultdict(list)
        for r in group:layouts[r['layout']].append(r)
        complete={k:v for k,v in layouts.items() if len(v)==3}
        means=np.array([np.mean([r['qualified_success'] for r in v]) for v in complete.values()])
        ci=None
        if len(means)>=5 and np.ptp(means)>0:
            rng=np.random.default_rng(20261116)
            samples=rng.choice(means,(20000,len(means)),replace=True).mean(axis=1)
            ci=np.quantile(samples,[.025,.975]).tolist()
        by_color={}
        for color in COLORS:
            subset=[r for r in group if r['color']==color];k=sum(r['qualified_success'] for r in subset);n=len(subset)
            by_color[color]={'successes':k,'episodes':n,'wilson_95':wilson(k,n)}
        by_slot={}
        for slot in range(3):
            subset=[r for r in group if r['target']==slot]
            by_slot[str(slot)]={'successes':sum(r['qualified_success'] for r in subset),'episodes':len(subset)}
        groups[split]={'successes':sum(r['qualified_success'] for r in group),'episodes':len(group),
            'complete_layouts':len(complete),'partial_layouts':[k for k,v in layouts.items() if len(v)!=3],
            'complete_layout_mean':float(means.mean()) if len(means) else None,
            'layout_cluster_bootstrap_95':ci,'by_color':by_color,'by_position_slot':by_slot,'color_position':{color:{str(slot):{'successes':sum(r['qualified_success'] for r in group if r['color']==color and r['target']==slot),'episodes':sum(r['color']==color and r['target']==slot for r in group)} for slot in range(3)} for color in COLORS},
            'failures':dict(Counter(r['failure'] for r in group if not r['qualified_success'])),
            'episodes_with_forbidden_contact':sum(bool(r['forbidden_contacts']) for r in group),'episodes_with_non_target_contact':sum(r['non_target_contact'] for r in group)}
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extra-source',type=Path);args=p.parse_args()
    source=args.source;out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'videos').mkdir(exist_ok=True)
    sources=[source]+([args.extra_source] if args.extra_source else [])
    protocols=[json.loads((x/'protocol.json').read_text()) for x in sources]
    protocol=protocols[0]
    assert all(p['model_sha256']==protocol['model_sha256'] for p in protocols)
    all_cases=[dict(c,source_dir=str(src),extension=(src!=source)) for src,p in zip(sources,protocols) for c in p['cases']]
    assert len({c['key'] for c in all_cases})==len(all_cases)
    workers=[json.loads(x.read_text()) for src in sources for x in sorted(src.glob('worker_*.json'))]
    attempts={r['key']:r for w in workers for r in w['rollouts']}
    rows=[];errors=[];pending=[]
    for case in all_cases:
        attempt=attempts.get(case['key'])
        if attempt is None:pending.append(case['key']);continue
        if attempt['status']!='completed':errors.append(attempt);continue
        directory=Path(case['source_dir'])/case['key']
        row=audit(directory,case)
        if case['extension']:row['split']='heldout_extended'
        assert row['qualified_success']==attempt['qualified_grasp_success']
        seed=case['seed'];mp4=directory/f'seed_{seed}_multiview.mp4' if case['review_video'] else directory/f'seed_{seed}.mp4'
        if not mp4.exists():raise FileNotFoundError(mp4)
        video=out/'videos'/f"{case['key']}.mp4"
        if not video.exists() or video.stat().st_size!=mp4.stat().st_size:shutil.copy2(mp4,video)
        row['video']=str(video.relative_to(out));row['video_sha256']=sha(video)
        rows.append(row)
    groups=summarize(rows)
    summary={'created_at':datetime.now().astimezone().isoformat(),'checkpoint':'lab_pi05_three_targets_20261112/5744',
        'conditions':protocol['conditions'],'batch_conditions':[{'name':p['arguments']['name'],'conditions':p['conditions']} for p in protocols],'planned':len(all_cases),'completed':len(rows),'operational_errors':errors,'uncompleted':pending,
        'finished':all((src/'execution_finished.json').exists() for src in sources),'protocol_sha256':sha(source/'protocol.json'),
        'groups':groups,'rows':rows,'limitations':['Only simulated left-arm grasp/lift/0.6s hold; no full handover or real robot.',
        'Single frozen checkpoint and fixed policy noise42; not a model comparison or run-to-run robustness estimate.',
        'No per-layout scripted-expert feasibility control was run for these new heldout cases.',
        'Heldout layouts from same scene generator; not unseen object categories or lighting/camera domains.',
        'Three episodes share each layout; overall uncertainty uses layout-cluster bootstrap, colors use Wilson per independent layout.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps({'batches':protocols},ensure_ascii=False,indent=2)+'\n')
    fields=['split','layout','target','color','seed','qualified_success','physical_grasp_success','failure','max_lift_cm','hold_seconds','first_close_horizontal_error_cm','max_target_horizontal_movement_cm','target_movement_before_first_close_cm','non_target_contact','max_measured_joint_speed_rad_s','steps','wall_seconds','history_reused_layout','audit_passed','video']
    with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    md=['# 当前 π0.5 模型性能评测 · 2026-11-16','',f"更新：{summary['created_at']}。已完成 {len(rows)}/{len(all_cases)} 回合，运行异常 {len(errors)}，未完成 {len(pending)}。",'',
        '**固定模型：three_targets/5744；K=25、H=50、固定 noise42；无新增训练。**','',
        ('**当前结论：**全部独立留出已完成 '+str(groups['heldout_all']['episodes'])+' 回合，合格成功 '+str(groups['heldout_all']['successes'])+' 回合（'+format(groups['heldout_all']['successes']/groups['heldout_all']['episodes'],'.1%')+'）。模型具备部分抓取能力，但当前成功率不足以视为可靠操作策略。' if groups['heldout_all']['episodes'] else '独立留出尚未形成结果。'),'',
        '首批93回合使用两个worker，补充240回合在执行前根据显存与吞吐调整为三个worker；策略、控制器和判据不变。并发数与模型/代码哈希均保存在协议中。','', '测试任务为仿真左臂抓取—抬升—持物。合格要求：抬升超过5cm、测得夹爪开度8%–90%、至少2/3手指接触目标、右臂不接触目标，连续至少0.6秒，并且无终止异常及非预定接触。达到阈值即停止，不代表长时间持物、双臂交接或真机性能。','',
        '| 分组 | 合格成功/完成回合 | 成功比例 | 完整布局 | 按布局聚类的95%区间 |','|---|---:|---:|---:|---|']
    for split,g in groups.items():
        n=g['episodes'];k=g['successes'];ci=g['layout_cluster_bootstrap_95'];interval=f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else '样本或变异不足，不估计'
        md.append(f"| {LABELS[split]} | {k}/{n} | {k/n:.1%}"+f" | {g['complete_layouts']} | {interval} |" if n else f"| {LABELS[split]} | 0/0 | — | 0 | — |")
    md += ['', '原定独立留出集为布局2000–2019；依据吞吐预登记追加2020–2099，分别统计并汇总。两批在本次固定模型前未使用；本轮不依据该集调参。开发1000曾参与历史方案选择，开发1001–1009此前未参与该选择。训练布局0仅作复现，不能合并计算泛化成功率。', '',
        '![性能汇总](performance.png)','', '## 颜色与目标位置','', '| 分组 | 蓝 | 红 | 黄 | 位置0 | 位置1 | 位置2 |','|---|---:|---:|---:|---:|---:|---:|']
    for split,g in groups.items():
        values=[g['by_color'][c] for c in COLORS]+[g['by_position_slot'][str(i)] for i in range(3)]
        md.append('| '+LABELS[split]+' | '+' | '.join(f"{v['successes']}/{v['episodes']}" for v in values)+' |')
    md += ['', '### 全部独立留出：颜色 × 位置', '', '| 颜色 | 位置0 | 位置1 | 位置2 |', '|---|---:|---:|---:|']
    for color in COLORS:
        cells=groups['heldout_all']['color_position'][color]
        md.append('| '+COLORS[color]+' | '+' | '.join(f"{cells[str(i)]['successes']}/{cells[str(i)]['episodes']}" for i in range(3))+' |')
    md+=['','每个布局三种颜色各出现一次；target表示固定空间槽位，颜色会随布局置换。位置0约(-0.58,0.18)m、位置1约(-0.42,0.32)m、位置2约(-0.10,0.32)m，均有预定位置/尺寸/yaw变化。分色95% Wilson区间在summary.json，整体区间使用完整布局为单位的20000次bootstrap，避免把同布局三回合当完全独立样本。如果存在部分布局，总比例保留所有完成回合，区间仅对应完整布局；该子集均值见complete_layout_mean。该区间只反映本场景生成分布及固定噪声下的抽样不确定性。','','## 失败与完整性','']
    for split,g in groups.items():md.append(f"- {LABELS[split]}：{json.dumps(g['failures'],ensure_ascii=False)}；有非预定接触 {g['episodes_with_forbidden_contact']} 回合，其中非目标接触 {g['episodes_with_non_target_contact']} 回合。")
    md+=['',f'已完成 {len(rows)} 份逐250Hz物理证据复核，保存全部成功与失败。运行异常不悄悄删除，不作为普通策略失败；如有部分布局或时间截断，见summary.json的uncompleted及operational_errors。', '',
        '## 历史结果复现与解释边界','', '11月12日训练布局0为1/3、开发1000为2/3；本次同布局分别为1/3、1/3，蓝块开发案例未复现成功。已核对开发1000蓝块的首次输入：14维状态、三路RGB和固定噪声逐值相同，但首次动作最大绝对差约0.001895。差异原因尚未隔离，不能声称跨进程逐数值确定性，也不能仅据此认定失败根因。旧回合不并入本轮独立留出统计。完整输入核查见initial_observation_audit.json。','', '本轮没有对每个新留出布局另跑专家可达性对照；结果反映固定场景生成分布上的策略表现，尚不能把全部失败都归为单一学习缺陷。','', '当前模型在已有stage3/14854权重上，以布局0三条示教1170帧再优化5744步；5744不是从零训练的累计总步数。当前结果衡量这一固定检查点，不能作为扩数据或某个训练改动的因果收益。','', '## 查看与复现','', '[汇报版PDF](JAKA_pi05_性能评测报告.pdf) · [成功与失败Demo](demo_success_and_failure.mp4)（说明案例，不代表成功比例）。','', '打开 [index.html](index.html) 可按分组、颜色、成功/失败筛选全部录像；[results.csv](results.csv)为逐回合表，[summary.json](summary.json)含完整统计，[protocol.json](protocol.json)含预注册场景清单、固定参数和模型/代码哈希。', '',
        '高清多视角录像在执行前按布局ID固定选择；其余保留真实评测头部相机视频。视频为25Hz仿真时间，高清片头2秒/片尾3秒标记为静帧；不代表实际推理延迟。未进行模型优劣、训练改进、语言作用消融或Sim2Real结论验证。']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    payload=json.dumps({'groups':groups,'rows':rows},ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JAKA π0.5 当前模型性能评测</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f4f6fa;color:#172b3a;margin:0}main{max-width:1180px;margin:auto;padding:30px}h1{margin:0}small,.note{color:#536777}a{color:#126287}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.card,article{background:white;border-radius:12px;padding:18px;box-shadow:0 2px 10px #1231}.card strong{font-size:28px;display:block}select{font-size:16px;padding:8px;margin:8px 8px 18px 0}#videos{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}video{width:100%;background:#101820;max-height:420px}article h3{font-size:17px;margin:0 0 8px}.yes{color:#126b53}.no{color:#af442b}details{margin-top:8px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}@media(max-width:700px){.cards,#videos{grid-template-columns:1fr}main{padding:16px}}</style><main><h1>π0.5 当前模型性能评测</h1><p>2026-11-16 · 固定5744检查点 · K=25 / H=50 / noise42 · 已完成 COMPLETED / PLANNED</p><p class="note">仿真左臂抓取、抬升与连续0.6秒持物；所有成功与失败均展示。训练布局复现、开发集与独立留出集分别统计。</p><p><a href="JAKA_pi05_性能评测报告.pdf">汇报版 PDF</a> · <a href="REPORT.md">技术报告</a> · <a href="demo_success_and_failure.mp4">成功与失败 Demo</a> · <a href="results.csv">逐回合CSV</a> · <a href="summary.json">统计与审计</a> · <a href="protocol.json">固定评测协议</a></p><div class="cards" id="cards"></div><img src="performance.png" alt="成功率及颜色、目标位置分布" style="width:100%;border-radius:12px;margin-bottom:16px"><select id="split"><option value="">全部分组</option><option value="reproduction">训练布局复现</option><option value="development">开发集</option><option value="heldout">原定独立留出集</option><option value="heldout_extended">追加独立留出集</option></select><select id="color"><option value="">全部颜色</option><option value="blue">蓝</option><option value="red">红</option><option value="yellow">黄</option></select><select id="verdict"><option value="">全部结果</option><option value="yes">合格成功</option><option value="no">失败</option></select><span id="count"></span><div id="videos"></div><p class="note">独立留出来自同一场景生成器，不代表新物体类别或真机泛化。每布局三回合存在相关性；整体不确定性按布局计算。高清录像片头/片尾静帧不计入持物时长。</p></main><script>const D=PAYLOAD;const names={reproduction:'训练布局复现',development:'开发集',heldout:'原定独立留出集',heldout_extended:'追加独立留出集',heldout_all:'全部独立留出集'},colors={blue:'蓝',red:'红',yellow:'黄'};for(const [s,g] of Object.entries(D.groups)){let c=document.createElement('div');c.className='card';c.innerHTML=names[s]+'<strong>'+g.successes+' / '+g.episodes+'</strong>'+g.complete_layouts+' 个完整布局';document.getElementById('cards').append(c)}function render(){let s=document.getElementById('split').value,c=document.getElementById('color').value,v=document.getElementById('verdict').value;let rows=D.rows.filter(r=>(!s||r.split===s)&&(!c||r.color===c)&&(!v||(v==='yes')===r.qualified_success));document.getElementById('count').textContent=rows.length+' 个回合';let box=document.getElementById('videos');box.innerHTML='';for(const r of rows){let a=document.createElement('article');a.innerHTML='<h3>'+names[r.split]+' · 布局'+r.layout+' · '+colors[r.color]+' · 位置'+r.target+' <span class="'+(r.qualified_success?'yes':'no')+'">'+(r.qualified_success?'合格成功':'失败')+'</span></h3><video controls preload="none" src="'+r.video+'"></video><div>最大抬升 '+r.max_lift_cm.toFixed(2)+' cm · 合格持物 '+r.hold_seconds.toFixed(3)+' s</div><small>'+ (r.review_video?'高清多视角':'头部相机原始评测录像')+'</small><details><summary>证据</summary><pre></pre></details>';a.querySelector('pre').textContent=JSON.stringify({seed:r.seed,failure:r.failure,forbidden_contacts:r.forbidden_contacts,first_close_horizontal_error_cm:r.first_close_horizontal_error_cm,trace_sha256:r.trace_sha256},null,2);box.append(a)}}for(const id of ['split','color','verdict'])document.getElementById(id).addEventListener('change',render);render();</script></html>'''
    page=page.replace('COMPLETED',str(len(rows))).replace('PLANNED',str(len(all_cases))).replace('PAYLOAD',payload)
    (out/'index.html').write_text(page)
    print(json.dumps({'completed':len(rows),'errors':len(errors),'groups':{s:{k:g[k] for k in ['successes','episodes','complete_layouts']} for s,g in groups.items()}},ensure_ascii=False))

if __name__=='__main__':main()
