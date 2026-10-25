import json
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'figures').mkdir(exist_ok=True)
rows=[json.loads(x) for x in (ROOT/'results/runtime_repeats.jsonl').read_text().splitlines()]
METHODS=['raw','repair','cold_opf','warm_opf'];NAMES={'raw':'Raw GENCO','repair':'GENCO\n+ repair','cold_opf':'Cold\nAC-OPF','warm_opf':'GENCO\n+ AC-OPF'}
MC={'raw':'#9C677D','repair':'#137F82','cold_opf':'#536179','warm_opf':'#DA9A38'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.14,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
parts=['GENCO inference pipeline','Constraint checks','NLP construction','NLP solve','Other solver overhead'];colors=['#B4D9D5','#93ACCA','#77859D','#137F82','#D2D9E1']
summary={'repeats':5,'test_scenarios_per_grid':50,'primary_online_scope':'CPU batch-one inference/pre/post, raw check, model construction, solve, solution extraction, final independent check, diagnostics and return overhead; excludes common prepared case input, disk I/O, model startup, warmup, historical training.','grids':{}}
for col,g in enumerate(['ieee14','ieee30']):
 rr=[r for r in rows if r['grid']==g];table=np.zeros((5,4));sm={}
 for j,m in enumerate(METHODS):
  online=[];pv=[]
  for r in rr:
   if m=='raw':v=[r['timing']['inference_s'],r['raw_check_s'],0,0,0];tot=r['raw_online_s']
   else:
    z=r['methods'][m];isml=m!='cold_opf';v=[r['timing']['inference_s'] if isml else 0,(r['raw_check_s'] if isml else 0)+z['validation_s'],z['build_s'],z['solve_s'],z['solve_call_wall_s']-z['build_s']-z['solve_s']-z['validation_s']];tot=z['online_s']
   assert abs(sum(v)-tot)<1e-8
   pv.append(np.array(v)*1000);online.append(tot*1000)
  table[:,j]=np.mean(pv,axis=0)
  cm=[];pairs=[]
  for cid in sorted(set(r['case_id'] for r in rr)):
   cr=[r for r in rr if r['case_id']==cid]
   o=[r['raw_online_s']*1000 if m=='raw' else r['methods'][m]['online_s']*1000 for r in cr]
   cm.append(float(np.median(o)));pairs.append(float(np.median([r['methods']['cold_opf']['online_s']/(r['raw_online_s'] if m=='raw' else r['methods'][m]['online_s']) for r in cr])))
  means=[float(np.mean([r['raw_online_s']*1000 if m=='raw' else r['methods'][m]['online_s']*1000 for r in rr if r['repeat']==rep])) for rep in range(5)]
  sm[m]={'mean_online_ms':float(np.mean(online)),'median_online_ms':float(np.median(online)),'median_of_case_medians_ms':float(np.median(cm)),'p95_of_case_medians_ms':float(np.quantile(cm,.95)),'repeat_mean_ms':means,'sd_repeat_means_ms':float(np.std(means,ddof=1)),'mean_stages_ms':dict(zip(parts,table[:,j].tolist())),'median_cold_over_method_time':float(np.median(pairs)),'feasibility_status':'raw unchecked dispatch; mostly infeasible' if m=='raw' else 'all 250 repeated runs independently feasible'}
  axs[1,col].step(np.sort(cm),np.arange(1,51)/50,where='post',label=NAMES[m].replace('\n',' '),color=MC[m],lw=2)
 bottom=np.zeros(4)
 for i,(part,c) in enumerate(zip(parts,colors)):
  axs[0,col].bar(np.arange(4),table[i],bottom=bottom,color=c,label=part,width=.65);bottom+=table[i]
 for j,v in enumerate(bottom):axs[0,col].text(j,v+.4,f'{v:.2f}',ha='center',fontsize=9)
 axs[0,col].set(title=f'{g.upper()} | mean full pipeline time',xticks=np.arange(4),xticklabels=[NAMES[m] for m in METHODS],ylabel='Time per scenario (ms)',ylim=(0,35))
 axs[1,col].set(title='Case medians over five repeated runs',xlabel='Full pipeline wall time (ms)',ylabel='Fraction of test scenarios',ylim=(0,1.02));axs[1,col].legend(fontsize=8,loc='lower right')
 summary['grids'][g]=sm
fig.suptitle('Feasibility is restored, but the full pipeline is slower on these small grids',fontsize=14,fontweight='bold')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
fig.savefig(ROOT/'figures/06_runtime.png',dpi=210,bbox_inches='tight');fig.savefig(ROOT/'figures/06_runtime.pdf',bbox_inches='tight');plt.close(fig)
(ROOT/'results/runtime_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({g:{m:round(s['mean_online_ms'],3) for m,s in v.items()} for g,v in summary['grids'].items()},indent=2))
