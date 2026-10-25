"""Publication figures and a machine-readable summary from frozen test outcomes."""
import json, math
from pathlib import Path
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import networkx as nx
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (ROOT/'results/test_results.jsonl').read_text().splitlines()]
cases={r['case_id']:r for g in ['ieee14','ieee30'] for r in [json.loads(x) for x in (ROOT/'data'/f'cases_{g}.jsonl').read_text().splitlines()]}
GRIDS=['ieee14','ieee30']; GN={'ieee14':'IEEE14','ieee30':'IEEE30'}
METHODS=['raw','repair','cold_opf','warm_opf']; NAMES={'raw':'Raw GENCO','repair':'GENCO + repair','cold_opf':'Cold AC-OPF','warm_opf':'GENCO + AC-OPF'}
COLORS={'raw':'#9C677D','repair':'#137F82','cold_opf':'#536179','warm_opf':'#DA9A38'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'axes.grid':True,'grid.alpha':.14})
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=210,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def rs(g):return [r for r in rows if r['grid']==g]
def costgap(r,m):return 100*(r['methods'][m]['validation']['cost']/r['reference_validation']['cost']-1)
def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def stats(a):
 a=np.asarray(a);return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'min':float(a.min()),'max':float(a.max())}
summary={'seed':42,'n_test_per_grid':50,'tolerance':1e-5,'adjustment_rms_units':{'pg':'MW','qg_bus':'MVar, active-generator buses only','vm':'0.001 p.u.','va':'degrees'},'cost_metric':'Signed relative difference to source local AC-OPF reference; not certified global optimality gap.','grids':{}}
for g in GRIDS:
 rr=rs(g); summary['grids'][g]={}
 for m in METHODS:
  mm=[r['methods'][m] for r in rr];ok=sum(x['validation']['feasible'] for x in mm)
  summary['grids'][g][m]={'feasible_n':ok,'n':len(mm),'wilson_95':wilson(ok,len(mm)),
    'cost_difference_percent':stats([costgap(r,m) for r in rr]),'online_ms_first_pass':stats([1000*x['online_s'] for x in mm]),
    'max_violation_normalized':stats([x['validation']['overall_max_normalized'] for x in mm]),
    'status_counts':{k:sum(x.get('solver_status','not_applicable')==k for x in mm) for k in sorted(set(x.get('solver_status','not_applicable') for x in mm))}}
( ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
# 1: Primary result and near-feasible predictions.
fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
x=np.arange(2);w=.19
for i,m in enumerate(METHODS):
 val=[summary['grids'][g][m]['feasible_n'] for g in GRIDS]
 ax[0].bar(x+(i-1.5)*w,np.array(val)*2,w,label=NAMES[m],color=COLORS[m])
 for xx,v in zip(x+(i-1.5)*w,val):ax[0].text(xx,2*v+2,f'{v}/50',ha='center',va='bottom',fontsize=8)
ax[0].set(xticks=x,xticklabels=['IEEE14','IEEE30'],ylim=(0,115),ylabel='All constraints passed (%)',title='Independent AC feasibility')
ax[0].legend(loc='upper center',ncol=2,fontsize=8,bbox_to_anchor=(.5,-.12))
for g,marker in zip(GRIDS,['o','s']):
 rr=rs(g);xx=[r['methods']['raw']['validation']['overall_max_normalized'] for r in rr];yy=[r['methods']['repair']['validation']['overall_max_normalized'] for r in rr]
 ax[1].scatter(xx,yy,s=32,alpha=.75,marker=marker,label=GN[g],color='#137F82' if g=='ieee14' else '#516FA0')
ax[1].axvline(1e-5,color='#B64646',ls='--',lw=1);ax[1].axhline(1e-5,color='#B64646',ls='--',lw=1)
ax[1].set(xscale='log',yscale='log',xlabel='Raw maximum normalized violation',ylabel='Repaired maximum normalized violation',title='Small corrections restore feasibility')
ax[1].legend();ax[1].text(.03,.88,'Dashed lines: acceptance threshold',transform=ax[1].transAxes,va='top',fontsize=9)
fig.suptitle('GENCO AC feasibility restoration | 100 held-out scenarios',fontsize=15,fontweight='bold');save(fig,'01_feasibility_overview')
# 2: Every scenario and every family of constraints.
keys=['p_balance','q_balance','p_bounds','q_bounds','voltage_bounds','thermal_from','thermal_to','angle_bounds','reference_angle'];labels=['P balance','Q balance','P bounds','Q bounds','Voltage','Thermal: from','Thermal: to','Angle limits','Reference angle']
fig,axs=plt.subplots(2,2,figsize=(13,6.5),layout='constrained')
for i,g in enumerate(GRIDS):
 for j,m in enumerate(['raw','repair']):
  a=np.array([[r['methods'][m]['validation']['max_violation_normalized'][k] for r in rs(g)] for k in keys]);z=np.log10(np.maximum(a,1e-10)/1e-5)
  im=axs[i,j].imshow(z,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=3),interpolation='nearest');axs[i,j].grid(False)
  axs[i,j].set(title=f'{GN[g]} | {NAMES[m]}',yticks=range(len(keys)),yticklabels=labels,xticks=[0,9,19,29,39,49],xticklabels=[1,10,20,30,40,50],xlabel='Frozen test scenario order')
c=fig.colorbar(im,ax=axs,shrink=.85,pad=.02);c.set_label('log10(violation / tolerance); blue passes, red fails')
fig.suptitle('Audit every constraint family, not just an average residual',fontsize=15,fontweight='bold');save(fig,'02_constraint_heatmap')
# 3: Sensitivity uses precomputed independent residuals, not solver retraining.
fig,axs=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained');tols=np.logspace(-9,-2,281)
for ax,g in zip(axs,GRIDS):
 for m in METHODS:
  v=np.array([r['methods'][m]['validation']['overall_max_normalized'] for r in rs(g)])
  ax.plot(tols,[100*np.mean(v<=t) for t in tols],label=NAMES[m],color=COLORS[m],lw=2,ls='--' if m=='warm_opf' else '-')
 ax.axvline(1e-5,color='#333333',ls=':',lw=1.3);ax.set(xscale='log',ylim=(-3,104),xlabel='Normalized feasibility tolerance',ylabel='Joint feasibility (%)',title=GN[g]);ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Strict checks expose the difference between prediction and feasibility',fontsize=14,fontweight='bold');save(fig,'03_tolerance_sensitivity')
# 4: Per-scenario RMS changes in engineering units.
fig,axs=plt.subplots(2,2,figsize=(11.5,7),layout='constrained')
for ax,key,scale,label in zip(axs.flat,['pg','qg_bus','vm','va'],[1,1,1000,180/np.pi],['Active generation RMS adjustment (MW)','Reactive generation RMS adjustment (MVar)','Voltage magnitude RMS adjustment (0.001 p.u.)','Voltage angle RMS adjustment (degrees)']):
 for g,color in zip(GRIDS,['#137F82','#516FA0']):
  vals=[]
  for r in rs(g):
   delta=(np.array(r['methods']['repair']['solution'][key])-r['methods']['raw']['solution'][key])*scale
   if key=='qg_bus':delta=delta[np.unique(cases[r['case_id']]['case']['gen_bus'])]
   vals.append(float(np.sqrt(np.mean(delta**2))))
  vals=np.sort(vals);ax.step(vals,np.arange(1,51)/50,where='post',color=color,lw=2,label=GN[g]);summary['grids'][g].setdefault('adjustment_rms',{})[key]=stats(vals)
 ax.set(xlabel=label,ylabel='Fraction of test scenarios',ylim=(0,1.02));ax.legend(loc='lower right')
fig.suptitle('Repair preserves the prediction while enforcing the full AC model',fontsize=14,fontweight='bold');save(fig,'04_adjustment')
# 5: All feasible objectives, and raw-to-repair cost change.
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for g,color,marker in zip(GRIDS,['#137F82','#516FA0'],['o','s']):
 rr=rs(g);v=np.sort([costgap(r,'repair') for r in rr]);axs[0].step(v,np.arange(1,51)/50,where='post',color=color,label=f'{GN[g]} repair',lw=2)
 axs[1].scatter([costgap(r,'raw') for r in rr],[costgap(r,'repair') for r in rr],label=GN[g],s=34,color=color,marker=marker,alpha=.75)
axs[0].axvline(0,color='#536179',ls='--',label='AC-OPF baselines: near 0%');axs[0].set(xlabel='Repair cost difference to source reference (%)',ylabel='Fraction of test scenarios',ylim=(0,1.02),title='Feasible repair has low reference cost difference');axs[0].legend(loc='lower right',fontsize=8)
axs[1].plot([-.2,.55],[-.2,.55],color='#777777',ls=':',lw=1);axs[1].set(xlabel='Raw predicted cost difference (%) — mostly infeasible',ylabel='Feasible repaired cost difference (%)',title='Feasibility can require an increase in cost');axs[1].legend()
fig.suptitle('Reference objective difference is not a certified global optimality gap',fontsize=14,fontweight='bold');save(fig,'05_cost')
# 7: Topology and voltage changes for a predeclared representative case.
fig,axs=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
reps={}
for ax,g in zip(axs,GRIDS):
 r=max(rs(g),key=lambda r:r['methods']['raw']['validation']['overall_max_normalized']);reps[g]=r['case_id'];c=cases[r['case_id']]['case'];G=nx.Graph();G.add_nodes_from(range(c['nb']));G.add_edges_from(zip(c['branch']['from'],c['branch']['to']));pos=nx.spring_layout(G,seed=42,iterations=150)
 delta=1000*(np.array(r['methods']['repair']['solution']['vm'])-r['methods']['raw']['solution']['vm']);lim=max(abs(delta).max(),.01)
 nx.draw_networkx_edges(G,pos,ax=ax,edge_color='#A8B3C0',width=1.6)
 nodes=nx.draw_networkx_nodes(G,pos,ax=ax,node_color=delta,cmap='RdBu_r',vmin=-lim,vmax=lim,node_size=440,edgecolors='#243545',linewidths=.6)
 nx.draw_networkx_labels(G,pos,{i:str(i+1) for i in G},font_size=8,ax=ax)
 gen=np.unique(c['gen_bus']);nx.draw_networkx_nodes(G,pos,nodelist=gen.tolist(),node_color='none',node_size=580,edgecolors='#DA9A38',linewidths=2,ax=ax)
 ax.set_title(f'{GN[g]} | scenario {r["scenario_id"]}\nLargest raw violation in frozen test subset',fontsize=11);ax.axis('off');fig.colorbar(nodes,ax=ax,shrink=.65,label='Voltage correction (0.001 p.u.)')
fig.suptitle('Where the repair changes voltage | gold rings mark generator buses',fontsize=14,fontweight='bold');save(fig,'07_network_case')
summary['representative_case_selection']='Largest raw normalized violation per grid, selected without repaired outcomes';summary['representative_cases']=reps
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
print('CREATED 6 figure groups and summary.json; runtime figure generated separately')
