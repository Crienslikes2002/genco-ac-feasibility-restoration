"""Repeated sequential CPU end-to-end benchmark; no changes to quality cases/results."""
from __future__ import annotations
import os
# Before importing NumPy, Torch or CasADi, keep numerical-solver BLAS single threaded.
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'vendor/gridfm_graphkit'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
sys.path.insert(0,str(REPO))
import numpy as np
import torch,yaml,casadi as ca
from torch_geometric.data import HeteroData
import gridfm_graphkit
from gridfm_graphkit.io.param_handler import NestedNamespace,load_model
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches,RemoveInactiveGenerators,ApplyMasking
from gridfm_graphkit.datasets.masking import AddOPFHeteroMask
from ac_physics import validate
from ac_solver import solve,SETTINGS

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def infer(raw,model,norm,args):
 t=time.perf_counter();g=raw.clone();norm.transform(g);g=RemoveInactiveBranches()(g);g=RemoveInactiveGenerators()(g);g=AddOPFHeteroMask()(g);g=ApplyMasking(args)(g);t1=time.perf_counter()
 with torch.no_grad():out=model(g.x_dict,g.edge_index_dict,g.edge_attr_dict,g.mask_dict)
 t2=time.perf_counter();norm.inverse_output(out,g)
 pred={'vm':out['bus'][:,0].tolist(),'va':out['bus'][:,1].tolist(),'pg':out['gen'][:,0].tolist(),'qg_bus':out['bus'][:,3].tolist()};t3=time.perf_counter()
 return pred,{'preprocessing_s':t1-t,'model_forward_s':t2-t1,'postprocessing_s':t3-t2,'inference_s':t3-t}

def compact_validation(v):
 return {k:v[k] for k in ['feasible','finite','overall_max_normalized','cost','max_violation','max_violation_normalized']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=5);a=ap.parse_args()
 out=ROOT/'results/runtime_repeats.jsonl';proto=ROOT/'results/runtime_protocol.json'
 if out.exists():raise FileExistsError(f'Refusing overwrite: {out}')
 out.parent.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
 # Explicitly limit all loaded BLAS implementations; leave Torch's OpenMP at two threads.
 pool_info=None;pool_controller=None
 try:
  from threadpoolctl import threadpool_limits,threadpool_info
  pool_controller=threadpool_limits(limits=1,user_api='blas');pool_info=threadpool_info()
 except ImportError:pass
 models={};norms={};configs={};rawgraphs={};rows_by_grid={};init={}
 for grid in ['ieee14','ieee30']:
  allrows=[json.loads(s) for s in (ROOT/'data'/f'cases_{grid}.jsonl').read_text().splitlines()]
  rows_by_grid[grid]=[r for r in allrows if r['split']=='test'];assert len(rows_by_grid[grid])==50
  # Disk load and graph construction are common prepared inputs, outside all online timers.
  for r in allrows:rawgraphs[r['case_id']]=HeteroData.from_dict(torch.load(ROOT/r['raw_graph'],weights_only=True))
  t=time.perf_counter();md=ROOT/'models'/grid;args=NestedNamespace(**yaml.safe_load((md/'config.yaml').read_text()));configs[grid]=args
  norm=HeteroDataMVANormalizer(args);stats=torch.load(md/'normalizer_stats.pt',weights_only=True,map_location='cpu');norm.fit_from_dict(stats[args.data.networks[0]]);norms[grid]=norm
  model=load_model(args).cpu().eval();state=torch.load(md/'best_model_state_dict.pt',weights_only=True,map_location='cpu');model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True);models[grid]=model
  loaded=time.perf_counter();val=next(r for r in allrows if r['split']=='val');infer(rawgraphs[val['case_id']],model,norm,args);warm=time.perf_counter()
  init[grid]={'load_construct_s':loaded-t,'one_validation_warmup_s':warm-loaded,'warmup_case_id':val['case_id'],'checkpoint_sha256':digest(md/'best_model_state_dict.pt')}
 t=time.perf_counter();z=ca.MX.sym('z');warm=ca.nlpsol('prime','ipopt',{'x':z,'f':z*z},{'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes'});warm(x0=1);ipopt_warm=time.perf_counter()-t
 protocol={'status':'RUNNING','seed':42,'repeats_planned':a.repeats,'cases_per_repeat':100,'n_unique_cases':100,'grids':['ieee14','ieee30'],'split':'test','inference_device':'cpu','torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'numpy':np.__version__,'torch':torch.__version__,'casadi':ca.__version__,'platform':platform.platform(),'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},'threadpools':pool_info,'initialization':init,'ipopt_library_warmup_s':ipopt_warm,'settings':SETTINGS,'method_order':'Independent per-case NumPy default_rng(42).permutation of repair,cold_opf,warm_opf; one continuous seeded stream across repeats and grids. Fixed selected-case order within grid.','execution':'Single process, sequential; no parallel solver calls. Raw graph tensors and canonical physical cases preloaded; each case/repeat redoes actual normalized masked GENCO inference and fresh solver model construction. No cached predictions or cached NLP solvers used.','time_contract':{'raw_online_s':'Fresh inference preprocessing + forward + output denormalization/list conversion + independent raw check.','repair_online_s':'raw_online_s + full ac_solver.solve wall time for distance repair.','warm_opf_online_s':'raw_online_s + full ac_solver.solve wall time for cost minimization initialized at GENCO.','cold_opf_online_s':'Full ac_solver.solve wall time, including fresh model construction, initial guess preparation, IPOPT solve, extraction and independent final check; no GENCO or raw-check charge.','solve_call_wall_s':'External wall around complete solve() call; includes distance diagnostic/solver.stats and Python return overhead beyond optimization_pipeline_s.','internal_stages':'build_s,solve_s,extract_s,validation_s retained from identical ac_solver implementation.','excluded':'Initial case parsing/disk I/O, model initialization, one model warmup per grid, dummy IPOPT priming, serialization/output I/O, hashes, historical training.','difference_from_initial_inference_timer':'Repeated benchmark omits physical graph clone/inverse_transform used only to export canonical metadata; that common preparation is precomputed. Original cases and initial inference records remain unchanged.'},'failure_policy':'Keep every record and exception; no discarding cases, retries or replacing random selections. Differences against original saved predictions/costs explicitly reported.','files_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'code/benchmark_runtime.py',ROOT/'code/ac_solver.py',ROOT/'code/ac_physics.py',ROOT/'data/frozen_selection.json',ROOT/'data/cases_ieee14.jsonl',ROOT/'data/cases_ieee30.jsonl']}}
 dump(proto,protocol);print('BENCHMARK_STARTED',flush=True)
 rng=np.random.default_rng(42);fails=0;prediction_diff_max=0.;max_vs_main_cost=0.;totalstart=time.perf_counter();mainrows={}
 mainpath=ROOT/'results/test_results.jsonl'
 if mainpath.exists():mainrows={r['case_id']:r for r in map(json.loads,mainpath.read_text().splitlines())}
 with out.open('w') as fh:
  for repeat in range(a.repeats):
   for grid,rows in rows_by_grid.items():
    for row in rows:
     pred,times=infer(rawgraphs[row['case_id']],models[grid],norms[grid],configs[grid]);t=time.perf_counter();rawv=validate(row['case'],pred,tolerance=SETTINGS['acceptance_tolerance']);raw_check=time.perf_counter()-t
     pdiff=max(float(np.max(np.abs(np.asarray(pred[k])-row['prediction'][k]))) for k in pred);prediction_diff_max=max(prediction_diff_max,pdiff)
     record={'repeat':repeat,'case_id':row['case_id'],'grid':grid,'scenario_id':row['scenario_id'],'timing':times,'raw_check_s':raw_check,'raw_online_s':times['inference_s']+raw_check,'raw_validation':compact_validation(rawv),'max_abs_prediction_difference_from_original':pdiff,'method_order':rng.permutation(['repair','cold_opf','warm_opf']).tolist(),'methods':{}}
     for method in record['method_order']:
      t=time.perf_counter()
      try:
       ans=solve(row['case'],pred,method);wall=time.perf_counter()-t
       z={k:ans[k] for k in ['solver_status','solver_success','iterations','build_s','solve_s','extract_s','validation_s','optimization_pipeline_s','distance_objective','final_inf_pr','final_inf_du']}
       z.update({'solve_call_wall_s':wall,'online_s':wall+(record['raw_online_s'] if method!='cold_opf' else 0.),'validation':compact_validation(ans['validation'])})
       main=mainrows.get(row['case_id'],{}).get('methods',{}).get(method)
       if main and 'validation' in main and 'cost' in main['validation']:
        diff=abs(ans['validation']['cost']-main['validation']['cost']);z['absolute_cost_difference_from_primary_run']=diff;max_vs_main_cost=max(max_vs_main_cost,diff)
       if not ans['validation']['feasible'] or not ans['solver_success']:fails+=1
      except Exception as e:
       fails+=1;z={'error':repr(e),'solve_call_wall_s':time.perf_counter()-t,'solver_success':False,'validation':{'feasible':False}}
      record['methods'][method]=z
     fh.write(json.dumps(record,allow_nan=True)+'\n');fh.flush()
    print('BENCHMARK_PROGRESS',repeat+1,grid,'elapsed_s',round(time.perf_counter()-totalstart,2),'solver_failures',fails,flush=True)
 protocol.update({'status':'COMPLETE','repeats_completed':a.repeats,'records':a.repeats*100,'solver_calls':a.repeats*300,'solver_failure_count':fails,'max_abs_prediction_difference_from_original':prediction_diff_max,'max_abs_cost_difference_from_primary_run':max_vs_main_cost,'benchmark_loop_wall_s':time.perf_counter()-totalstart,'output_sha256':digest(out)})
 dump(proto,protocol);print('BENCHMARK_COMPLETE',protocol['records'],protocol['benchmark_loop_wall_s'],fails,prediction_diff_max,flush=True)
if __name__=='__main__':main()
