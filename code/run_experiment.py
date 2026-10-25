"""Run fixed validation/test subsets. Never discard failures or replace cases."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
from ac_physics import validate
from ac_solver import solve, SETTINGS
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--limit',type=int);args=p.parse_args()
 np.random.seed(42)
 out=ROOT/'results'/f'{args.split}_results.jsonl'
 if out.exists(): raise RuntimeError(f'Refusing to overwrite {out}')
 out.parent.mkdir(parents=True,exist_ok=True)
 # Prime dynamic solver libraries independently of any measured scenario.
 import casadi as ca
 z=ca.MX.sym('z');f=ca.nlpsol('warmup','ipopt',{'x':z,'f':z*z},{'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes'});f(x0=1)
 protocol={'settings':SETTINGS,'split':args.split,'seed':42,'platform':platform.platform(),'python':sys.version,'casadi':ca.__version__,
           'thread_env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
           'scope':'One CPU process; existing seed42 checkpoints; no training; all methods built per scenario; model initialization and disk I/O excluded from online timings.',
           'method_order':'Random permutation per case, numpy RNG seed42',
           'distance':'One quarter of mean squared normalized deviations in each of vm,va,pg,qg_bus. Vm range floor .01, Pg/Qg range floor 1 MW/MVar, angle scale 1 rad.',
           'files_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list((ROOT/'code').glob('*.py'))+list((ROOT/'data').glob('cases_*.jsonl'))}}
 (ROOT/'results'/f'{args.split}_protocol.json').write_text(json.dumps(protocol,indent=2))
 rng=np.random.default_rng(42)
 with out.open('w') as h:
  for grid in ['ieee14','ieee30']:
   rows=[json.loads(x) for x in (ROOT/'data'/f'cases_{grid}.jsonl').read_text().splitlines()]
   rows=[r for r in rows if r['split']==args.split]
   if args.limit: rows=rows[:args.limit]
   for idx,row in enumerate(rows):
    case,pred=row['case'],row['prediction']
    t=time.perf_counter();raw=validate(case,pred);check_s=time.perf_counter()-t
    record={k:row[k] for k in ['case_id','grid','split','scenario_id','sample_order','inference_time_s','model_forward_s','preprocessing_s','postprocessing_s']}
    record['reference_validation']=validate(case,row['reference']);record['methods']={
       'raw':{'solution':pred,'validation':raw,'validation_s':check_s,'online_s':row['inference_time_s']+check_s,'distance_objective':0.}}
    methods=rng.permutation(['repair','cold_opf','warm_opf']).tolist();record['method_order']=methods
    for method in methods:
     try:
      result=solve(case,pred,method)
      result['online_s']=result['optimization_pipeline_s']+(row['inference_time_s']+check_s if method!='cold_opf' else 0.)
      result['solver_pipeline_without_build_s']=result['solve_s']+result['extract_s']+result['validation_s']+(row['inference_time_s']+check_s if method!='cold_opf' else 0.)
      record['methods'][method]=result
     except Exception as e:
      record['methods'][method]={'error':repr(e),'validation':{'feasible':False},'solver_success':False}
    h.write(json.dumps(record,allow_nan=True)+'\n');h.flush()
    print(grid,args.split,idx+1,'/'.join(f'{m}:{int(r["validation"]["feasible"])}' for m,r in record['methods'].items()),flush=True)
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
