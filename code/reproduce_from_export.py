"""Replay AC repair/OPF from the exported frozen cases, without ML dependencies.
Usage: python code/reproduce_from_export.py [--split test] [--output-dir results/reproduced]
This reuses saved GENCO predictions; it DOES NOT remeasure neural inference.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
from ac_physics import validate
from ac_solver import solve, SETTINGS
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data',help='Directory containing cases_ieee14.jsonl and cases_ieee30.jsonl');p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'reproduced');a=p.parse_args()
 inputs=[a.data_dir.resolve()/f'cases_{grid}.jsonl' for grid in ['ieee14','ieee30']]
 for path in inputs:
  if not path.is_file():p.error(f'Missing input {path}; supply case exports with --data-dir.')
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 dest=out/f'{a.split}_results.jsonl'
 if dest.exists():raise FileExistsError(f'Choose a new --output-dir; refusing to overwrite {dest}')
 total=0;passed={m:0 for m in ['raw','repair','cold_opf','warm_opf']}
 with dest.open('w') as h:
  for grid in ['ieee14','ieee30']:
   for line in (a.data_dir/f'cases_{grid}.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['split']!=a.split:continue
    rec={'case_id':r['case_id'],'grid':grid,'scenario_id':r['scenario_id'],'methods':{},'reference_validation':validate(r['case'],r['reference'])}
    rec['methods']['raw']={'solution':r['prediction'],'validation':validate(r['case'],r['prediction'])}
    for m in ['repair','cold_opf','warm_opf']:
     try:rec['methods'][m]=solve(r['case'],r['prediction'],m)
     except Exception as e:rec['methods'][m]={'error':repr(e),'validation':{'feasible':False}}
    total+=1
    for m in passed:passed[m]+=int(rec['methods'][m]['validation']['feasible'])
    h.write(json.dumps(rec)+'\n');h.flush()
 summary={'seed':42,'cases':total,'feasible_counts':passed,'settings':SETTINGS,'scope':'Saved predictions; no inference remeasurement or training.','python':sys.version,'platform':platform.platform(),'source_hashes':{label:hashlib.sha256(f.read_bytes()).hexdigest() for label,f in [('code/ac_physics.py',ROOT/'code/ac_physics.py'),('code/ac_solver.py',ROOT/'code/ac_solver.py')]+[(f'inputs/{f.name}',f) for f in inputs]}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
