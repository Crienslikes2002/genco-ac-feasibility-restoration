"""Portable numerical replay using only bundled sources, models and selected raw graphs."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata as metadata,json,os,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/'vendor/gridfm_graphkit'
sys.dont_write_bytecode=True
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
sys.path.insert(0,str(UPSTREAM))
import numpy as np
import torch,yaml
from torch_geometric.data import HeteroData
import gridfm_graphkit
from gridfm_graphkit.io.param_handler import NestedNamespace,load_model
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches,RemoveInactiveGenerators,ApplyMasking
from gridfm_graphkit.datasets.masking import AddOPFHeteroMask

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative_output(name):
 p=(ROOT/name).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Output must be inside bundle root')
 return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/reinference_predictions.jsonl');ap.add_argument('--manifest',default='results/reinference_manifest.json');a=ap.parse_args()
 out=relative_output(a.output);mp=relative_output(a.manifest)
 if out.exists() or mp.exists():raise FileExistsError('Refusing overwrite; supply new relative --output and --manifest paths')
 assert Path(gridfm_graphkit.__file__).resolve().is_relative_to(UPSTREAM.resolve()),'Not importing bundled upstream source'
 source_manifest=json.loads((UPSTREAM/'source_manifest.json').read_text())
 for entry in source_manifest['files']:assert sha(UPSTREAM/entry['path'])==entry['sha256'],entry['path']
 torch.set_num_threads(2);torch.manual_seed(42)
 frozen=json.loads((ROOT/'data/frozen_selection.json').read_text());assert frozen['seed']==42
 records=[];maxdiff=0.;model_records={}
 for grid in ['ieee14','ieee30']:
  rows=[json.loads(s) for s in (ROOT/'data'/f'cases_{grid}.jsonl').read_text().splitlines()]
  assert len(rows)==60
  md=ROOT/'models'/grid;args=NestedNamespace(**yaml.safe_load((md/'config.yaml').read_text()));network=args.data.networks[0]
  assert args.seed==42 and network=={'ieee14':'case14_ieee','ieee30':'case30_ieee'}[grid]
  norm=HeteroDataMVANormalizer(args);stats=torch.load(md/'normalizer_stats.pt',weights_only=True,map_location='cpu');norm.fit_from_dict(stats[network])
  model=load_model(args).cpu().eval();state=torch.load(md/'best_model_state_dict.pt',weights_only=True,map_location='cpu');model.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True)
  checkpointsplits=json.loads((md/(network+'_scenario_splits.json')).read_text());gridmax=0.
  for row in rows:
   sid=row['scenario_id'];split=row['split'];assert sid in frozen['grids'][grid][split] and sid in checkpointsplits[split]
   p=ROOT/row['raw_graph'];assert sha(p)==row['raw_graph_sha256'];g=HeteroData.from_dict(torch.load(p,weights_only=True));assert int(g.scenario_id.item())==sid
   norm.transform(g);g=RemoveInactiveBranches()(g);g=RemoveInactiveGenerators()(g);g=AddOPFHeteroMask()(g);g=ApplyMasking(args)(g)
   with torch.no_grad():prediction=model(g.x_dict,g.edge_index_dict,g.edge_attr_dict,g.mask_dict)
   norm.inverse_output(prediction,g)
   pred={'vm':prediction['bus'][:,0].tolist(),'va':prediction['bus'][:,1].tolist(),'pg':prediction['gen'][:,0].tolist(),'qg_bus':prediction['bus'][:,3].tolist()}
   diffs={k:float(np.max(np.abs(np.asarray(v)-row['prediction'][k]))) for k,v in pred.items()}
   delta=max(diffs.values());gridmax=max(gridmax,delta);maxdiff=max(maxdiff,delta)
   records.append({'case_id':row['case_id'],'grid':grid,'split':split,'scenario_id':sid,'prediction':pred,'absolute_difference_by_array':diffs,'max_absolute_difference':delta,'finite':all(bool(np.isfinite(v).all()) for v in map(np.asarray,pred.values()))})
  model_records[grid]={'checkpoint_sha256':sha(md/'best_model_state_dict.pt'),'config_sha256':sha(md/'config.yaml'),'normalizer_sha256':sha(md/'normalizer_stats.pt'),'n_val':10,'n_test':50,'max_absolute_difference':gridmax}
  print('REINFERENCE',grid,60,'max_abs_diff',gridmax,flush=True)
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w') as f:
  for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
 versions={}
 for package in ['torch','torch-geometric','torch-scatter','numpy','pandas','pyyaml','lightning','pytorch-lightning','torchmetrics','mlflow','matplotlib','seaborn','networkx','scipy']:
  try:versions[package]=metadata.version(package)
  except metadata.PackageNotFoundError:versions[package]=None
 report={'status':'PASS_EXACT' if maxdiff==0. else 'NUMERICAL_DIFFERENCE_REPORTED','purpose':'Numerical portable replay; no training and no timing conclusion','seed':42,'n_cases':len(records),'all_predictions_finite':all(r['finite'] for r in records),'max_absolute_difference_from_original':maxdiff,'device':'cpu','torch_threads':torch.get_num_threads(),'python':platform.python_version(),'platform':platform.platform(),'package_versions':versions,'source_directory':'vendor/gridfm_graphkit','source_manifest_sha256':sha(UPSTREAM/'source_manifest.json'),'source_files_verified':len(source_manifest['files']),'models':model_records,'output_sha256':sha(out),'script_sha256':sha(Path(__file__)),'portability':'Every source, model, normalizer, raw graph and case path used for inference resolves relative to this bundle; installed Python dependencies remain required. Original absolute paths inside historical configs are provenance only and are not used to locate data.'}
 mp.parent.mkdir(parents=True,exist_ok=True);mp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print('REINFERENCE_COMPLETE',len(records),maxdiff,flush=True)
if __name__=='__main__':main()
