"""Frozen inference only. Extract native change coordinates, never invert endpoints."""
from pathlib import Path
import argparse,json,sys,pickle,hashlib,traceback,os
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,dtype={'patient_id':str},float_precision='round_trip')

def main(a):
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 import torch
 sys.path.insert(0,a.source)
 from lac_itransformer.data.preprocessing import FoldPreprocessor,subset_arrays
 from lac_itransformer.experiments.prediction import build_torch_model,summarize_features
 from lac_itransformer.models.lac import LACConfig
 from lac_itransformer.models.lac_v60 import LACV60Config
 from lac_itransformer.training import v27_nested as v27,v60_nested as v60
 from lac_itransformer.training.v60_selector import V60GenericDecisionLayer
 from lac_itransformer.training.trainer import _loader
 from lac_itransformer.training.dataset import model_inputs
 V60GenericDecisionLayer.configured_generic_feature_count=92
 V60GenericDecisionLayer.configured_min_samples_leaf=20
 v27._feature_flags=v60._feature_flags
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 jobs=json.loads(Path(a.jobs).read_text())
 keys=[job['key'] for job in jobs]
 if len(set(keys))!=len(keys) or any(Path(k).name!=k or '/' in k or '\\' in k for k in keys):
  raise ValueError('Job keys must be unique plain filenames')
 failures=[]
 for job in jobs:
  key=job['key'];dest=out/(key+'.csv');auditpath=out/(key+'.json')
  native_fold=job['model'].startswith(('vascmtl','vasctimeformer','lac_v60')) and 'fold' in job
  if dest.exists() or auditpath.exists():
   raise FileExistsError(f'Refusing to overwrite extraction: {key}')
  try:
   folder=Path(job['folder']);modelname=job['model'];sourcepaths=[Path(job['arrays']),Path(job['reference'])]
   if modelname!='persistence':sourcepaths += [folder/'preprocessor.json',folder/('model.pkl' if modelname in ['elastic_net','xgboost'] else 'model.pt')]
   if (folder/'decision_layer.pkl').exists():sourcepaths.append(folder/'decision_layer.pkl')
   hashes={str(p):sha(p) for p in sourcepaths}
   arrays=dict(np.load(job['arrays'],allow_pickle=False))
   reference=read(job['reference'])
   if 'fold' in job:reference=reference.loc[reference.outer_fold.eq(job['fold'])].copy()
   ids=reference.patient_id.astype(str).tolist()
   if len(ids)!=len(set(ids)):raise ValueError('Duplicate reference patients')
   ix=pd.Index(arrays['patient_ids'].astype(str)).get_indexer(ids)
   if (ix<0).any():raise ValueError('Missing input patients')
   raw=subset_arrays(arrays,ix)
   # Only dummy targets are made available to the predictor.
   targets=raw.get('targets')
   evaluation_ids=arrays['patient_ids'].astype(str)
   if 'order' in job:
    orderpath=Path(job['order']);hashes[str(orderpath)]=sha(orderpath)
    if orderpath.suffix=='.npz':evaluation_ids=np.load(orderpath,allow_pickle=False)['patient_ids'].astype(str)
    else:evaluation_ids=read(orderpath).patient_id.astype(str).to_numpy()
   patient_order=pd.Index(evaluation_ids).get_indexer(ids)
   if (patient_order<0).any():raise ValueError('Evaluation order is incomplete')
   inputs={**raw,'targets':np.zeros((len(ids),2),dtype=np.float32)}
   baseline=raw['baseline'].astype(float)
   if modelname=='persistence':
    residual=np.zeros_like(baseline);endpoint=baseline.copy();parameterization='zero_change_reference'
   else:
    transformed=FoldPreprocessor.load(folder/'preprocessor.json').transform(inputs)
    if modelname in ['elastic_net','xgboost']:
     with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
     features=summarize_features(transformed)
     residual=np.column_stack([m.predict(features) for m in model.models])
     endpoint=model.predict(transformed)
     parameterization='native_residual_regressors'
    else:
     native=modelname.startswith(('vascmtl','vasctimeformer','lac_v60'))
     if native and 'fold' in job:v60._seed_everything_v60(2026)
     else:torch.use_deterministic_algorithms(False)
     torch.backends.cuda.enable_flash_sdp(not(native and 'fold' in job))
     torch.backends.cuda.enable_mem_efficient_sdp(not(native and 'fold' in job))
     torch.backends.cuda.enable_math_sdp(True)
     checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
     conf=LACV60Config(**checkpoint['model_config']) if native else LACConfig(**checkpoint['model_config'])
     model=build_torch_model(modelname,conf).to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
     if native:
      bundle=v27._predict_base_with_features(model,transformed,inputs,64,device,modelname)
      b_ids=list(map(str,bundle['patient_ids']));bi=pd.Index(transformed['patient_ids'].astype(str)).get_indexer(b_ids)
      bundle['features']['cac']=np.column_stack([bundle['features']['cac'],v60._generic_temporal_context(transformed)[bi]])
      with (folder/'decision_layer.pkl').open('rb') as f:decision=pickle.load(f)
      native_output,_=decision.predict(bundle['heads'],bundle['features'],v60._decision_modes(modelname))
      oi=pd.Index(b_ids).get_indexer(ids);native_output=native_output[oi]
      if bool(getattr(conf,'direct_endpoint_prediction_enabled',False)):
       # This ablation is trained in endpoint coordinates. Subtract baseline
       # in those coordinates, before expm1 or any nonnegative truncation.
       residual=native_output-np.column_stack([baseline[:,0],np.log1p(baseline[:,1])])
       endpoint=np.column_stack([native_output[:,0],np.maximum(0,np.expm1(native_output[:,1]))])
       parameterization='native_unclipped_endpoint_coordinates_minus_baseline'
      else:
       residual=native_output;endpoint=v27._residual_to_endpoint(baseline,residual)
       parameterization='native_residual_after_saved_decision_layer'
     else:
      rs=[];es=[];b_ids=[]
      with torch.inference_mode():
       for batch in _loader(transformed,64,False):
        batch=batch.to(device);pred=model(**model_inputs(batch))
        rs.append(torch.stack([pred['delta_tbr'],pred['delta_log_cac']],-1).cpu().numpy())
        es.append(torch.stack([pred['endpoint_tbr'],pred['endpoint_cac']],-1).cpu().numpy());b_ids+=batch.patient_ids
      oi=pd.Index(b_ids).get_indexer(ids);residual=np.concatenate(rs)[oi];endpoint=np.concatenate(es)[oi]
      parameterization='native_neural_residual_heads'
     del model
     if device.type=='cuda':torch.cuda.empty_cache()
   reference=reference.set_index('patient_id').loc[ids]
   pc=job.get('reference_columns',['pred_tbr','pred_cac'])
   archived=reference[pc].to_numpy(float)
   if native_fold:
    # Prefer actually archived native float32 heads for identity/central paths.
    # This restores original bits from float32 CSV serialization and avoids
    # tiny re-inference differences masquerading as paired improvements.
    da=decision.audit()['unique_fixed_configuration']
    assert da['tbr']['method']=='identity'
    direct_endpoint=bool(getattr(conf,'direct_endpoint_prediction_enabled',False))
    if direct_endpoint:
     native_tbr=reference.direct_head_tbr_endpoint.to_numpy(np.float32).astype(float)
     residual[:,0]=native_tbr-baseline[:,0]
    else:residual[:,0]=reference.median_pred_delta_tbr.to_numpy(np.float32).astype(float)
    if v60._decision_modes(modelname)['cac']=='central' or da['cac'].get('calibration_weight')==0:
     if direct_endpoint:residual[:,1]=reference.direct_head_log_endpoint_cac.to_numpy(np.float32).astype(float)-np.log1p(baseline[:,1])
     else:residual[:,1]=reference.median_pred_delta_log_cac.to_numpy(np.float32).astype(float)
    endpoint=v27._residual_to_endpoint(baseline,residual)
   diff=np.abs(endpoint-archived)
   if not np.allclose(endpoint,archived,rtol=2e-6,atol=2e-5):
    raise ValueError(f'Frozen inference differs from archived predictions: max={diff.max()}')
   if not np.isfinite(residual).all():raise ValueError('Nonfinite direct predictions')
   # External labels are opened only after frozen prediction and archival checks.
   if targets is None:
    sealed=dict(np.load(job['outcomes'],allow_pickle=False));index=pd.Index(sealed['patient_ids'].astype(str)).get_indexer(ids)
    if (index<0).any():raise ValueError('Missing sealed outcomes')
    targets=sealed['targets'][index];hashes[job['outcomes']]=sha(job['outcomes'])
   target=np.asarray(targets,float)
   truth=np.column_stack([target[:,0]-baseline[:,0],np.log1p(target[:,1])-np.log1p(baseline[:,1])])
   policy=job.get('truth_policy','raw_arrays_float64')
   if policy=='archived_change_columns':
    truth=reference[['true_delta_tbr','true_delta_log_cac']].to_numpy(float)
   elif policy=='b31_legacy_baseline_log_float32':
    truth[:,1]=np.log1p(target[:,1])-np.log1p(raw['baseline'][:,1].astype(np.float32))
   elif policy!='raw_arrays_float64':raise ValueError('Unknown truth policy')
   old=np.column_stack([archived[:,0]-baseline[:,0],np.log1p(archived[:,1])-np.log1p(baseline[:,1])])
   treatment_count=(np.max(raw['treatments'],axis=1)>0).sum(axis=1)
   frame=pd.DataFrame({'patient_id':ids,'development':job['development'],'centre':job['centre'],'model':job.get('public_model',modelname),
    'experiment':job['experiment'],'outer_fold':job.get('fold',0),'patient_order':patient_order,'baseline_tbr':baseline[:,0],'baseline_tac':baseline[:,1],
    'true_tbr':target[:,0],'true_tac':target[:,1], 'followup_months':raw['followup_months'].astype(float),
    'true_delta_tbr':truth[:,0],'true_delta_log_tac':truth[:,1],
    'pred_delta_tbr':residual[:,0],'pred_delta_log_tac':residual[:,1],
    'legacy_pred_delta_tbr':old[:,0],'legacy_pred_delta_log_tac':old[:,1],
    'age':raw['static'][:,0],'sex_code':raw['static'][:,1],
    'treatment_group':np.where(treatment_count==0,'No treatment',np.where(treatment_count==1,'Single treatment','Combined treatment')),
    'prediction_parameterization':parameterization})
   assert len(frame)==frame.patient_id.nunique()
   frame.to_csv(dest,index=False,encoding='utf-8-sig')
   for p,h in hashes.items():assert sha(p)==h
   audit={'status':'PASS','key':key,'n':len(frame),'source':'frozen_weights_inference_native_outputs','no_training':True,
          'original_deterministic_settings':bool(native_fold),
          'archived_native_identity_heads_preferred':bool(native_fold),
          'parameterization':parameterization,'truth_policy':policy,'device':str(device),'max_archived_endpoint_difference_tbr':float(diff[:,0].max()),
          'max_archived_endpoint_difference_tac':float(diff[:,1].max()),'hashes':hashes,'output_sha256':sha(dest)}
   auditpath.write_text(json.dumps(audit,indent=2));print('EXTRACTED',key,len(frame),diff.max(),flush=True)
  except Exception as e:
   failures.append(key)
   auditpath.write_text(json.dumps({'status':'Needs verification','key':key,'reason':str(e),'traceback':traceback.format_exc()},indent=2))
   print('NEEDS_VERIFICATION',key,str(e),flush=True)
 if failures:raise RuntimeError(f'Extraction failed for {len(failures)} job(s); inspect the local audit files')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]/'src'));p.add_argument('--jobs',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');main(p.parse_args())
