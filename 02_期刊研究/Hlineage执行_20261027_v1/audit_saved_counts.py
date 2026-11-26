"""Audit saved integer contingency tables; no new hypothesis or execution."""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;O=HERE/'audit_v1';SRC=HERE/'attempt01'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def xl(v):return float(np.sum(v[v>0]*np.log2(v[v>0])))
def main():
 assert not O.exists();O.mkdir();z=read(SRC/'results.json');assert read(SRC/'verification.json')['status']=='PASS';rows=[];sources={}
 for p in (Path(__file__),SRC/'results.json',SRC/'verification.json'):sources[str(p)]=sha(p)
 for key,v in z['conditions'].items():
  p=SRC/f'{key}_primary_arrays.npz';sources[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as a:
   g=np.unique(a['strata_code'],return_inverse=True)[1];y=a['label'];j=a['Jreal'];jg=a['Jgeo'];im=a['image_id'];n=len(y);nz=int(g.max())+1
   cy=np.bincount(g*2+y,minlength=nz*2).reshape(nz,2);cr=np.bincount(g*6+j*2+y,minlength=nz*6).reshape(nz,3,2);cg=np.bincount(g*6+jg*2+y,minlength=nz*6).reshape(nz,3,2)
   hy=(xl(cy.sum(1))-xl(cy))/n
   def mi(ct):return (xl(ct)+xl(ct.sum((1,2)))-xl(ct.sum(1))-xl(ct.sum(2)))/n
   rmi,gmi=mi(cr),mi(cg);s=v['primary_qualified'];assert abs(rmi-s['real_CMI_bits'])<1e-12 and abs(gmi-s['geometry_CMI_bits'])<1e-12 and abs(hy-s['HY_given_controls_upper_bound_bits'])<1e-12
   geo_mixed=(cg[:,:,0]>0)&(cg[:,:,1]>0);real_mixed=(cr[:,:,0]>0)&(cr[:,:,1]>0);cgflat=cg.reshape(nz*3,2);ge_mixedpairs=int(cgflat[geo_mixed.flatten()].sum());geomPure=not bool(geo_mixed.any())
   r={'condition':key,'qualified_pair_denominator':n,'qualified_images':len(np.unique(im)),'label1':int(y.sum()),'label0':int(n-y.sum()),'control_strata':nz,'control_singleton_strata':int((cy.sum(1)==1).sum()),'control_outcome_mixed_strata':int((cy>0).all(1).sum()),'control_outcome_entropy_bits':hy,'real_CMI_bits':rmi,'geometry_CMI_bits':gmi,'geometry_mixed_C_J_cells':int(geo_mixed.sum()),'geometry_mixed_C_J_pairs':ge_mixedpairs,'geometry_reaches_exact_entropy_ceiling_integer_certificate':geomPure,'real_geo_joint_tables_identical':bool(np.array_equal(cr,cg)),'real_geo_J_difference_pairs':int((j!=jg).sum()),'real_geo_J_same_fraction':float(np.mean(j==jg)),'real_C_J_mixed_cells':int(real_mixed.sum()),'strong_null_min_CMI_bits':min(s['strong_null_CMI_bits_199']),'strong_null_median_CMI_bits':float(np.median(s['strong_null_CMI_bits_199'])),'strong_null_max_CMI_bits':max(s['strong_null_CMI_bits_199']),'real_exceeds_strong_null_min':rmi>min(s['strong_null_CMI_bits_199'])+1e-12,'informative_pairs':s['informative_pairs'],'informative_images':s['informative_images'],'advance_gate_pass':s['advance_gate_pass']}
   assert rmi<=hy+1e-12 and gmi<=hy+1e-12
   if geomPure:assert abs(gmi-hy)<1e-12
   rows.append(r)
 assert all(sha(p)==h for p,h in sources.items())
 with open(O/'integer_and_entropy_audit.csv','x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (O/'results.json').write_text(json.dumps({'status':'PASS','scope':'Saved-table integrity, entropy/degeneracy explanation only; no new bins/statistic selection or hypotheses','conditions':{r['condition']:r for r in rows},'input_sha256':sources,'all_CMI_exactly_recomputed_within_1e_12':True,'operations':{'model_forwards':0,'evaluator_calls':0,'training':0}},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
