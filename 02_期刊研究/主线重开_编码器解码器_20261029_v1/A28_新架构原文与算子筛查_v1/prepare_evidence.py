"""Assemble source-backed scope and an unapproved research candidate."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent
LIB=NEW.parent/'文献_相关原始论文_20261004'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    p=ROOT/name;assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    papers=[
      dict(id='MI_DETR',title='MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism',authors=['Zhixiong Nan','Xianghong Li','Jifeng Dai','Tao Xiang'],url='https://openaccess.thecvf.com/content/CVPR2025/papers/Nan_MI-DETR_An_Object_Detection_Model_with_Multi-time_Inquiries_Mechanism_CVPR_2025_paper.pdf',tier='CVPR2025 official proceedings',stem='A28_MI_DETR_CVPR2025',read_scope='Complete3.1-3.4 and4.4;selected4.1-4.3/tables1,2,5,6. Figure3/Eq1-5 visually checked onPDFpage4. Author decoder/configuration/core branches statically inspected;not all experiments or reproduced models.',author_method='Parallel independently parameterized inquiry branches; concatenate/project query fusion; U-like reverse-depth encoder/decoder connections; Lite-MI shares self-attention.',our_consequence='Parallel inquiries,encoder skip interaction and matched-parameter/compute controls already present. Table6 compares6layersx2/4inquiries with12/24serial layers;cannot call that control an unstudied gap.',unresolved='UAV benefit and a distinct task-specific increment are not established. Public R50 weights/config are not an automatic reproduction of the Relation-DETR extension.'),
      dict(id='AugDETR',title='AugDETR: Improving Multi-scale Learning for Detection Transformer',authors=['Jinpeng Dong','Yutong Lin','Chen Li','Sanping Zhou','Nanning Zheng'],url='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03484.pdf',tier='ECCV2024 official ECVA version',stem='A28_AugDETR_ECCV2024',read_scope='Complete3.1-3.3 and selected main-result table;not all ablations,appendices or code. Eq5 read in primary full text.',author_method='Hybrid dense/sparse encoder and encoder-mixing cross-attention; query-conditioned sigmoid weights aggregate multiple encoder depths.',our_consequence='Query-adaptive encoder-depth access is explicit prior;replacing MI fixed reverse-depth pairing by a learned gate alone is insufficient novelty.',unresolved='No our UAV experiment or a distinct new operator follows.'),
      dict(id='Le_DETR',title='Revisiting Real-Time Detection Transformer with Efficient Encoder Design',authors=['Jiannan Huang','Aditya Kane','Fengzhe Zhou','Yunchao Wei','Humphrey Shi'],url='https://openaccess.thecvf.com/content/CVPR2026F/papers/Huang_Revisiting_Real-Time_Detection_Transformer_with_Efficient_Encoder_Design_CVPRF_2026_paper.pdf',tier='CVPR2026 Findings,not CVPR main track',stem='A28_Le_DETR_CVPRFindings2026',read_scope='Complete3.1-3.5,selected4.1/tables1,3,4. Not full experimental audit or appendix. No verified complete author-code/weight release in this search.',author_method='EfficientNAT combines efficient convolutions and final-stage neighborhood attention; NAIFI replaces encoder attention; ImageNet1K pretraining instead of extra4M-image recipe.',our_consequence='A new-backbone study must compare pretraining cost and source availability;local-attention replacement and lower pretraining cost are already addressed.',unresolved='Paper reports COCO results,not VisDrone gains or single24GB training timing;no reproduction.'),
      dict(id='VBased',title='VBased: Revisiting Architectural Design for Simple Linear Attention Models in Vision',authors=['Anonymous authors in accessed submission'],url='https://openreview.net/pdf?id=bMDc1wMkNe',tier='Accessed TMLR submission says under review;not asserted accepted',read_scope='Indexed primary abstract/introduction and1.2-1.3 fragments only. Official PDF fetch403/challenge;no full-text reading,code or experiments.',author_method='Studies scanning and hybrid attention;fragments describe DeltaNet without sequence permutation plus tiled higher-order sliding-window attention.',our_consequence='A generic DeltaNet-plus-local-attention vision encoder or a new scan order is not an untouched concept.',unresolved='Exact full method,authors,status,weights and detailed results remain unverified.'),
      dict(id='mHC_DEIM',title='mHC-DEIM: Object Detection for Shelf Scenes under Dense Arrangement, Multi-Angle Inclination and Low-Illumination',authors=['Mingxiao Sun','Jinhui Li','Tiantian Luan'],url='https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6557880',tier='SSRN preprint listing posted11Apr2026;not peer-review claim',read_scope='Author abstract in indexed primary listing only. Full40-page document not retrieved/read;direct page403.',author_method='Abstract describes inserting manifold-constrained hyper-connections into DEIM for dense shelf detection.',our_consequence='Bare mHC-to-DETR transfer already has a direct detection preprint;abstract is enough to exclude a first-transfer claim,not to compare complete operators.',unresolved='Full method,code,training and exact overlap are unknown.'),
      dict(id='DF_Mamba',title='Dynamic Fractal Mamba: A Neural Renormalization Group Flow for Scale-Invariant Sequence Modeling',authors=['Shenglei Fang','Xianfang Sun','You Zhou'],url='https://proceedings.mlr.press/v306/fang26p.html',tier='ICML2026,PMLR306:29318-29346',stem='A28_DF_Mamba_ICML2026',read_scope='Complete3.1-3.4;selectedAppendixA.14-A.15. Figure1/Eq3,5 visually checkedPDFpage3. Not all theory,experiments or author code.',author_method='Shared recursive selective SSM,content-adaptive step multiplied by2^depth,learned pairwise coarsening,local/global gated fusion and reconstruction/consistency losses.',our_consequence='Scale-consistent selective dynamics,time-step scaling and preservation of fine residuals already have prior motivation;UAV transfer alone does not clear novelty.',unresolved='No bounding-box UAV task was verified here;exact comparison with checkpoint-preserving inference adaptation remains open.'),
      dict(id='Tabular_Length_Drift',title='Adapting Linear-Time Architectures for Tabular In-Context Learning',authors=['David Schnurr','Felix Sarnthein','Thomas Hofmann','Imanol Schlag'],url='https://arxiv.org/abs/2609.36337v1',tier='arXiv2609.36337v1,28Sep2026;not asserted peer reviewed',stem='A28_Tabular_Linear_Architectures_arXiv2609_36337',read_scope='Complete4.4/5;selected4.1/4.3;Figure3 and recurrence visually checkedPDFpage8. Selected native beta-decay helper and configuration inspected;not whole appendix/training/reproduction.',author_method='Controlled tabular-ICL study;state oracle and depth/length drift analysis;beta_eff=beta*c/(t+c),c256;final-state noncausal readout.',our_consequence='Length-conditioned write calibration and final-state readout are prior;task-specific evidence cannot be treated as a demonstrated UAV or VSSD defect.',unresolved='Applicability to a given visual operator and causal small-object AP effects are unproven. Current code pin is later than v1 publication.'),
      dict(id='S4ND',title='S4ND: Modeling Images and Videos as Multidimensional Signals Using State Spaces',authors=['Eric Nguyen','Karan Goel','Albert Gu','Gordon W. Downs','Preey Shah','Tri Dao','Stephen A. Baccus','Christopher Ré'],url='https://arxiv.org/abs/2210.06583v1',tier='NeurIPS2022;read arXivv1,not asserted identical to final version',stem='A28_S4ND_arXiv2210_06583v1',read_scope='Sec3,selected4.1,4.2 and selected5.3. PDFpage5 visually checked for multidimensional construction and resolution-dependent step. Not whole paper or code.',author_method='Multidimensional continuous SSM kernels;change step with resolution and bandlimit kernels to reduce aliasing.',our_consequence='Two-dimensional resolution-adaptive SSMs and simple step rescaling are longstanding prior;fractional affine-flow algebra is not a new primitive.',unresolved='A reference-preserving policy for an existing selective-scan checkpoint still requires exact prior comparison and UAV task evidence.'),
      dict(id='VMamba',title='VMamba: Visual State Space Model',authors=['Yue Liu','Yunjie Tian','Yuzhong Zhao','Hongtian Yu','Lingxi Xie','Yaowei Wang','Qixiang Ye','Jianbin Jiao','Yunfan Liu'],url='https://arxiv.org/abs/2401.10166v4',tier='NeurIPS2024 spotlight;read arXivv4,29Dec2024',stem='A28_VMamba_arXiv2401_10166v4',read_scope='Abstract,Sec3 andAppendixA,selectedFig2 scan diagram. Official reference scan,SS2Dv2 forwarding and one active Tiny config statically inspected. Not full empirical paper or trained-model reproduction.',author_method='Four-route 2D selective scan. Native Torch reference uses exp(delta*A) state retention and delta*B*u input;AppendixA explicitly identifies first-order B approximation.',our_consequence='The input update is a documented design,not a newly discovered bug. One current Tiny config hasstate_dim1,permitting a potentially cheap per-channel input correction;not all VMamba variants.',unresolved='Full native-wrapper fidelity,CUDA equivalence,pretrained checkpoint loading and a VisDrone detector host remain unverified.'),
      dict(id='Mamba3',title='Mamba-3: Improved Sequence Modeling using State Space Principles',authors=['Aakash Lahoti','Kevin Y. Li','Berlin Chen','Caitlin Wang','Aviv Bick','J. Zico Kolter','Tri Dao','Albert Gu'],url='https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html',tier='ICLR2026 official proceedings',stem='A28_Mamba3_ICLR2026',read_scope='Sec2.2,complete3.1 through remarks andAppendixB.1. Other innovations,full experiments and kernels not fully read/reproduced.',author_method='Formalizes prior Mamba exponential-Euler update;generalized exponential-trapezoidal recurrence also reads previous input,plus complex states/MIMO in the larger model.',our_consequence='A better discretization or state-input correction is not an untouched research idea;candidateP must be positioned as a specific inference calibration policy,not a new numerical primitive.',unresolved='Whether reference-preserving resolution adaptation is already implied/implemented elsewhere remains a novelty risk;no our UAV benefit follows.')
    ]
    for row in papers:
        row.update(read_by='assistant',human_read_attested=False,source_recorded_utc=now)
        stem=row.pop('stem',None)
        if stem:
            record=read(LIB/(stem+'.source.json'))
            assert sha(record['file'])==record['sha256'] and sha(record['text_file'])==record['text_sha256']
            record.update(read_scope=row['read_scope'],read_by='assistant',human_read_attested=False)
            (LIB/(stem+'.source.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            row['local_pdf']=record['file'];row['pdf_sha256']=record['sha256'];row['access_record']=str(LIB/(stem+'.source.json'))
        else:
            record={**row,'pdf_obtained':False,'access_limitation':'403/challenge;indexed primary fragments/abstract only.'}
            path=LIB/('A28_'+row['id']+'_public_access.source.json')
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');row['access_record']=str(path)
    save('new_core_records.json',{'review_type':'targeted non-systematic architecture followup','rows':papers})
    manifests=read(ROOT/'selected_source_manifest.json')['rows']+[read(ROOT/'vmamba_source_manifest.json')]
    count=0
    for group in manifests:
        base=Path(group.get('source_root',ROOT/'vmamba_source'))
        for row in group['files']:
            p=base/row['path'];data=p.read_bytes();assert sha(p)==row['sha256']
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1'];count+=1
    extra=read(ROOT/'additional_source_manifest.json');assert sha(extra['path'])==extra['sha256'];count+=1
    vm=read(ROOT/'vmamba_source_manifest.json');config=ROOT/'vmamba_source/classification/configs/vssm/vmambav2v_tiny_224.yaml'
    assert 'SSM_D_STATE: 1' in config.read_text(encoding='utf-8')
    for path in [ROOT/'vmamba_source/classification/models/csms6s.py',ROOT/'source/CQU-ADHRI-Lab__MI-DETR/projects/midetr/modeling/midetr_transformer.py']:
        ast.parse(path.read_text(encoding='utf-8'))
    audit={'status':'STATIC_PRIMARY_SOURCE_AUDIT_COMPLETE','verified_git_blobs':count,
      'MI_DETR_commit':read(ROOT/'pinned_author_repositories.json')['rows'][0]['commit'],
      'tabular_commit':read(ROOT/'pinned_author_repositories.json')['rows'][1]['commit'],
      'VMamba_commit':vm['commit'],'VMamba_selected_config':str(config),'selected_state_dim':1,
      'native_reference':{'file':str(ROOT/'vmamba_source/classification/models/csms6s.py'),'sha256':sha(ROOT/'vmamba_source/classification/models/csms6s.py'),
        'lines':'25-68;retention55,input56,skip67','rule':'h=exp(delta*A)*h+delta*B*u; output=C*h+D*u;softplus applies after delta_bias.'},
      'VSSD_exclusion':{'file':str(NEW/'source_audit_v1/A18_VSSD_mamba2.py'),'sha256':sha(NEW/'source_audit_v1/A18_VSSD_mamba2.py'),
        'lines':'404-479','consequence':'Noncausal pooled KV operator has no native sequential beta update;tabular DeltaNet write-rate mechanism does not directly identify an N/VSSD failure.'},
      'MI_scope':'Branches deepcopy decoder layers. UFI is final+intermediate encoder concatenation,projection and reversed depth. state_aware_query_fusion is Linear+LayerNorm,not a demonstrated adaptive routing mechanism. Active R50 config900queries,6encoder/6decoder layers,4inquiries,batch16;source defaults are not single24GB evidence.',
      'author_code_imported_or_executed':False,'new_runtime_or_packages':False}
    assert count==23;save('operator_source_audit.json',audit)
    algebra=read(ROOT/'algebra_result.json');assert algebra['status']=='PASS_CONDITIONAL_AFFINE_FLOW_IDENTITY_ONLY'
    with np.load(ROOT/'algebra_fixture.npz',allow_pickle=False) as z:
        a=z['log_a'];b=z['b'];h0=z['h0'];native=z['native']
        assert np.array_equal(np.exp(a)*h0+b,native)
        for m in [2,4,8,16]:
            r=1/m;ratio=np.divide(np.expm1(a*r),np.expm1(a),out=np.full_like(a,r),where=a!=0)
            # Closed-form geometric-series accumulation, not the prototype loop.
            total=sum(np.exp(a*r*j) for j in range(m))
            own=np.exp(a)*h0+ratio*b*total
            assert np.max(np.abs(own-native))<1e-10
    assert sha(ROOT/'algebra_fixture.npz')==algebra['fixture_sha256']
    save('source_and_algebra_verification.json',{'status':'PASS_BYTE_SCOPE_AND_SYNTHETIC_GEOMETRIC_SERIES_CHECK',
      'new_core_rows':len(papers),'PDFs_archived':sum('local_pdf' in p for p in papers),
      'fragment_or_abstract_only_records':sum('local_pdf' not in p for p in papers),
      'verified_author_git_blobs':count,'algebra_cases':1000,'independent_closed_form_checks':4,
      'trained_model_calls':0,'optimizer_updates':0,'mainline_validated':False})
    method={'candidate_id':'P_REFERENCE_PRESERVING_RESOLUTION_STATE_UPDATE',
      'status':'SOURCE_MOTIVATED_DRAFT_ALGEBRA_ONLY_NATIVE_TASK_AND_NOVELTY_PENDING',
      'question':'变分辨率航拍检测中，选择性扫描的状态保留与输入写入如何共同适配token网格变化？参考保持的联合校准能否改善小目标检测，而不损失参考分辨率下的既有行为？',
      'native_scalar_update':'h_next=exp(a)*h+b,where a=delta*A<=0,b=delta*B*u.',
      'proposed_coefficients':'a_rho=rho*a;b_rho=expm1(rho*a)/expm1(a)*b;ratio tends to rho when a=0;rho=1 bypasses calibration.',
      'proposed_grid_policy_DRAFT':'rho=N_reference/N_current per scan stage,using explicitly fixed feature-grid token counts. This is an unvalidated area/scan-progress policy,not a proven2D physical law;axis/padding/route behavior must be checked before freezing a real test.',
      'cheap_implementation_scope':'Only state_dim1 permits gain g=expm1(rho*a)/(rho*expm1(a)) folded into per-channel u. Pass rho*delta after softplus;retain original D*u outside the scan. This is an interface proposal,not a verified native or CUDA implementation.',
      'formal_contract_scope':'rho1 preserves toy reference coefficients;constant coefficients in a block admit exact fractional refinement. This is classical affine-flow algebra,not a new theorem or arbitrary2D-image invariance.',
      'nearest':['S4ND','DF_Mamba','Mamba3','VMamba','Tabular_Length_Drift'],
      'potential_increment_NOT_CLEARED':'Checkpoint-preserving inference calibration policy for an existing approximate selective scan,with joint retention/input scaling and a state_dim1 fast path. Generic rediscretization,high-order integration and bare task transfer are not claimed new.',
      'novelty_risk':'Could reduce to existing SSM rediscretization or affine reparameterization. Need exact prior/implementation comparison beyond these selected sources;absence in this bounded search is not priority proof.',
      'task_risk':'No measured resolution-induced UAV defect or AP gain;higher resolution changes information,positions,convolutions,scan ordering and detector head behavior. Algebra does not isolate these.',
      'dataset':'VisDrone primary candidate;an independent UAV dataset is deferred until its sources/assets are checked.',
      'algorithm_host':'The selected VMamba-Tiny state_dim1 config is an operator candidate,not yet a frozen or pretrained VisDrone detector. A reproducible detector host and checkpoint must be chosen/verified. Standard YOLO26s can remain a contextual baseline;backbone transfer itself is not innovation.',
      'new_module_training_approved':False,'increment_approved':False,'causal_evidence':False,
      'mainline_validated':False,'native_model_or_component_calls_allocated':0}
    save('P_method_draft.json',method)
    protocol={'status':'DRAFT_NOT_EXECUTION_ALLOCATED','candidate':method['candidate_id'],
      'stage0_completed':'Primary source/formula scope and1000 synthetic affine coefficient cases only.',
      'next_stage':'Freeze a24-call CPU native-scan component protocol after exact operator/prior checks;4synthetic fixturesx6arms. No trained model or real image is needed for this engineering stage.',
      'proposed_CPU_scan_cap_NOT_ALLOCATED':24,'actual_CPU_scan_allocation':0,
      'six_component_arms':['native reference','calibrated rho1 bypass','naive refinementx2','joint refinementx2','naive refinementx4','joint refinementx4'],
      'component_requirements':['Native function source SHA andAST selection frozen;no external code import side effects.',
        'Softplus/bias conventions,D*u skip,float32 accumulation and return dtype preserved.',
        'rho1 full returned tensor bitwise matches native reference.',
        'Constant repeated-coefficient block endpoints agree within frozen float32 tolerance;no arbitrary-image equivariance claim.',
        'Reject unsupported state_dim>1 fast path and rho<=0;stop on source/finiteness/fidelity failures.'],
      'after_component_pass':'Verify a compatible official Tiny checkpoint and a detector host;measure peak memory/timing on single24GB before allocating baseline training. No automatic continuation.',
      'task_design_DRAFT':['Use VisDrone train/validation split;select fresh prediction-blind development cases after auditing past image-use inventory. Final claim needs full validation and independent data/seed evidence.',
        'Fixed common baseline checkpoint and same query/output cap at each resolution;compare native,delta-only,strict-ZOH,reference-preserving correction and a simple amplitude control at equal input resolution.',
        'Reference resolution behavior is an identity gate. Different resolutions are compared within paired arms;input enlargement alone is not evidence of the mechanism.',
        'Measure official-policy-aligned AP/APsmall,coverage/false positives,latency and peak memory. Freeze scene/prefix sampling,thresholds,CI and stop conditions before seeing outcomes.',
        'Novelty gate can reject the candidate before experiments if a closest method implements the same policy. No old A22/DEIM restart.'],
      'new_whole_detector_call_cap':0,'new_training_update_cap':0,'new_GPU_call_cap':0,
      'not_a_scientific_protocol_FREEZE':'This file is a reviewable draft;no task effects,training,resource claim or next experiment approved by its existence.'}
    save('minimal_validation_DRAFT.json',protocol)
    print(json.dumps({'new_core_rows':10,'archived_PDFs':8,'author_git_blobs':23,'new_candidate':method['candidate_id'],
      'native_component_calls_allocated':0,'new_model_calls':0,'mainline_validated':False}))

if __name__=='__main__':main()
