"""Independent checks of completed arms' provenance, budgets and data flow."""
import argparse,hashlib,json,math,pathlib,statistics,zipfile
D=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main(arms):
    assert arms[0]=='B'and len(set(arms))==len(arms)
    roots={a:D/'completed_results'/a for a in arms}
    initializations={};metrics={};results={};protocols={};flow=[]
    for arm,root in roots.items():
        folder=root/('arm_'+arm)
        result=read(folder/'result.json');results[arm]=result
        assert result['status']=='COMPLETED'and result['epochs']==12 and result['optimizer_updates']==9708 and result['full_model_forwards']==10159
        assert read(folder/'exit_receipt.json')['exit_code']==0
        cpu=read(D/('result_verification_v1/'+arm+'_epoch12_checkpoint_cpu_verification.json'))
        ig=read(D/('result_verification_v1/'+arm+'_epoch12_independent_GT_ignore.json'))
        m=read(D/('result_verification_v1/'+arm+'_epoch12_independent_metrics.json'))
        dl=read(D/('completed_arm_'+arm+'_download_verification.json'))
        assert all(x['status']=='PASS'for x in [cpu,ig,m,dl])
        assert cpu['checkpoint_sha256']==dl['metadata']['epoch12_checkpoint_sha256']
        if arm!='B':assert cpu['reconstructed_initial_reader_matches_saved_digest']
        metrics[arm]=m['metrics'];initializations[arm]=read(folder/'initialization.json')
        protocols[arm]=sha(root/'execution_protocol_FROZEN.json')
        plan=read(root/'execution_protocol_FROZEN.json')
        with zipfile.ZipFile(D/('completed_arm_'+arm+'_bundle.zip'))as z:manifest=json.loads(z.read('member_sha256.json'))
        for name,digest in plan['files_sha256'].items():
            if name=='common_init_10class.pth':assert sha(D/name)==digest
            else:assert manifest[name]==digest,name
        samples=result['latency']['samples_ms']
        assert len(samples)==30 and all(math.isfinite(v)and v>0 for v in samples)
        assert statistics.median(samples)==result['latency']['median_ms']
        assert abs(statistics.fmean(samples)-result['latency']['mean_ms'])<1e-9
        epochs=[json.loads(line)for line in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [x['epoch']for x in epochs]==list(range(1,13))
        assert all(x['optimizer_updates']==809*x['epoch']for x in epochs)
    assert len(set(protocols.values()))==1
    assert len({v['common_init_sha256']for v in initializations.values()})==1
    assert all(v['all_base_tensors_equal']for v in initializations.values())
    assert all(v['protocol_sha256']==protocols[a]for a,v in initializations.items())
    if 'S'in arms and 'L'in arms:
        assert initializations['S']['branch_initial_tensor_digest']==initializations['L']['branch_initial_tensor_digest']
        assert initializations['S']['model_parameters']==initializations['L']['model_parameters']
    for epoch in range(1,13):
        rows={a:read(root/('arm_'+a)/('epoch%02d_input_flow.json'%epoch))for a,root in roots.items()}
        for a,row in rows.items():
            assert row['epoch']==epoch and row['optimizer_updates']==809*epoch
            assert len(row['first_eight_actual_input_sha256'])==8
            assert row['order_sha256']==rows['B']['order_sha256']
            assert row['first_eight_actual_input_sha256']==rows['B']['first_eight_actual_input_sha256']
        flow.append({'epoch':epoch,'same_sampler_order':True,'same_first_eight_actual_batches':True})
    output={'status':'PASS','scope':'Completed arms only; '+('full three-arm comparison available'if len(arms)==3 else 'L endpoint still pending'),'arms':arms,'common_initialization_same':True,'frozen_protocol_same':True,'all_frozen_file_hashes_match_verified_archives':True,'all_twelve_sampler_order_hashes_same':True,'actual_first_eight_batches_match_each_epoch':True,'epoch_details':flow,'metrics':metrics,'deltas_vs_B_pp':{a:{k:metrics[a][k]-metrics['B'][k]for k in ['AP','APs','APs75']}for a in arms if a!='B'},'optimizer_updates_each':9708,'new_forward_calls':0,'new_optimizer_updates':0,'input_scope':'Full planned sampler-order hashes and first eight actual input-batch digests per epoch; not a direct byte comparison of every actual batch.'}
    dest=D/('result_verification_v1/'+'_'.join(arms)+'_controls_verified.json')
    dest.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k]for k in ['status','arms','common_initialization_same','all_twelve_sampler_order_hashes_same','actual_first_eight_batches_match_each_epoch','deltas_vs_B_pp']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',choices=['B','S','L'],nargs='+',required=True);main(p.parse_args().arms)
