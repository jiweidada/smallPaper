from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;PREV=P.parent/'A52_不同机制候选有界筛选_20261101_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not(P/'train_A54.py').exists()
    cap=json.loads((P/'capacity_v2/capacity_result.json').read_text());assert cap['status']=='PASS' and cap['cumulative_A54_capacity_forwards']==9 and cap['capacity_only_optimizer_updates']==6
    s=(PREV/'train_A52.py').read_text().replace('runtime_A52','runtime_A54').replace('9809592','9809322').replace('6480','6210')
    s=s.replace('capacity_v2/capacity_protocol_FROZEN_v1.json','capacity_v2/capacity_protocol_FROZEN_v2.json')
    s=s.replace('    assert all(torch.count_nonzero(x.last.weight)==0 for x in module.maps)','    assert torch.count_nonzero(module.map[-1].weight)==0 and float(module.gain.detach())==0.')
    diag='[float(x.last.weight.detach().norm()) for x in module.maps]';assert diag in s
    s=s.replace(diag,'module_diagnostics(module)').replace('relation_projection_norms','scene_scale_diagnostics')
    s=s.replace('native ECDet plus the same A52 marginal relation bias; no GT or teacher at inference','native ECDet plus the fixed A54 scale head/readout/reference-size update; no GT, telemetry or teacher at inference')
    marker='\n\ndef train(arm):'
    helper="""

def module_diagnostics(module):
    return {'current_gain':float(module.gain.detach().tanh()),
            'head_weight_norm':float(module.map[-1].weight.detach().norm()),
            'last_forward':dict(module.latest),
            'last_field_coefficients':dict(getattr(module,'field_observation',{}))}

def attach_field_observer(module):
    def observe(head,args,output):
        with torch.no_grad():
            a=module.coefficients(output.detach())
            module.field_observation={'mean_coefficients':a.mean(0).cpu().tolist(),
               'mean_abs_coefficients':a.abs().mean(0).cpu().tolist()}
    module.map.register_forward_hook(observe)
"""
    assert marker in s;s=s.replace(marker,helper+marker)
    s=s.replace('    del initial,actual','    assert extra_sha.hexdigest()==capacity[\'results\'][0][\'new_initial_tensors_sha256\']\n    attach_field_observer(module)\n    del initial,actual')
    s=s.replace('flow=hashlib.sha256();loss_sum=0.;ts=time.monotonic()','flow=hashlib.sha256();loss_sum=0.;scale_loss_sum=0.;ts=time.monotonic()')
    s=s.replace('value=float(loss.detach());loss_sum+=value',"value=float(loss.detach());loss_sum+=value;scale_loss_sum+=float(losses['loss_a54_scene_scale'].detach())")
    s=s.replace("'mean_loss':loss_sum/809,","'mean_loss':loss_sum/809,'mean_scale_auxiliary_loss':scale_loss_sum/809,")
    assert 'module.maps' not in s and 'relation_' not in s
    ast.parse(s);(P/'train_A54.py').write_text(s,encoding='utf-8')
    sup=(PREV/'supervise_A52.py').read_text().replace('train_A52.py','train_A54.py').replace('Own A52','Own A54')
    ast.parse(sup);(P/'supervise_A54.py').write_text(sup,encoding='utf-8')
    print(json.dumps({'trainer_sha256':sha(P/'train_A54.py'),'supervisor_sha256':sha(P/'supervise_A54.py'),'fixed_epochs_each':12,'new_parameters_each':6210,'algorithm_code_unchanged_since_capacity':True}))
if __name__=='__main__':main()
