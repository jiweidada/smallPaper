from pathlib import Path
import hashlib,json,py_compile
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'A49_下一主线有界筛选_20261101_v1'
protocol_sha='a301ec5fb1f6a5c4675ef129f11e8359fa46c5d49640f636ccbeb343703cb8e8'
old_sha='d3a971a21d1c339052c8a7ef41fcab345882edc8893c4762c66dbbc5bc10f8f3'
text=(OLD/'collect_completed_arm_A49.py').read_text(encoding='utf-8')
text=text.replace('/root/uav_a49_20261101_v1','/root/uav_a51_20261101_v1')
text=text.replace("choices=['B','M','Q']","choices=['M','Q']")
text=text.replace("    assert result['native_inference_no_mask_reads'] is True\n    assert result['extra_mask_core_reads']==(38832 if a=='Q' else 0)",
 "    assert result['model_parameters']==9876858 and result['extra_parameters']==73746")
text=text.replace('protocol_FROZEN.json','protocol_FROZEN_v1.json').replace(old_sha,protocol_sha)
text=text.replace("files=sorted(p for p in folder.rglob('*') if p.is_file())+[protocol]",
 "files=sorted(p for p in folder.rglob('*') if p.is_file())+[protocol,ROOT/'conservative_scale_exchange.py',ROOT/'runtime_A51.py',ROOT/'train_A51.py',ROOT/'capacity_result.json',ROOT/'capacity_protocol_FROZEN_v1.json']")
dest=ROOT/'collect_completed_arm_A51.py';assert not dest.exists();dest.write_text(text,encoding='utf-8')
text=(OLD/'verify_completed_arm_local.py').read_text(encoding='utf-8')
text=text.replace("choices=['B','M','Q']","choices=['M','Q']").replace(old_sha,protocol_sha)
text=text.replace("    assert result['native_inference_no_mask_reads'] is True\n    assert result['extra_mask_core_reads']==(38832 if a=='Q' else 0)",
 "    assert result['model_parameters']==9876858 and result['extra_parameters']==73746")
old="    initial=torch.load(initial_path,map_location='cpu',weights_only=True)['model']"
new=old+"\n    from conservative_scale_exchange import ScaleExchange\n    with torch.random.fork_rng(devices=[]):\n        torch.manual_seed(20261102)\n        module=ScaleExchange(192,a)\n    h=hashlib.sha256()\n    for key,value in module.state_dict().items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())\n    initialization=json.loads((folder/'initialization.json').read_text())\n    assert h.hexdigest()==initialization['new_module_initialization_sha256'], 'CPU RNG/version differs: obtain exact remote module init before claiming state comparison'\n    initial.update({'encoder.a51_exchange.'+k:v for k,v in module.state_dict().items()})"
assert old in text;text=text.replace(old,new)
text=text.replace("'extra_mask_core_reads':result['extra_mask_core_reads'],'native_inference_no_mask_reads':True,",
 "'model_parameters':9876858,'extra_parameters':73746,'inference_includes_exchange_module':True,")
text=text.replace("assert max(steps)==9708 and min(steps)>0", "assert dict(steps)=={9708:490}, dict(steps)")
text=text.replace("checks[label]={'changed_state_tensors':changed", "extra_changes={k:float((v-initial[k]).abs().max()) for k,v in state.items() if k.startswith('encoder.a51_exchange.')}\n        assert all(x>0 for x in extra_changes.values()),extra_changes\n        checks[label]={'new_module_changes':extra_changes,'changed_state_tensors':changed")
dest=ROOT/'verify_completed_arm_A51_local.py';assert not dest.exists();dest.write_text(text,encoding='utf-8')
for n in ['collect_completed_arm_A51.py','verify_completed_arm_A51_local.py']:
    py_compile.compile(str(ROOT/n),doraise=True)
result={'status':'PREPARED_ONLY_NOT_EXECUTED','files':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['collect_completed_arm_A51.py','verify_completed_arm_A51_local.py']},
 'no_detector_execution':True,'new_optimizer_updates':0,'scope':'future completed12 archives/checkpoint integrity, not prediction rerun/novelty proof'}
(ROOT/'result_tools_preparation_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
