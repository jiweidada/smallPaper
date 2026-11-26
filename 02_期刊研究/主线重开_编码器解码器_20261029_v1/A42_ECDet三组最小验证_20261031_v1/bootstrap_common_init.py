import pathlib
R=pathlib.Path(__file__).resolve().parent
assert not (R/'common_init_10class.pth').exists()
s=(R/'bootstrap_remote.py').read_text()
start=s.index('sys.path.insert(0,str(source))')
tail=s[start:]
old='assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())'
new='''assert all(torch.isfinite(p).all() for p in model.parameters())
bad={k:v for k,v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()}
assert set(bad) <= {'decoder.anchors'}, list(bad)
if bad:
    anchors=state['decoder.anchors']; valid=state['decoder.valid_mask'].expand_as(anchors)
    assert torch.isfinite(anchors[valid]).all()
    assert torch.isposinf(anchors[~valid]).all()
'''
assert old in tail
tail=tail.replace(old,new).replace("'all_float_weights_finite':True", "'all_float_parameters_finite':True,'expected_positive_infinity_anchor_buffer':bool(bad)")
pre='import hashlib,json,pathlib,sys\nR=pathlib.Path(__file__).resolve().parent\nsource=R/"source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg"\necvit=source/"engine/edgecrafter/ecvit.py"\nbefore=hashlib.sha256(ecvit.read_text().removeprefix("from __future__ import annotations\\n").encode()).hexdigest()\ndef digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()\ndef emit(name,x):\n (R/name).write_text(json.dumps(x,indent=2),encoding="utf-8");print(json.dumps(x),flush=True)\n'
exec(compile(pre+tail,str(R/'bootstrap_common_init.py'),'exec'),{'__file__':str(R/'bootstrap_common_init.py'),'__name__':'__main__'})
