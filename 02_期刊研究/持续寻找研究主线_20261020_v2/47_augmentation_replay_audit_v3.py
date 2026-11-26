"""Use the already installed CPython-3.12 COCO runtime, not old cp310 binaries.
Both earlier constructor failures occurred before image or candidate evaluation.
"""
import importlib.util,pathlib,sys,traceback
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('registered_audit',root/'46_augmentation_replay_audit_v2.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
m=v.m
sys.path.insert(0,str(root.parent/'主线循环验证_20261020_v1/runtime_pycoco'))
m.OUT=root/'47_augmentation_replay_audit_v3'
if __name__=='__main__':
 try:m.main()
 except BaseException:
  m.OUT.mkdir(exist_ok=True)
  m.n.save_json(m.OUT/'failure.json',{'traceback':traceback.format_exc(),'new_training':False})
  raise
