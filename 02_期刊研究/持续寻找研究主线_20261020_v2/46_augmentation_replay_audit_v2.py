"""Import the native transform registration modules omitted by the minimal model runtime.
The prior constructor failure read no image and evaluated no candidate.
"""
import importlib.util,pathlib,traceback
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('source_audit',root/'45_augmentation_replay_audit.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from engine.data.transforms import mosaic,_transforms
m.OUT=root/'46_augmentation_replay_audit_v2'
if __name__=='__main__':
 try:m.main()
 except BaseException:
  m.OUT.mkdir(exist_ok=True)
  m.n.save_json(m.OUT/'failure.json',{'traceback':traceback.format_exc(),'new_training':False})
  raise
