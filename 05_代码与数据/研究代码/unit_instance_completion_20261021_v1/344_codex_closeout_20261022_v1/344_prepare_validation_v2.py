"""Use the installed Pillow's enum names; preserve the failed source and receipt."""
import importlib,difflib
C=importlib.import_module('344_context')
old=(C.HERE/'344_validate_1466.py').read_text(encoding='utf-8')
new=old.replace('Image.Resampling.BILINEAR','Image.BILINEAR').replace('Image.Resampling.NEAREST','Image.NEAREST')
new=new.replace('344_result_1466独立验收与6PNG.json','344_result_1466独立验收与6PNG_v2.json').replace('344_result_1466独立验收_实现失败.json','344_result_1466独立验收_实现失败_v2.json')
C.text('344_validate_1466_v2.py',new)
C.text('344_Pillow兼容修正.diff',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='344_validate_1466.py',tofile='344_validate_1466_v2.py')))
