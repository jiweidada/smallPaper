"""Read installed metadata before/after imports; no model or GPU computation."""
import importlib.metadata as m,json,sys,sysconfig
def snap():
    names=['packaging','tomli','wheel']
    return {'all_distributions':{d.metadata['Name']:d.version for d in m.distributions()},
            'direct_versions':{n:m.version(n) for n in names},'sys_path':sys.path[:],
            'locations':[{ 'name':d.metadata['Name'],'version':d.version,'root':str(d.locate_file(''))}
                         for d in m.distributions() if d.metadata['Name'].lower() in names]}
out={'before_imports':snap(),'installed_site_packages':{d.metadata['Name']:d.version for d in m.distributions(path=[sysconfig.get_path('purelib')])}}
import torch
out['after_torch']=snap()
import torchvision
out['after_torchvision']=snap()
import packaging
out['actual_packaging_import']={'file':packaging.__file__,'version':packaging.__version__}
out['model_executions']=0
print(json.dumps(out))
