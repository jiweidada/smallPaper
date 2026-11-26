"""CPU-only equivalence against two inspected author AST definitions;no project imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import numpy as np
import torch
from torch import nn
from a6_label_geometry import level_eligibility,source_roundtrip,SOURCE_RANGES,STRIDES

ROOT=Path(__file__).resolve().parent

def main():
    p=json.loads((ROOT/'protocol.json').read_text())
    source=ROOT/'foreground_supervision_author.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==p['author_foreground_source_sha256']
    tree=ast.parse(source.read_text())
    names={'coords_fmap2orig','GenTargets'}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert {n.name for n in selected}==names
    # No external import,top-level statement or arbitrary source initializer runs.
    namespace={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    torch.set_num_threads(2)
    generator=namespace['GenTargets'](list(STRIDES),list(SOURCE_RANGES))
    assert sum(v.numel() for v in generator.parameters())==0
    cases=np.array([[0,0,1,1],[3,3,5,5],[4,4,12,12],[1,3,130,5],
                    [0,0,200,200],[100,120,125,155],[300,310,630,639]],dtype=np.float32)
    tests=[]
    for height,width in ((640,640),(800,1333)):
        boxes=source_roundtrip(cases,height,width)
        original=torch.tensor(cases,dtype=torch.float32)
        size=torch.tensor([width,height,width,height],dtype=torch.float32)
        norm=original/size;tl,br=norm[:,:2],norm[:,2:]
        center=(tl+br)*.5;extent=br-tl;expanded=torch.cat([center,extent],dim=1)*size
        torch_roundtrip=torch.cat([expanded[:,:2]-.5*expanded[:,2:],expanded[:,:2]+.5*expanded[:,2:]],dim=1).numpy()
        np.testing.assert_array_equal(boxes,torch_roundtrip)
        outputs=[torch.zeros((1,1,int(np.ceil(height/s)),int(np.ceil(width/s))),dtype=torch.float32) for s in STRIDES]
        for indices in ([i] for i in range(len(boxes))):
            chosen=boxes[indices]
            reference=generator(outputs,torch.tensor(chosen)[None],torch.ones((1,len(chosen)),dtype=torch.float32))
            expected=np.concatenate([level_eligibility(chosen,height,width,s,r)[1].any(axis=1) for s,r in zip(STRIDES,SOURCE_RANGES)])
            np.testing.assert_array_equal(reference[0,:,0].numpy()>0,expected)
            tests.append(dict(height=height,width=width,gt_indices=list(indices),positive=int(expected.sum())))
        reference=generator(outputs,torch.tensor(boxes)[None],torch.ones((1,len(boxes)),dtype=torch.float32))
        expected=np.concatenate([level_eligibility(boxes,height,width,s,r)[1].any(axis=1) for s,r in zip(STRIDES,SOURCE_RANGES)])
        np.testing.assert_array_equal(reference[0,:,0].numpy()>0,expected)
        tests.append(dict(height=height,width=width,gt_indices='all7_overlap_union',positive=int(expected.sum())))
    result=dict(status='PASS_CPU_AUTHOR_TARGET_FUNCTION_EQUIVALENCE',utc=datetime.now(timezone.utc).isoformat(),
                author_source_sha256=p['author_foreground_source_sha256'],tests=tests,
                source_execution_scope='Only verified coords_fmap2orig and parameter-free GenTargets AST definitions on synthetic CPU tensors;not full project or detector.',
                torch_version=torch.__version__,numpy_version=np.__version__,new_full_detector_forwards=0,new_fits=0,detector_updates=0)
    target=ROOT/'author_source_equivalence.json'
    target.open('x').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(tests),torch_version=torch.__version__,new_full_detector_forwards=0,new_fits=0)))

if __name__=='__main__':
    main()
