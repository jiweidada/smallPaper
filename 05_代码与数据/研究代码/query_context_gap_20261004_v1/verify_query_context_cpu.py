"""Falsifiable synthetic CPU checks; never a VRAM or detection-performance test."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import torch
from query_context import QueryContext, grid_boxes, support_geometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(37)
    checks = []
    def check(name, condition, **measurements):
        row = dict(name=name, passed=bool(condition), **measurements)
        checks.append(row)
        print(json.dumps(row), flush=True)
        if not condition:
            raise AssertionError(name)
    result = dict(status='running_CPU_only', checks=checks,
                  synthetic_data_only=True, GPU_capacity_measured=False, APs_improvement_demonstrated=False,
                  torch=torch.__version__, created=datetime.datetime.now().astimezone().isoformat())
    # Reserve a new result file; never overwrite a failure or an earlier run.
    with args.output.open('x', encoding='utf8') as h:
        json.dump(result, h, indent=2)
    try:
        cells = grid_boxes((40, 40), torch.device('cpu'))
        crop = torch.tensor([[640/1920, 192/1080, 1280/1920, 832/1080]])
        refs = torch.tensor([[[.5, .5, .02, .02], [20/640, .5, .02, .02]]])
        geom = support_geometry(refs, crop, cells, 'gap')
        check('known interior and boundary gap ratios', torch.allclose(geom['rho'], torch.tensor([[0., .421875]]), atol=1e-6), rho=geom['rho'].tolist())
        covered_area = (geom['fraction'] / 1600).sum(-1)
        check('fractional grid weights conserve analytic gap area', torch.allclose(covered_area, geom['gap_area'], atol=2e-7))
        narrow = refs[:, 1:].clone(); narrow[..., 0] = 127.99 / 640
        gn = support_geometry(narrow, crop, cells, 'gap')
        check('subcell narrow gap is retained', bool(gn['rho'].item() > 0 and gn['fraction'].sum() > 0))
        edge = torch.tensor([[0., 192/1080, 640/1920, 832/1080]])
        ge = support_geometry(refs[:, 1:], edge, cells, 'gap')
        check('original-image exterior is not treated as missing context', ge['rho'].item() == 0)
        padded_crop = torch.tensor([[0., 0., 640/500, 640/480]])
        pad_ref = torch.tensor([[[550/640, 240/640, .1, .1]]])
        gp = support_geometry(pad_ref, padded_crop, cells, 'gap')
        check('query center in padding is invalid', not gp['valid'].item() and gp['fraction'].sum().item() == 0)
        mirror_crop = crop.clone(); mirror_crop[:, 0] = 1-crop[:, 2]; mirror_crop[:, 2] = 1-crop[:, 0]
        mirror_refs = refs.clone(); mirror_refs[..., 0] = 1-refs[..., 0]
        gm = support_geometry(mirror_refs, mirror_crop, cells, 'gap')
        check('horizontal flip preserves gap and mirrors support',
              torch.allclose(gm['rho'], geom['rho'], atol=1e-6) and
              torch.allclose(gm['fraction'].reshape(1,2,40,40).flip(-1), geom['fraction'].reshape(1,2,40,40), atol=3e-5))
        # Random image sizes/crops/references: independent area conservation and bounds.
        for width, height, x, y in [(2560,1440,817,431),(900,700,260,60),(480,300,0,0)]:
            c = torch.tensor([[x/width,y/height,(x+640)/width,(y+640)/height]])
            r = torch.rand(1,79,4); r[...,2:] *= .7
            g = support_geometry(r,c,cells,'gap')
            check(f'area conservation and bounds {width}x{height}',
                  torch.allclose((g['fraction']/1600).sum(-1), g['gap_area']*g['valid'], atol=4e-7) and
                  bool(((g['rho']>=0)&(g['rho']<=1)).all()))
        block = QueryContext(dim=32, attention_dim=16, heads=4)
        q = torch.randn(1,2,32,requires_grad=True)
        feature = torch.randn(1,32,40,40,requires_grad=True)
        learned_refs = refs.clone().requires_grad_()
        output = block(q,learned_refs,crop,block.prepare_memory(feature),'gap')
        check('zero initialization is exact identity', torch.equal(output,q))
        (output.square().sum()).backward()
        check('first backward learns output projection', block.output.weight.grad.abs().sum().item()>0)
        check('geometry has no new regression gradient', learned_refs.grad is None)
        check('zero output delays inner projection gradient as expected', block.value.weight.grad.abs().sum().item()==0)
        with torch.no_grad(): block.output.weight.normal_(0,.02)
        block.zero_grad(set_to_none=True); feature.grad=None
        block.trace_enabled=True
        memory = block.prepare_memory(feature)
        output = block(q,learned_refs,crop,memory,'gap')
        check('rho-zero query unchanged at injection after learning', torch.equal(output[:,0],q[:,0]))
        check('boundary query actually receives context after learning', not torch.equal(output[:,1],q[:,1]))
        output.square().sum().backward()
        check('learned block carries gradient to global memory and QKV',
              feature.grad.abs().sum().item()>0 and all(getattr(block,name).weight.grad.abs().sum().item()>0 for name in ('query','key','value')))
        check('empty support chooses null exactly', block.last_trace['null_weight'][0,0].item()==1.)
        with torch.no_grad():
            memory = block.prepare_memory(feature.detach())
            original = block(q.detach(),refs,crop,memory,'gap')
            modified = {k:v.clone() for k,v in memory.items()}
            forbidden = (geom['fraction']==0).all(1)[0]
            modified['key'][:,:,forbidden] += 100
            modified['value'][:,:,forbidden] += 100
            changed = block(q.detach(),refs,crop,modified,'gap')
        check('forbidden memory locations cannot affect injection', torch.equal(original,changed))
        # Variable Q includes potential denoising queries; not hard-coded to 300.
        with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
            mixed = block(torch.randn(2,517,32),torch.rand(2,517,4),crop.expand(2,-1),
                          block.prepare_memory(torch.randn(2,32,40,40)),'gap')
        check('variable query count and CPU autocast are finite', mixed.shape==(2,517,32) and torch.isfinite(mixed).all())
        counts={}
        for mode in ['global','gap','local','support','gap_unit']:
            local_crop = crop if mode=='local' else None
            o = block(q.detach(),refs,crop,block.prepare_memory(feature.detach(),local_crop),mode)
            counts[mode]=sum(p.numel() for p in block.parameters())
            check(f'{mode} arm finite', torch.isfinite(o).all())
        check('attention controls share identical parameter counts',len(set(counts.values()))==1,parameters=counts)
        result['status']='passed_CPU_geometry_and_module_only'
    except BaseException as e:
        result.update(status='technical_failure_preserved',error=repr(e))
        raise
    finally:
        result['checks']=checks
        result['implementation_sha256']=hashlib.sha256(Path(__file__).with_name('query_context.py').read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__': main()
