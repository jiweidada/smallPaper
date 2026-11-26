"""Create train342.py exclusively, with audited diff against train341.py."""
import importlib, difflib
C = importlib.import_module('342_context')

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old,new)

src = (C.WORK/'train341.py').read_text(encoding='utf-8')
dst = src.replace('"""341 training link = train340 + 317v4 (G class_null lr x10). CPU only; NO GPU.',
    '"""342 CPU pretraining review link; no real-training entry point. Based on train341.')
dst = replace_once(dst, 'import core as core311',
    "import geom342 as geom\nfrom importlib import import_module\nhead342 = import_module('342_head')\nimport core as core311")
old = '''        ix = torch.clamp(torch.minimum(b[:,2], W[2]) - torch.maximum(b[:,0], W[0]), min=0)
        iy = torch.clamp(torch.minimum(b[:,3], W[3]) - torch.maximum(b[:,1], W[1]), min=0)
        inter = ix * iy
        keep = inter > 0'''
dst = replace_once(dst,old,'''        keep = geom.intersecting_candidates(b, W)''')
dst = replace_once(dst,'''        loc = torch.stack([bb[:,0]-W[0], bb[:,1]-W[1], bb[:,2]-W[0], bb[:,3]-W[1]], -1)''',
    '''        loc = geom.win_norm_xyxy(bb, W)''')
dst = dst.replace('window-local xyxy coordinates (unclipped per 317)',
                  'window-normalized xyxy coordinates (unclipped per 317)')
dst = dst.replace('# window-local xyxy','# window-normalized xyxy')
dst = replace_once(dst,'return GeometryCalibratedHead()','return head342.GeometryCalibratedHead342()')
dst = replace_once(dst,'return core311.CompletionHead(arm)','return head342.CompletionHead342(arm)')
start = '''            x0,y0,x1,y1 = sel[:,0],sel[:,1],sel[:,2],sel[:,3]
            ww = float(W[2]-W[0]); wh = float(W[3]-W[1])
            ubox = torch.stack([(x0+x1)/2-W[0].item(), (y0+y1)/2-W[1].item(),
                                x1-x0, y1-y0], -1) / torch.tensor([ww,wh,ww,wh])'''
dst = replace_once(dst,start,'            ubox = geom.win_norm_cxcywh(sel, W)')
start = '''                x0,y0,x1,y1 = sel[:,0],sel[:,1],sel[:,2],sel[:,3]
                ww=float(W[2]-W[0]); wh=float(W[3]-W[1])
                ubox = torch.stack([(x0+x1)/2-W[0].item(),(y0+y1)/2-W[1].item(), x1-x0, y1-y0],-1)/torch.tensor([ww,wh,ww,wh])'''
dst = replace_once(dst,start,'                ubox = geom.win_norm_cxcywh(sel, W)')
old = '''        ix = torch.clamp(torch.minimum(b[:,2], W[2]) - torch.maximum(b[:,0], W[0]), min=0)
        iy = torch.clamp(torch.minimum(b[:,3], W[3]) - torch.maximum(b[:,1], W[1]), min=0)
        keep = (ix*iy) > 0'''
dst = replace_once(dst,old,'        keep = geom.intersecting_candidates(b, W)')
# Exact implementation corrections to explicit 296 formulas; no label, loss
# weight, target, window, or scientific hyperparameter is adjusted.
dst = replace_once(dst,'per_image_denoms.append(max(1, int(UW.sum())))',
    'per_image_denoms.append(int(UW.sum()))')
dst = replace_once(dst,''' for t in det_terms) / len(batch["entries"])''',
    ''' for t in det_terms)\n    if denom_mode == "per_image_mean":\n        ld = ld / len(batch["entries"])''')
dst = replace_once(dst,'''        if GW.sum() == 0:
            lm.append(torch.tensor(0.0)); continue
''','')
dst = replace_once(dst,'''        term = torch.where(mstar > 0, term, torch.zeros_like(term))
''','''        # At m*=0 only 0*log(0) vanishes; the +m penalty remains (296 §5).
''')
dst = replace_once(dst,'''    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "epoch": epoch, "shuffle_seed": shuffle_seed,
                "rng": rng, "lr": opt.param_groups[0]["lr"]}, path)''',
'''    with open(path, "xb") as checkpoint_file:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "epoch": epoch, "shuffle_seed": shuffle_seed,
                    "rng": rng, "lr": opt.param_groups[0]["lr"]}, checkpoint_file)''')
dst = replace_once(dst,'''    for g in opt.param_groups: g["lr"] = ck["lr"]
''', '''    # load_state_dict restores EACH group's LR, including G's x10 group.
''')
dst = replace_once(dst,'''    log = []
    for k in range(steps):''', '''    assert 0 <= steps <= 20 and total_steps == 1218, "CPU smoke only"
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert sum(len(b["entries"]) for b in batches) <= 32, "development smoke limit"
    log = []
    for k in range(steps):''')
with open(C.HERE/'train342.py','x',encoding='utf-8',newline='\n') as f:
    f.write(dst)
C.write_text('342_train342_与341.diff',''.join(difflib.unified_diff(src.splitlines(True),dst.splitlines(True),
    fromfile=str(C.WORK/'train341.py'),tofile=str(C.HERE/'train342.py'))))
print('created train342.py and diff')
