"""Descriptive finite-action loss diagnostics; no model execution or selection fitting."""
from pathlib import Path
from collections import Counter
import json,numpy as np
root=Path('/root/c10_region_restore_20261029_v1/restoration_probe_v1')
out={}
for arm in ['D0','DA']:
    rows=[json.loads(s) for s in (root/(arm+'_image_diagnostics.jsonl')).read_text().splitlines()]
    assert len(rows)==100 and len({r['image_id'] for r in rows})==100
    losses=np.asarray([r['losses'] for r in rows]);assert losses.shape==(100,19) and np.isfinite(losses).all()
    def summary(key):
        selections={name:np.asarray([r[key]['choices'][name] for r in rows]) for name in rows[0][key]['choices']}
        values=np.asarray([r['fixed_assignment_losses'] if key=='fixed_assignment' else r['losses'] for r in rows])
        true=values[:,:16].min(1)
        return dict(selection_counts={name:dict(Counter(map(int,v))) for name,v in selections.items()},
                    independent_vs_quadratic_different_images=int(np.count_nonzero(selections['independent']!=selections['quadratic'])),
                    mean_loss_regret={name:float(np.mean(values[np.arange(100),idx]-true)) for name,idx in selections.items()},
                    mean_holdout_additive_MAE=float(np.mean([r[key]['holdout_additive_MAE'] for r in rows])),
                    mean_holdout_quadratic_MAE=float(np.mean([r[key]['holdout_quadratic_MAE'] for r in rows])))
    out[arm]=dict(images=100,mean_baseline_loss=float(losses[:,0].mean()),mean_full_restoration_utility=float(np.mean(losses[:,0]-losses[:,15])),
                   full_restoration_loss_positive_images=int(np.count_nonzero(losses[:,15]<losses[:,0])),
                   dynamic=summary('dynamic'),fixed_assignment=summary('fixed_assignment'),
                   dynamic_vs_fixed_quadratic_different_images=sum(r['dynamic']['choices']['quadratic']!=r['fixed_assignment']['choices']['quadratic'] for r in rows))
(root/'loss_analysis.json').write_text(json.dumps(dict(status='DESCRIPTIVE_ONLY',arms=out,detector_forwards=0,training_updates=0),indent=2))
print(json.dumps(out))
