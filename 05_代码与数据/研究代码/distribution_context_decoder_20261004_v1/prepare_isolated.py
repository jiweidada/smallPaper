"""One-time construction, isolated directory only; never edits upstream."""
import hashlib
import json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent
p = root/'native/engine/deim/dfine_decoder.py'
s = p.read_text(encoding='utf8')
assert 'distribution_update' not in s
s = s.replace('query_pos_embed=None): #', 'query_pos_embed=None, distribution_update=None): #', 1)
anchor = 'target = self.norm1(target)'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor+'\n        if distribution_update is not None:\n            target = distribution_update(target)')
s = s.replace('dn_meta=None):', 'dn_meta=None, distribution_update=None):', 1)
anchor = 'output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)'
assert s.count(anchor) == 1
s = s.replace(anchor, '''update = None
            if i == 1 and distribution_update is not None:
                # Closure executes synchronously in layer2 after masked self-attn.
                update = lambda q: distribution_update(q, pred_corners_undetach.detach(),
                    ref_points_initial, project, reg_scale)
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask,
                           query_pos_embed, distribution_update=update)''')
anchor = 'def forward(self, feats, targets=None):'
assert s.count(anchor) == 1
s = s.replace(anchor, 'def forward(self, feats, targets=None, distribution_update=None):')
# Only public native forward; preserve the unused RT-DETRv3 path unchanged.
i = s.index('def forward(self, feats, targets=None, distribution_update=None):')
before, after = s[:i], s[i:]
after = after.replace('dn_meta=dn_meta)', 'dn_meta=dn_meta, distribution_update=distribution_update)', 1)
p.write_text(before+after, encoding='utf8')
p = root/'factory.py'
s = p.read_text(encoding='utf8').replace('from dual_view import DualViewDEIM',
    'from dual_view import DualViewDEIM\nfrom dual_distribution import DualDistributionDEIM')
s = s.replace("return DualViewDEIM(backbone, encoder, decoder, cfg['arm'], cfg['fusion_width'])",
    "return DualDistributionDEIM(backbone, encoder, decoder, cfg['arm'])")
s = s.replace("not k.startswith('interaction.')", "not k.startswith(('interaction.', 'distribution_interaction.'))")
p.write_text(s, encoding='utf8')
p = root/'dual_view.py'
s = p.read_text(encoding='utf8').replace('for k in a.keys() | b.keys()', 'for k in sorted(a.keys() | b.keys())')
p.write_text(s, encoding='utf8')
template = yaml.safe_load((root/'configs/dual_independent.yml').read_text())
for arm in 'ABCDE':
    cfg = dict(template, arm=arm, priority='distribution_context_reviewed_candidate',
               candidate='distribution_context_v1', protocol='protocol_v1.json')
    (root/f'configs/distribution_{arm}.yml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf8')
manifest = json.loads((root/'native_source_manifest.json').read_text(encoding='utf8'))
for row in manifest['files']:
    row['sha256'] = hashlib.sha256((root/row['local']).read_bytes()).hexdigest()
manifest['isolation_note'] = 'Only dfine_decoder optional layer2 callback differs; upstream receipt retained.'
(root/'native_source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
print('Isolated native callback, factory and five arm configs prepared')
