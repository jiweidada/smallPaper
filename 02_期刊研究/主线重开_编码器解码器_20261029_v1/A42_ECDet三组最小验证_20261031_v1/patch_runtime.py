import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parent
S=R/'source/EdgeCrafter-d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecdetseg'
rows=[]
def patch(relative,old,new):
    p=S/relative;t=p.read_text();assert t.count(old)==1,relative
    before=hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text(t.replace(old,new),encoding='utf-8')
    rows.append({'file':relative,'before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'old':old,'new':new})

patch('engine/edgecrafter/decoder.py',
'''            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach''',
'''            detail_delta = None
            if getattr(self, 'detail_reader', None) is not None and i == self.eval_idx:
                assert self.detail_feature is not None
                detail_delta = self.detail_reader(self.detail_feature, ref_points_detach).to(output.dtype)
            regression_input = output + output_detach
            if detail_delta is not None:
                regression_input = regression_input + detail_delta
            pred_corners = bbox_head[i](regression_input) + pred_corners_undetach''')
patch('engine/edgecrafter/decoder.py',
'''                scores = score_head[i](output)''',
'''                classification_input = output
                if detail_delta is not None and self.detail_mode == 'S':
                    classification_input = classification_input + detail_delta
                scores = score_head[i](classification_input)''')
patch('engine/edgecrafter/criterion.py',
'''        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}''',
'''        nonfinite = [k for k, v in losses.items() if not torch.isfinite(v).all()]
        if nonfinite:
            raise FloatingPointError('A42 raw loss nonfinite before native nan_to_num: ' + str(nonfinite))
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}''')
(R/'isolated_source_patch_receipt.json').write_text(json.dumps({'changes':rows,'native_math_when_B_and_finite_losses':'unchanged; compatibility and fail-fast checks only','classification_input_in_L':'unchanged; native LQE score coupling retained'},indent=2),encoding='utf-8')
print('A42 isolated source patch installed:',len(rows),'exact replacements',flush=True)
