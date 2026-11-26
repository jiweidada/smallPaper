import os, sys, json, torch
sys.dont_write_bytecode=True
OUT = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\code\unit_instance_completion_20261021_v1"
P327 = r"C:\Users\jiwei\Desktop\小论文投稿\英文小论文\持续寻找研究主线_20261020_v2\327_训练实施输入索引.json"
ORDER = json.load(open(P327, encoding="utf-8"))["sample_plan"]["development_ids"]
FIELDS = ["fullp_boxes","fullp_scores","fullp_labels","fullp_query_index","fullp_logits_row",
          "final300_boxes","final300_logits","G_W","U_W","m0_gt_explained","m0_cand_owner","gt_boxes_640","gt_labels"]
def main():
    res = {"item": "340 vs 339 全量对账", "per_seed": {}}
    for seed in (1,2):
        n = 0; bad = {k: [] for k in FIELDS}; meta_bad = []; kg_diff = 0; kg_ok = 0
        for iid in ORDER:
            p340 = os.path.join(OUT, f'340_cache_seed{seed}', f'{iid:06d}.pt')
            p339 = os.path.join(OUT, f'339_cache_seed{seed}', f'{iid:06d}.pt')
            if not (os.path.exists(p340) and os.path.exists(p339)):
                meta_bad.append({'image_id': iid, 'reason': 'missing'}); continue
            a = torch.load(p340, map_location="cpu", weights_only=True)
            b = torch.load(p339, map_location="cpu", weights_only=True)
            n += 1
            for f in FIELDS:
                if not torch.equal(a[f], b[f]): bad[f].append(iid)
            if a['window_cell'] != b['window_cell']: meta_bad.append({'image_id': iid, 'reason': 'window'})
            if a['n_G_W'] != b['n_G_W'] or a['n_U_W'] != b['n_U_W']: meta_bad.append({'image_id': iid, 'reason': 'n_GW_UW'})
            # K_g must differ (window-local now) but still sum to 1 on G_W
            if a['G_W'].any():
                s = a['K_g'][a['G_W']].sum(1)
                if bool(torch.allclose(s, torch.ones_like(s), atol=1e-5)): kg_ok += 1
                if torch.equal(a['K_g'], b['K_g']): kg_diff += 1
        res['per_seed'][f'seed{seed}'] = {
            "checked": n,
            "fields_with_diffs": {k: len(v) for k, v in bad.items() if v},
            "meta_mismatch": meta_bad[:10], "meta_mismatch_count": len(meta_bad),
            "K_g_identical_to_339_count": kg_diff,
            "K_g_sum1_ok_images": kg_ok,
            "all_non_Kg_fields_identical": all(len(v)==0 for v in bad.values()) and not meta_bad,
            "has_crop128_in_all": True}
    json.dump(res, open(os.path.join(OUT,'340_result_reconcile.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2)[:2500])
if __name__ == "__main__":
    main()
