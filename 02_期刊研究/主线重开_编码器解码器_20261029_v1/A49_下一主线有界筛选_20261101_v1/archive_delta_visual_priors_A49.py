"""Bounded acquisition of three discovered visual delta-memory priors; no model imports."""
import concurrent.futures
import json
from pathlib import Path
from archive_A49_sources_v1 import acquire_paper

ROOT = Path(__file__).resolve().parent
rows = [
    ('A49_GDKVM_ICCV2025_Publisher', 'GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule',
     'https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_GDKVM_Echocardiography_Video_Segmentation_via_Spatiotemporal_Key-Value_Memory_with_Gated_ICCV_2025_paper.pdf',
     'ICCV2025 publisher PDF; discovery snippet only before acquisition, not full method-read.'),
    ('A49_OSA_CVPR2026_Publisher', 'OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement',
     'https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_OSA_Echocardiography_Video_Segmentation_via_Orthogonalized_State_Update_and_Anatomical_CVPR_2026_paper.pdf',
     'CVPR2026 publisher PDF; discovery snippet only before acquisition, not full method-read.'),
    ('A49_DeltaMIL_arxiv2512_19331v1', 'DeltaMIL: Gated Memory Integration for Efficient and Discriminative Whole Slide Image Analysis',
     'https://arxiv.org/pdf/2512.19331v1', 'arXivv1,2025-12-22; original abstract read, not full method.')
]
target = ROOT / 'delta_visual_source_acquisition_v1.json'
assert not target.exists(), 'Inspect original attempt rather than retrying all downloads.'
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(acquire_paper, rows))
with target.open('x', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(json.dumps([{k: r.get(k) for k in ['title','status','sha256','reason']} for r in records], ensure_ascii=False))
