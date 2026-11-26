"""Read BASE12 buffers and regenerate primitive anchor math on CPU, no model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect(checkpoint, checkpoint_sha, source, source_sha, output, seed):
    import torch
    torch.set_num_threads(1)
    checkpoint, source, output = map(Path, (checkpoint, source, output))
    assert not output.exists() and not torch.cuda.is_initialized()
    before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns, sha(checkpoint))
    assert before[2] == checkpoint_sha and sha(source) == source_sha
    text = source.read_text(encoding='utf8')
    for fragment in ('eps=1e-2', 'grid_size=0.05', 'valid_mask.to(memory.dtype) * memory',
                     'anchors = torch.where(valid_mask, anchors, torch.inf)'):
        assert fragment in text, fragment
    started = time.monotonic()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['last_epoch'] == 11
    levels, raw, offsets, level_stats = [(80, 80), (40, 40), (20, 20)], [], 0, []
    for level, (height, width) in enumerate(levels):
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)
        grid = (grid.unsqueeze(0) + .5) / torch.tensor([width, height], dtype=torch.float32)
        wh = torch.ones_like(grid) * .05 * (2. ** level)
        anchors = torch.concat([grid, wh], dim=-1).reshape(-1, height * width, 4)
        mask = ((anchors > .01) * (anchors < 1 - .01)).all(-1, keepdim=True)
        invalid = (~mask[0, :, 0]).nonzero().flatten().tolist()
        level_stats.append(dict(level=level, stride=8 * 2 ** level, height=height, width=width,
            invalid_count=len(invalid), global_invalid_indices=[offsets + i for i in invalid],
            invalid_row_column=[[i // width, i % width] for i in invalid]))
        raw.append(anchors)
        offsets += height * width
    raw = torch.concat(raw, dim=1)
    valid = ((raw > .01) * (raw < 1 - .01)).all(-1, keepdim=True)
    expected = torch.log(raw / (1 - raw))
    expected = torch.where(valid, expected, torch.inf)
    groups = {}
    for name, values in [('model', state['model']), ('EMA', state['ema']['module'])]:
        anchors, mask = values['decoder.anchors'], values['decoder.valid_mask']
        assert anchors.shape == (1, 8400, 4) and mask.shape == (1, 8400, 1)
        assert anchors.device.type == mask.device.type == 'cpu'
        finite = torch.isfinite(expected) & torch.isfinite(anchors)
        groups[name] = dict(anchor_shape=list(anchors.shape), mask_shape=list(mask.shape),
            mask_dtype=str(mask.dtype), anchor_dtype=str(anchors.dtype),
            mask_equal=torch.equal(mask, valid), anchor_equal=torch.equal(anchors, expected),
            invalid_count=int((~mask).sum()), positive_inf_elements=int(torch.isposinf(anchors).sum()),
            finite_max_abs_difference=float((anchors[finite] - expected[finite]).abs().max()),
            nonfinite_pattern_equal=torch.equal(torch.isfinite(anchors), torch.isfinite(expected)))
    matched = all(g['mask_equal'] and g['anchor_equal'] for g in groups.values())
    assert before == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns, sha(checkpoint))
    assert sha(source) == source_sha and not torch.cuda.is_initialized()
    report = dict(schema='native_anchor_buffer_cpu_v1', seed=seed,
        status='SAVED_BUFFERS_MATCH_NATIVE_640_ANCHORS' if matched else 'SAVED_BUFFERS_DIFFER',
        checkpoint_sha256=checkpoint_sha, native_decoder_source_sha256=source_sha,
        reader_sha256=sha(__file__), groups=groups, levels=level_stats,
        elapsed_seconds=time.monotonic()-started, torch=str(torch.__version__),
        new_model_constructions=0, new_model_forwards=0, new_optimizer_updates=0, new_epochs=0,
        CUDA_initialized=False,
        scope='Only saved buffers and source-equivalent primitive math. Masked grid positions are not a hard exclusion of all nearby GT: other query regressions can reach border objects. No root-cause or AP claim.')
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'checkpoint_sha', 'source', 'source_sha', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(**vars(args)), allow_nan=False), flush=True)
