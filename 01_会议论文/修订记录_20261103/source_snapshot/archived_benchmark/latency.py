# -*- coding: utf-8 -*-
"""
统一推理时延测试（论文表用）

  python tools/benchmark/latency.py -c <config.yml> [-r <ckpt.pth>] [--half] [--json out.json]

协议：
  输入 640x640、batch=1、预热 50 次、计时 200 次
  分别报告「纯网络」与「网络+后处理（top-k，无 NMS）」两档
  GPU：torch.cuda.Event 计时 + synchronize，排除首帧与异步开销
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
from engine.core import YAMLConfig


def bench(fn, warmup, iters, ctx=None):
    """ctx：上下文管理器工厂。整段「预热+计时」只在最外层进入一次，
    避免把 autocast 上下文创建开销算进单次时延。"""

    def _run():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        out = []
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e))
        return out

    with torch.no_grad():
        if ctx is None:
            ts = _run()
        else:
            with ctx():
                ts = _run()
    ts.sort()
    n = len(ts)
    mean = sum(ts) / n
    var = sum((t - mean) ** 2 for t in ts) / (n - 1) if n > 1 else 0.0
    return {
        'mean_ms': round(mean, 3),
        'median_ms': round(ts[n // 2], 3),
        'min_ms': round(ts[0], 3),
        'p95_ms': round(ts[int(n * 0.95)], 3),
        'std_ms': round(var ** 0.5, 3),
        'fps': round(1000.0 / mean, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', default=None)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--half', action='store_true', help='用 torch.autocast 做 FP16 推理')
    ap.add_argument('--amp', action='store_true', help='同 --half（别名）')
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--no-post', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    cfg = YAMLConfig(args.config, resume=None)
    model = cfg.model.deploy().to(dev).eval()
    n_param = sum(p.numel() for p in model.parameters())

    loaded = 'random-init(仅用于测速，权重不影响 FPS)'
    if args.resume:
        st = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = None
        for k in ('ema', 'model'):
            v = st.get(k)
            if isinstance(v, dict):
                sd = v.get('module', v)
                break
        if sd is None:
            sd = st
        miss, unexp = model.load_state_dict(sd, strict=False)
        loaded = 'ckpt (missing=%d, unexpected=%d)' % (len(miss), len(unexp))

    post = None
    if not args.no_post:
        try:
            post = cfg.postprocessor.to(dev).eval()
        except Exception as ex:
            print('  [warn] 后处理构建失败，跳过 e2e: %s' % ex)

    use_half = args.half or args.amp
    dtype = torch.float16 if use_half else torch.float32
    # FP16 用 autocast 实现：不整体转半精度，避免个别算子/权重 dtype 不匹配
    if use_half:
        import contextlib
        amp_ctx = lambda: torch.autocast('cuda', dtype=torch.float16)  # noqa: E731
    else:
        import contextlib
        amp_ctx = contextlib.nullcontext  # noqa: E731

    x = torch.randn(args.batch, 3, args.size, args.size, device=dev, dtype=torch.float32)
    osz = torch.tensor([[args.size, args.size]], device=dev)

    name = args.tag or os.path.basename(args.config)
    print('=' * 74)
    print('模型     : %s' % name)
    print('配置     : %s' % args.config)
    print('权重     : %s' % loaded)
    print('参数量   : %.3f M' % (n_param / 1e6))
    print('精度     : %s   输入: %dx%d   batch=%d' %
          ('FP16(autocast)' if use_half else 'FP32', args.size, args.size, args.batch))
    print('GPU      : %s' % torch.cuda.get_device_name(0))
    print('预热/计时: %d / %d 次' % (args.warmup, args.iters))
    print('-' * 74)

    result = {
        'tag': name, 'config': args.config, 'resume': args.resume,
        'params_M': round(n_param / 1e6, 3),
        'precision': 'FP16(autocast)' if use_half else 'FP32',
        'size': args.size, 'batch': args.batch,
        'warmup': args.warmup, 'iters': args.iters,
        'gpu': torch.cuda.get_device_name(0),
    }

    r_net = bench(lambda: model(x), args.warmup, args.iters, amp_ctx)
    result['net'] = r_net
    print('纯网络        : mean %7.3f ms  median %7.3f  min %7.3f  p95 %7.3f  std %6.3f  =>  %7.2f FPS'
          % (r_net['mean_ms'], r_net['median_ms'], r_net['min_ms'],
             r_net['p95_ms'], r_net['std_ms'], r_net['fps']))

    if post is not None:
        try:
            r_e2e = bench(lambda: post(model(x), osz), args.warmup, args.iters, amp_ctx)
            result['e2e'] = r_e2e
            print('网络+后处理   : mean %7.3f ms  median %7.3f  min %7.3f  p95 %7.3f  std %6.3f  =>  %7.2f FPS'
                  % (r_e2e['mean_ms'], r_e2e['median_ms'], r_e2e['min_ms'],
                     r_e2e['p95_ms'], r_e2e['std_ms'], r_e2e['fps']))
            print('后处理开销    : %.3f ms (%.1f%%)'
                  % (r_e2e['mean_ms'] - r_net['mean_ms'],
                     100.0 * (r_e2e['mean_ms'] - r_net['mean_ms']) / r_net['mean_ms']))
        except Exception as ex:
            print('  [warn] e2e 测速失败: %s' % ex)

    print('=' * 74)

    if args.json:
        old = []
        if os.path.isfile(args.json):
            try:
                old = json.load(open(args.json, encoding='utf-8'))
            except Exception:
                old = []
        old.append(result)
        json.dump(old, open(args.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('已追加结果到 %s' % args.json)


if __name__ == '__main__':
    main()
