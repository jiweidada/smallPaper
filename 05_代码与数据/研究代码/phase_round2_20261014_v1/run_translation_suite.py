"""Reuse Python imports across all inference-only shifts of one checkpoint."""
import argparse
import gc
import pathlib
import subprocess
import sys
import dump_preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--zero-predictions', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    destination = pathlib.Path(args.out)
    destination.mkdir(parents=True, exist_ok=False)
    offsets = [(value, 0) for value in [-32,-8,-4,-2,-1,1,2,4,8,32]]
    offsets += [(0, value) for value in [-32,-8,-4,-2,-1,1,2,4,8,32]]
    for dx, dy in offsets:
        name = f'dx{dx}_dy{dy}'
        export = destination/name
        sys.argv = ['dump_preds.py', '--config', args.config, '--checkpoint', args.checkpoint,
                    '--out', str(export), '--shift', str(dx), str(dy)]
        dump_preds.main()
        subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('analyze_preds.py')),
                        '--annotations', args.annotations, '--reference', args.zero_predictions,
                        '--candidate', str(export/'predictions.json'), '--border', '32',
                        '--out', str(destination/(name+'_analysis'))], check=True)
        import torch
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
