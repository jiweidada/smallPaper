"""Launch the native trainer with registered P2 model and identical audits."""
import json
import os
import sys
from pathlib import Path
import train
import p2_candidate  # register before YAMLConfig materializes the model
from screen_runtime import install_runtime


def main(args):
    train.dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    train.check_cuda()
    assert not (args.tuning and args.resume)
    update_dict = train.yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k != 'update' and v is not None})
    cfg = train.YAMLConfig(args.config, **update_dict)
    if args.resume or args.tuning: cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    receipt = os.environ['SCREEN_RECEIPT_DIR']
    install_runtime(cfg, receipt)
    solver = train.TASKS[cfg.yaml_cfg['task']](cfg)
    if args.test_only: solver.val()
    else: solver.fit(json.dumps(cfg.__dict__, indent=2, default=str))
    train.dist_utils.cleanup()


if __name__ == '__main__':
    main(train.build_arg_parser().parse_args())
