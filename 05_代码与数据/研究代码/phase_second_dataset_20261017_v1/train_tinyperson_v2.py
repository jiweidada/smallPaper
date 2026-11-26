"""Isolated native trainer, official evaluator, explicit last-12-round selection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_protocol import install_native_adapter
from tiny_stage2 import install_stage2_checkpointing
install_native_adapter(); install_stage2_checkpointing()
import train
if __name__ == '__main__': train.main(train.build_arg_parser().parse_args())

