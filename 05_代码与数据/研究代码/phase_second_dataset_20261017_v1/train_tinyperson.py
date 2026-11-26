"""Use the frozen native training loop with a separately registered TinyPerson evaluator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_protocol import install_native_adapter
install_native_adapter()
import train

if __name__ == '__main__':
    train.main(train.build_arg_parser().parse_args())

