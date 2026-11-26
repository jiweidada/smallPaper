"""Fresh isolated DEIM training with native UAVDT validation and safe output binding."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from uavdt_protocol import install_native_adapter
from uavdt_stage2 import install_stage2_checkpointing
install_native_adapter();install_stage2_checkpointing()
import train

def validate_paths(args):
    root=Path(__file__).resolve().parent.parent
    assert args.output_dir,'Controlled UAVDT training requires an explicit unique output directory'
    output=Path(args.output_dir).resolve();arm=Path(args.config).stem.removesuffix('_full96')
    assert output.parent==(root/'outputs').resolve()
    assert output.name==f'uavdt_{arm}_full96_seed{args.seed}'
    if args.resume:
        assert Path(args.resume).resolve().parent==output,'Native resume overrides output_dir: checkpoint must belong to this exact job'
    else:assert not output.exists(),'Never increment into a second directory or overwrite an existing run'

if __name__=='__main__':
    args=train.build_arg_parser().parse_args();validate_paths(args);train.main(args)
