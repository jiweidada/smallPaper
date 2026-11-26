from pathlib import Path
import zipfile
import numpy as np


def save_npz(destination,arrays):
    """Standard non-pickle NPZ, with unrestricted array key 'file'."""
    with zipfile.ZipFile(Path(destination),'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,value in arrays.items():
            assert '/' not in name and '\\' not in name
            with archive.open(name+'.npy','w') as output:
                np.lib.format.write_array(output,np.asarray(value),allow_pickle=False)
