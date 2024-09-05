import pickle as pkl
from pathlib import Path

import blosc2


def blosc_pkl_load(path: Path):  # load *.blosc file
    assert path.suffix == '.blosc'
    with path.open('rb') as f:
        compressed_pickled_data = f.read()
    pickled_data = blosc2.decompress(compressed_pickled_data)
    obj = pkl.loads(pickled_data)
    return obj


def blosc_pkl_dump(obj, path: Path):  # dump *.blosc file
    assert path.suffix == '.blosc'
    pickled_data = pkl.dumps(obj)
    compressed_pickled_data = blosc2.compress(pickled_data)
    if not path.parent.exists():
        path.mkdir(parents=True)
    with path.open('wb') as fw:
        fw.write(compressed_pickled_data)
    return


if __name__ == '__main__':
    # convert *.pkl to *.blosc
    dir = Path('dataset/supply/concealer/fga/GCN/benign')
    for path in dir.glob('*.pkl'):
        with path.open('rb') as f:
            obj = pkl.load(f)
        blosc_pkl_dump(obj, path.with_suffix('.blosc'))
        # NOTE delete *.pkl
        path.unlink()
