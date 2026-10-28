"""Split resources shared by source checkouts and installed wheels."""

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

DATASETS = ("PolyU", "Tongji", "HFUT", "Blue", "Red", "Green")


@contextmanager
def split_file(dataset, protocol, phase, splits_dir=None):
    if dataset not in DATASETS or protocol not in ("open", "closed"):
        raise ValueError("Unsupported dataset or protocol")
    if phase not in ("train", "test"):
        raise ValueError("Split phase must be train or test")
    name = f"{phase}_{dataset}.txt"
    if splits_dir is not None:
        yield Path(splits_dir) / protocol / name
    else:
        resource = files("dnu_ss").joinpath("splits").joinpath(protocol).joinpath(name)
        with as_file(resource) as path:
            yield path
