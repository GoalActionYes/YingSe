"""Explicit local dataset modes shared by byte preparation commands."""
from pathlib import Path


def dataset_options(root,expanded=False,multiref=False):
    if expanded and multiref:raise ValueError('Select exactly one dataset mode')
    base=Path(root)/'outputs/adaptive-reference-20261113/tst100k'
    name='cohort-multiref-v1' if multiref else 'cohort-v2' if expanded else 'cohort-v1'
    reuse=['cohort-v2','cohort-v1'] if multiref else ['cohort-v1'] if expanded else []
    return dict(directory=base/name,max_bytes=(2048 if expanded or multiref else 600)*1024**2,reuse=[base/n for n in reuse])
