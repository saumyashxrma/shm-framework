"""Leakage-free evaluation splits (identical to notebook cell C1, "Fix 2").

* Leave-damage-out folds: each fold hides three damage states (tags) completely; the fold's models never see them.
* Inside the training pool, the chronologically last 20 % of every tag is the validation set (no random shuffling,
  so near-identical consecutive recordings cannot sit on both sides of a split).
* The external month (October 2022) is never used for training or model selection.
"""
from __future__ import annotations

import numpy as np

from . import config as C


class LeakageError(AssertionError):
    """Raised when a split would let test information reach training."""


def chrono_split(idx, tags, tkey, val_frac: float = C.VAL_FRAC):
    """Per tag: earliest (1 - val_frac) by timestamp -> train, the rest -> validation. Returns sorted index arrays."""
    idx, tags, tkey = np.asarray(idx), np.asarray(tags), np.asarray(tkey)
    tr, va = [], []
    for t in np.unique(tags[idx]):
        it = idx[tags[idx] == t]
        it = it[np.argsort(tkey[it], kind="stable")]
        cut = int(round((1 - val_frac) * len(it)))
        tr.append(it[:cut])
        va.append(it[cut:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def assert_no_overlap(**parts) -> None:
    """Raise LeakageError if any two named index sets share a recording."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = np.intersect1d(np.asarray(parts[a]), np.asarray(parts[b]))
            if len(common):
                raise LeakageError(f"{a} and {b} share {len(common)} recordings")


def stage2_fold(fold: str, tags, tkey, month):
    """Stage-2 (damage level) split for one leave-damage-out fold. Returns dict(train, val, test, ext)."""
    tags, month = np.asarray(tags), np.asarray(month)
    hidden = C.TAG_FOLDS[fold]
    is_ext = month == C.EXTERNAL_MONTH
    damaged = tags > 0
    pool = np.where(damaged & ~is_ext & ~np.isin(tags, hidden))[0]
    tr, va = chrono_split(pool, tags, tkey)
    te = np.where(damaged & ~is_ext & np.isin(tags, hidden))[0]
    ext = np.where(is_ext)[0]
    assert_no_overlap(train=tr, val=va, test=te, external=ext)
    if np.isin(tags[tr], hidden).any() or np.isin(tags[va], hidden).any():
        raise LeakageError(f"fold {fold}: a hidden tag reached the training pool")
    return {"train": tr, "val": va, "test": te, "ext": ext}


def check_folds() -> None:
    """The three folds must hide disjoint tags; tag 0 (healthy), 1 (first state) and 13 (external) are never hidden."""
    all_hidden = [t for v in C.TAG_FOLDS.values() for t in v]
    if len(all_hidden) != len(set(all_hidden)):
        raise LeakageError("a tag is hidden in more than one fold")
    if {0, 1, 13} & set(all_hidden):
        raise LeakageError("tags 0, 1 and 13 must not be hidden")
