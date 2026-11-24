"""REQ-Q-01 leakage-free evaluation: folds, chronological validation, no shared recordings."""
import numpy as np
import pytest

from shm import config as C
from shm.splits import LeakageError, assert_no_overlap, check_folds, chrono_split, stage2_fold


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    tags = np.repeat(np.arange(14), 50)
    month = np.where(tags == 13, C.EXTERNAL_MONTH, "2021_06").astype(object)
    month[(tags == 12)[:]] = np.where(rng.random((tags == 12).sum()) < 0.5, C.EXTERNAL_MONTH, "2022_09")
    tkey = rng.permutation(len(tags))
    return tags, tkey, month


def test_fold_definition_is_valid():
    check_folds()
    hidden = sorted(t for v in C.TAG_FOLDS.values() for t in v)
    assert hidden == [2, 3, 4, 5, 7, 8, 10, 11, 12]


@pytest.mark.parametrize("fold", list(C.TAG_FOLDS))
def test_hidden_tags_never_in_training(fold):
    tags, tkey, month = _toy()
    s = stage2_fold(fold, tags, tkey, month)
    assert not np.isin(tags[s["train"]], C.TAG_FOLDS[fold]).any()
    assert not np.isin(tags[s["val"]], C.TAG_FOLDS[fold]).any()
    assert set(np.unique(tags[s["test"]])) <= set(C.TAG_FOLDS[fold])
    assert not (month[s["train"]] == C.EXTERNAL_MONTH).any()


def test_validation_is_chronologically_last():
    tags, tkey, _ = _toy()
    idx = np.arange(len(tags))
    tr, va = chrono_split(idx, tags, tkey)
    for t in np.unique(tags):
        assert tkey[tr[tags[tr] == t]].max() < tkey[va[tags[va] == t]].min()


def test_overlap_is_detected():
    with pytest.raises(LeakageError):
        assert_no_overlap(train=[1, 2, 3], test=[3, 4])
