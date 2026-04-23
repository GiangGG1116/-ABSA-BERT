import torch
import pytest
from absa.utils import set_seed, accuracy, avg


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_accuracy_all_correct():
    preds  = torch.tensor([0, 1, 2])
    labels = torch.tensor([0, 1, 2])
    assert accuracy(preds, labels) == pytest.approx(1.0)


def test_accuracy_all_wrong():
    preds  = torch.tensor([0, 0, 0])
    labels = torch.tensor([1, 2, 1])
    assert accuracy(preds, labels) == pytest.approx(0.0)


def test_accuracy_partial():
    preds  = torch.tensor([0, 1, 2, 0])
    labels = torch.tensor([0, 1, 0, 0])
    assert accuracy(preds, labels) == pytest.approx(0.75)


def test_avg_normal():
    assert avg([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_avg_empty():
    assert avg([]) == pytest.approx(0.0)


def test_avg_single():
    assert avg([5.0]) == pytest.approx(5.0)
