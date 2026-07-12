"""CPU tests for the ListMLE ranking loss (Track B).

Covers the README §8 requirement: ListMLE loss on a toy batch decreases.
torch is imported via importorskip so a minimal CI (no torch) skips these.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ltr.ranker.losses import listmle_loss, listmle_loss_batched  # noqa: E402


def test_loss_is_nonnegative():
    relevance = torch.tensor([3.0, 2.0, 1.0, 0.0])
    scores = torch.tensor([1.0, 0.5, 0.2, 0.0])
    assert listmle_loss(scores, relevance).item() >= 0.0


def test_correct_order_beats_reversed_order():
    relevance = torch.tensor([3.0, 2.0, 1.0, 0.0])  # item 0 should rank first
    good = torch.tensor([10.0, 5.0, 2.0, 0.0])       # scores agree with relevance
    bad = torch.tensor([0.0, 2.0, 5.0, 10.0])        # scores reversed
    assert listmle_loss(good, relevance).item() < listmle_loss(bad, relevance).item()


def test_loss_decreases_under_optimization():
    torch.manual_seed(0)
    relevance = torch.tensor([3.0, 2.0, 1.0, 0.0])
    scores = torch.zeros(4, requires_grad=True)
    opt = torch.optim.SGD([scores], lr=0.5)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        loss = listmle_loss(scores, relevance)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first  # ListMLE loss went down as scores learned the order


def test_batched_mean_matches_manual_average():
    relevance = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    batched = listmle_loss_batched(scores, relevance).item()
    manual = (
        listmle_loss(scores[0], relevance[0]).item()
        + listmle_loss(scores[1], relevance[1]).item()
    ) / 2
    assert batched == pytest.approx(manual, rel=1e-5)
