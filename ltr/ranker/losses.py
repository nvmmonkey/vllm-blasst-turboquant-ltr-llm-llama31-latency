"""ListMLE listwise ranking loss for the OPT-125M output-length ranker (Track B).

The prior LTR work [11][12] trains the ranker with **ListMLE** to predict the
order in which requests should run (shortest predicted output first → SJF-like).
ListMLE is the negative log-likelihood of the ground-truth ordering under the
Plackett-Luce model: for the true order o_1..o_n,

    L = - Σ_i [ s_{o_i} - logsumexp(s_{o_i}, ..., s_{o_n}) ]

Lower loss ⇔ the predicted scores rank items in the ground-truth order.

torch is imported lazily so importing this module (and the rest of the harness)
never requires a DL backend; the training path pulls it in.
"""

from __future__ import annotations


def listmle_loss(scores, relevance, *, eps: float = 1e-10):
    """ListMLE loss for ONE list (1-D tensors).

    Args:
        scores: (n,) predicted scores; higher ⇒ should rank earlier.
        relevance: (n,) ground-truth key; higher ⇒ should rank earlier. For
            shortest-job-first use e.g. ``-output_length`` so shorter outputs
            rank first.
    Returns:
        A scalar tensor (the loss). Numerically stabilised via max-subtraction.
    """
    import torch

    scores = scores.reshape(-1)
    relevance = relevance.reshape(-1)
    order = torch.argsort(relevance, descending=True)
    s = scores[order]

    s_max = s.max().detach()
    exp = torch.exp(s - s_max)
    # reverse cumulative sum → Σ_{j>=i} exp(s_j - s_max)
    rev_cumsum = torch.flip(torch.cumsum(torch.flip(exp, dims=[0]), dim=0), dims=[0])
    logcumsumexp = torch.log(rev_cumsum + eps) + s_max
    return torch.sum(logcumsumexp - s)


def listmle_loss_batched(scores, relevance, *, eps: float = 1e-10):
    """Mean ListMLE over a batch of equal-length lists.

    ``scores`` / ``relevance`` are (B, L); returns the mean per-list loss.
    """
    import torch

    losses = [listmle_loss(scores[i], relevance[i], eps=eps) for i in range(scores.shape[0])]
    return torch.stack(losses).mean()
