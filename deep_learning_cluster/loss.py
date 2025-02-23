from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from pycox.models import utils
from torchtuples import TupleTree

def cox_ph_loss_sorted(log_h: Tensor, weights: Tensor,events: Tensor, eps: float = 1e-7,d_theta: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    weights=weights.view(-1)
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())+ weights.T.dot(weights).mul(0.5*d_theta)

def cox_ph_loss(log_h: Tensor, weights: Tensor,durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    print(log_h.shape)
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    weights=weights[idx]
    return cox_ph_loss_sorted(log_h,weights, events, eps)



class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """


    def forward(self, log_h: Tensor, weights: Tensor,durations: Tensor, events: Tensor) -> Tensor:
        print(log_h)
        return cox_ph_loss(log_h,weights, durations, events)