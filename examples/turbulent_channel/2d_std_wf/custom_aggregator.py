# Import libraries
import torch
from torch import nn
from typing import Dict, List

# Import from Modulus
from modulus.sym.eq.derivatives import gradient
from modulus.sym.loss.aggregator import Aggregator


class CustomSum(Aggregator):
    """
    Loss aggregation by summation
    """

    def __init__(self, params, num_losses, weights=None):
        super().__init__(params, num_losses, weights)

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Aggregates the losses by summation

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses
        step : int
            Optimizer step

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        smoothness = 0.0005  # use 0.0005 to smoothen the transition over ~10k steps
        step_tensor = torch.tensor(step, dtype=torch.float32)
        decay_weight = (torch.tanh((20000 - step_tensor) * smoothness) + 1.0) * 0.5

        # Add losses
        for key in losses.keys():
            if "init" not in key:
                loss += (1 - decay_weight) * (losses[key])
            else:
                loss += decay_weight * (losses[key])
        return loss
