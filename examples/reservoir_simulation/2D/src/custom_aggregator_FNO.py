"""
//////////////////////////////////////////////////////////////////////////////
Copyright (C) NVIDIA Corporation.  All rights reserved.

NVIDIA Sample Code

Please refer to the NVIDIA end user license agreement (EULA) associated

with this source code for terms and conditions that govern your use of

this software. Any use, reproduction, disclosure, or distribution of

this software and related documentation outside the terms of the EULA

is strictly prohibited.

//////////////////////////////////////////////////////////////////////////////
@Author: Clement Etienam
"""
import torch
from typing import Dict

# Import from Modulus
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
        decay_weight1 = (torch.tanh((10000 - step_tensor) * smoothness) + 1.0) * 0.5
        lambda_pressure = 1.0
        lambda_saturation = 1.0
# # Add losses
        for key in losses.keys():
            if "pressure" in key:
                loss += lambda_pressure*(1-decay_weight1)*((losses[key]))
            if "water_sat" in key : 
                loss += lambda_saturation*(1-decay_weight1) *((losses[key]))

        return loss

