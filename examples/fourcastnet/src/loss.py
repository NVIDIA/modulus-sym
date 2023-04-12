import torch
from typing import Dict

Tensor = torch.Tensor


class LpLoss(torch.nn.Module):
    def __init__(
        self,
        d: float = 2.0,
        p: float = 2.0,
    ):
        """Relative Lp loss normalized seperately in the batch dimension.
        Expects inputs of the shape [B, C, ...]

        Parameters
        ----------
        p : float, optional
            Norm power, by default 2.0
        """
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert p > 0.0
        self.p = p

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> float:
        num_examples = x.size()[0]

        xv = x.reshape(num_examples, -1)
        yv = y.reshape(num_examples, -1)

        diff_norms = torch.linalg.norm(xv - yv, ord=self.p, dim=1)
        y_norms = torch.linalg.norm(yv, ord=self.p, dim=1)

        return torch.mean(diff_norms / y_norms)

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, float]:
        losses = {}
        for key, value in pred_outvar.items():
            losses[key] = self._rel(pred_outvar[key], true_outvar[key])
        return losses
