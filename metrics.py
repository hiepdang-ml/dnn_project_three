import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert torch.all((groundtruths == 0) | (groundtruths == 1))
        groundtruths: torch.Tensor = groundtruths.float()
        probabilities: torch.Tensor = torch.sigmoid(input=logits)
        numerator: torch.Tensor = 2. * (probabilities * groundtruths).sum() + 1.
        denorminator: torch.Tensor = (probabilities + groundtruths).sum() + 1.
        return 1. - numerator / denorminator


class IOU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, groundtruths: torch.Tensor) -> torch.Tensor:
        assert torch.all((groundtruths == 0) | (groundtruths == 1))
        groundtruths: torch.Tensor = groundtruths.int()
        probabilities: torch.Tensor = torch.sigmoid(input=logits)
        predictions: torch.Tensor = (probabilities > 0.5).int()
        intersection: torch.Tensor = predictions & groundtruths
        union: torch.Tensor = predictions | groundtruths
        return intersection.sum() / union.sum()

