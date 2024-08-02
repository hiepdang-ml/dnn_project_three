import os
from typing import List

import datetime as dt
import matplotlib.pyplot as plt
import torch



def plot_predictions(
    images: torch.Tensor,
    groundtruths: torch.Tensor,
    predictions: torch.Tensor,
    notes: List[str],
) -> None:

    assert groundtruths.shape == predictions.shape
    assert groundtruths.ndim == 4   # (batch_size, n_channels, height, width)
    assert groundtruths.shape[1] == 1, 'Expect n_channels to be 1'
    assert notes is None or len(notes) == groundtruths.shape[0]

    os.makedirs(f"{os.getenv('PYTHONPATH')}/results", exist_ok=True)

    images = images.to(device=torch.device('cpu'))
    groundtruths = groundtruths.to(device=torch.device('cpu'))
    predictions = predictions.to(device=torch.device('cpu'))

    # Ensure that the plot respect the tensor's shape

    for idx in range(predictions.shape[0]):
        image: torch.Tensor = images[idx]
        groundtruth: torch.Tensor = groundtruths[idx]
        prediction: torch.Tensor = predictions[idx]
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        axs[0].imshow(
            image.squeeze(dim=0),
            cmap='gray',
        )
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title(f'$image$', fontsize=20)
        axs[1].imshow(
            groundtruth.squeeze(dim=0),
            cmap='gray',
        )
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title(f'$groundtruth$', fontsize=20)
        prediction: torch.Tensor = (torch.sigmoid(input=prediction) > 0.5).int()
        axs[2].imshow(
            prediction.squeeze(dim=0),
            cmap='gray',
        )
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title(f'$prediction - {notes[idx]}$', fontsize=20)
        fig.subplots_adjust(hspace=0.1)
        fig.tight_layout()
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{os.getenv('PYTHONPATH')}/results/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)

