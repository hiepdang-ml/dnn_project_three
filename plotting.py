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
    height: int = groundtruths.shape[2]
    width: int = groundtruths.shape[3]
    aspect_ratio: float = width / height

    for idx in range(predictions.shape[0]):
        image: torch.Tensor = images[idx]
        groundtruth: torch.Tensor = groundtruths[idx]
        prediction: torch.Tensor = predictions[idx]
        fig, axs = plt.subplots(3, 1, figsize=(10, 30))
        axs[0].imshow(
            image.squeeze(dim=0),
            aspect=aspect_ratio, origin="lower",
            extent=[0., 1., 0., 1.],
            cmap='gray',
        )
        axs[0].set_title(f'$image$', fontsize=20)
        axs[1].imshow(
            groundtruth.squeeze(dim=0),
            aspect=aspect_ratio, origin="lower",
            extent=[0., 1., 0., 1.],
            cmap='gray',
        )
        axs[1].set_title(f'$prediction$\n${notes[idx]}$', fontsize=20)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.85, wspace=0.05)
        timestamp: dt.datetime = dt.datetime.now()
        fig.savefig(
            f"{os.getenv('PYTHONPATH')}/results/{timestamp.strftime('%Y%m%d%H%M%S')}"
            f"{timestamp.microsecond // 1000:03d}.png"
        )
        plt.close(fig)

