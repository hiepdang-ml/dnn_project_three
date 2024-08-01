from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.optim import Optimizer

from utils import Accumulator, EarlyStopping, Timer, Logger, CheckpointSaver
from datasets import BirdSoundDataset
from metrics import SoftDiceLoss, IOU
from plotting import plot_predictions

class Trainer:

    def __init__(
        self, 
        model: nn.Module,
        optimizer: Optimizer,
        train_dataset: BirdSoundDataset,
        val_dataset: BirdSoundDataset,
        train_batch_size: int,
        val_batch_size: int,
        device: torch.device,
    ):
        self.model: nn.Module = model.to(device=device)
        self.optimizer: Optimizer = optimizer
        self.train_dataset: BirdSoundDataset = train_dataset
        self.val_dataset: BirdSoundDataset = val_dataset
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.device: torch.device = device

        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False)
        self.loss_function: nn.Module = SoftDiceLoss()
        self.evaluation_metric: nn.Module = IOU(cutoff_probability=0.5)

    def train(
        self, 
        n_epochs: int,
        patience: int,
        tolerance: float,
        checkpoint_path: Optional[str] = None,
        save_frequency: int = 5,
    ) -> None:
        
        train_accumulator = Accumulator()
        early_stopping = EarlyStopping(patience, tolerance)
        timer = Timer()
        logger = Logger()
        checkpoint_saver = CheckpointSaver(
            model=self.model,
            optimizer=self.optimizer,
            dirpath=checkpoint_path,
        )
        self.model.train()
        
        # loop through each epoch
        for epoch in range(1, n_epochs + 1):
            timer.start_epoch(epoch)
            # Loop through each batch
            for batch, (batch_image, batch_groundtruth) in enumerate(self.train_dataloader, start=1):
                timer.start_batch(epoch, batch)
                assert batch_image.ndim == 4
                batch_size, n_channels, height, width = batch_image.shape
                batch_image: torch.Tensor = batch_image.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                self.optimizer.zero_grad()
                batch_prediction: torch.Tensor = self.model(input=batch_image)
                assert batch_prediction.shape == batch_groundtruth.shape
                dice_loss = self.loss_function(
                    logits=batch_prediction, groundtruths=batch_groundtruth,
                )
                dice_loss.backward()
                self.optimizer.step()

                # Accumulate the train metrics
                with torch.no_grad():
                    iou = self.evaluation_metric(
                        logits=batch_prediction, groundtruths=batch_groundtruth,
                    )

                train_accumulator.add(
                    dice_loss=dice_loss.item(), 
                    iou=iou.item(),
                )
                timer.end_batch(epoch=epoch)
                logger.log(
                    epoch=epoch, n_epochs=n_epochs, 
                    batch=batch, n_batches=len(self.train_dataloader), 
                    took=timer.time_batch(epoch, batch), 
                    train_dice_loss=train_accumulator['dice_loss'] / batch,  
                    train_iou=train_accumulator['iou'] / batch, 
                )
        
            # Ragularly save checkpoint
            if checkpoint_path is not None and epoch % save_frequency == 0:
                checkpoint_saver.save(
                    model_states=self.model.state_dict(), 
                    optimizer_states=self.optimizer.state_dict(),
                    filename=f'epoch{epoch}.pt',
                )
            
            # Reset metric records for next epoch
            train_accumulator.reset()

            # Evaluate
            val_dice_loss, val_iou = self.evaluate()
            timer.end_epoch(epoch)
            logger.log(
                epoch=epoch, n_epochs=n_epochs, 
                took=timer.time_epoch(epoch), 
                val_dice_loss=val_dice_loss, val_iou=val_iou,
            )
            print('=' * 20)

            early_stopping(val_dice_loss)
            if early_stopping:
                print('Early Stopped')
                break

        # Always save last checkpoint
        if checkpoint_path:
            checkpoint_saver.save(self.model, filename=f'epoch{epoch}.pt')

    def evaluate(self) -> float:
        val_accumulator = Accumulator()
        self.model.eval()
        with torch.no_grad():
            # Loop through each batch
            for batch, (batch_image, batch_groundtruth) in enumerate(self.val_dataloader, start=1):
                assert batch_image.ndim == 4
                batch_size, n_channels, height, width = batch_image.shape
                batch_image: torch.Tensor = batch_image.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor = self.model(input=batch_image)
                assert batch_prediction.shape == batch_groundtruth.shape
                
                dice_loss = self.loss_function(
                    logits=batch_prediction, target=batch_groundtruth,
                )
                iou = self.evaluation_metric(
                    logits=batch_prediction, target=batch_groundtruth,
                )
                # Accumulate the val metrics
                val_accumulator.add(
                    val_dice_loss=dice_loss.item(),
                    val_iou=iou.item(),
                )

        # Compute the aggregate metrics
        val_dice_loss: float = val_accumulator['val_dice_loss'] / batch
        val_iou: float = val_accumulator['val_iou'] / batch
        return val_dice_loss, val_iou


class Predictor:

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model: nn.Module = model.to(device=device)
        self.device: torch.device = device
        self.loss_function: nn.Module = SoftDiceLoss()
        self.evaluation_metric: nn.Module = IOU(cutoff_probability=0.5)

    def predict(self, dataset: BirdSoundDataset) -> None:
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # sample-level method, not batch-level

        batch_images: List[torch.Tensor] = []
        batch_groundtruths: List[torch.Tensor] = []
        batch_predictions: List[torch.Tensor] = []
        metric_notes: List[str] = []

        with torch.no_grad():
            # Loop through each batch
            for batch_image, batch_groundtruth in dataloader:
                assert batch_image.ndim == 4
                batch_image: torch.Tensor = batch_image.to(device=self.device)
                batch_groundtruth: torch.Tensor = batch_groundtruth.to(device=self.device)
                batch_prediction: torch.Tensor = self.model(input=batch_image)
                assert batch_prediction.shape == batch_groundtruth.shape

                dice_loss = self.loss_function(
                    logits=batch_prediction, target=batch_groundtruth,
                )
                iou = self.evaluation_metric(
                    logits=batch_prediction, target=batch_groundtruth,
                )
                batch_images.append(batch_image)
                batch_groundtruths.append(batch_groundtruth)
                batch_predictions.append(batch_prediction)
                metric_notes.append(f'Dice Loss: {dice_loss:.4f}, IoU: {iou:.4f}')

            images = torch.cat(tensors=batch_images, dim=0)
            predictions = torch.cat(tensors=batch_predictions, dim=0)
            groundtruths = torch.cat(tensors=batch_groundtruths, dim=0)
            assert predictions.shape == groundtruths.shape
            # Plot the prediction
            plot_predictions(
                images=images,
                groundtruths=groundtruths, 
                predictions=predictions, 
                notes=metric_notes, 
            )
