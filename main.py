import torch
import torch.nn as nn
from torch.optim import Adam

from datasets import BirdSoundDataset
from models import Encoder, BottleNeck, Decoder, UNet
from workers import Trainer, Predictor


# Initialize the training datasets
train_dataset = BirdSoundDataset(
    dataroot='data/train',
    resolution=(128, 512),
)
val_dataset = BirdSoundDataset(
    dataroot='data/valid',
    resolution=(128, 512),
)
test_dataset = BirdSoundDataset(
    dataroot='data/test',
    resolution=(128, 512),
)

# Load model
device: torch.device = torch.device('cuda')
# net: nn.Module = UNet(encoder=Encoder(), bottleneck=BottleNeck(), decoder=Decoder())
# optimizer = Adam(params=net.parameters(), lr=0.001)

# trainer = Trainer(
#     model=net, optimizer=optimizer,
#     train_dataset=train_dataset, val_dataset=val_dataset,
#     train_batch_size=16, val_batch_size=4,
#     device=device,
# )
# trainer.train(
#     n_epochs=100, patience=5,
#     tolerance=0., checkpoint_path='./checkpoints/',
#     save_frequency=5,
# )


from utils import CheckpointLoader
model, optimizer = CheckpointLoader(r'checkpoints_old/epoch20.pt').load(scope=globals())
predictor = Predictor(model=model, device=device)
test_iou: float = predictor.predict(dataset=test_dataset)
print(test_iou)


