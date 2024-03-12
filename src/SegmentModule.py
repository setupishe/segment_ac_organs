import lightning as L
import torchvision.models.convnext as convnext
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import torch
from statistics import mean
import torchmetrics
from torchvision import transforms as T
import albumentations as A
import json
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score

from UNet import UNet


default_config_file = '../configs/default.json'
with open(default_config_file, 'r') as file:
    config = json.load(file)


def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
    """Smooth Dice coefficient."""
    ground_truth_oh = F.one_hot(ground_truths, num_classes=num_classes)
    prediction_norm = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)
    intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)
    summation = prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (summation + smooth)

    dice_mean = dice.mean()

    return 1.0 - dice_mean

class SegmentModule(L.LightningModule):
    def __init__(self,  config=config):
        super().__init__()
        self.config = config
        self.used_classes = config['used_classes']
        if config['architecture'] == 'unet':
            self.model = UNet(config['in_channels'], len(self.used_classes))
            
        self.metric_fn = MulticlassF1Score(num_classes=len(self.used_classes), 
                                        average="macro")
        self.loss_fn = dice_coef_loss

    def process_batch(self, batch, mode=None):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_fn(outputs, labels, len(self.used_classes))
        preds = F.softmax(outputs, dim=1)
        metric = self.metric_fn(preds, labels)
        if mode is not None:
            self.log(f'{mode}_loss', loss, on_epoch=True, logger=True)
            self.log(f'{mode}_metric', metric, on_epoch=True, logger=True) #TODO: проверить усреднение

        return loss, preds
    
    
    def training_step(self, train_batch):
        loss, _ = self.process_batch(train_batch, 'train')
        return loss
    
    def validation_step(self, val_batch):
        loss, _ = self.process_batch(val_batch, 'val')
        return loss
    
    def test_step(self, test_batch):
        loss, _ = self.process_batch(test_batch, 'test')
        return loss
    
    def forward(self, x):
        return self.model(x)
    

    def predict_step(self, batch, batch_idx):
        _, preds = self.process_batch(batch)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.config["learning_rate"],
                                # weight_decay=self.config["weight_decay"],
                                )
        # scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {"optimizer": optimizer, 
                # "lr_scheduler": scheduler, 
                }