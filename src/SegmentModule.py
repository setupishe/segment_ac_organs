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
from OrgansUtils import *

default_config_file = '../configs/default.json'
config = load_config(default_config_file)


class TverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions, ground_truths):
        ground_truth_oh = F.one_hot(ground_truths, num_classes=self.num_classes).float()
        predictions = F.softmax(predictions, dim=1).permute(0, 3, 1, 2)

        TP = (predictions * ground_truth_oh).sum(dim=(2, 3))
        FP = (predictions * (1 - ground_truth_oh)).sum(dim=(2, 3))
        FN = ((1 - predictions) * ground_truth_oh).sum(dim=(2, 3))

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        Tversky_loss = 1 - Tversky.mean()

        return Tversky_loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, ground_truths):
        ground_truth_oh = F.one_hot(ground_truths, num_classes=self.num_classes).float()
        predictions = F.softmax(predictions, dim=1)
        
        predictions = predictions.permute(0, 2, 3, 1)  
        
        intersection = (predictions * ground_truth_oh).sum(dim=(1, 2))
        summation = predictions.sum(dim=(1, 2)) + ground_truth_oh.sum(dim=(1, 2))
        dice_score = (2.0 * intersection + self.smooth) / (summation + self.smooth)
        
        return 1.0 - dice_score.mean()
    
class SegmentModule(L.LightningModule):
    def __init__(self,  config=config):
        super().__init__()
        self.config = config
        self.used_classes = config['used_classes']
        if config['architecture'] == 'unet':
            self.model = UNet(config['in_channels'], len(self.used_classes))
            
        self.metric_fn = MulticlassF1Score(num_classes=len(self.used_classes), 
                                        average="macro")
        if config['loss'] == 'dice':
            self.loss_fn = DiceLoss(len(self.used_classes))
        elif config['loss'] == 'tversky':
            self.loss_fn = TverskyLoss(len(self.used_classes),
                                       alpha=config['alpha'],
                                       beta=config['beta'],
                                       )
    def on_train_start(self):
        self.logger.log_hyperparams(params=self.config)

    def process_batch(self, batch, mode=None):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_fn(outputs, labels)
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