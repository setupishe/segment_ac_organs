import lightning as L
import torchvision.models.convnext as convnext
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch
from statistics import mean
import torchmetrics
from torchvision import transforms as T
import albumentations as A
import json
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score

from UNet import UNet
from UNet_conv_transpose import UNet_conv_transpose
from OrgansUtils import *

default_config_file = '../configs/default.json'
default_config = load_config(default_config_file)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:  # 'none'
            return F_loss

class TverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions, ground_truths):
        ground_truth_oh = F.one_hot(ground_truths, num_classes=self.num_classes).float()
        predictions = F.softmax(predictions, dim=1).permute(0, 2, 3, 1)

        TP = (predictions * ground_truth_oh).sum(dim=(1, 2))
        FP = (predictions * (1 - ground_truth_oh)).sum(dim=(1, 2))
        FN = ((1 - predictions) * ground_truth_oh).sum(dim=(1, 2))

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        Tversky_loss = 1 - Tversky.mean()

        return Tversky_loss


class DiceScore(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, weights=None):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if weights is not None:
            if isinstance(weights, str) and weights == "auto":
                self.weights = weights
            elif isinstance(weights, (list, torch.Tensor)):
                weights_tensor = torch.tensor(weights, dtype=torch.float32) if isinstance(weights, list) else weights.float()
                self.weights = self._invert_weights(weights_tensor)
            else:
                raise ValueError("Weights must be 'auto', a list, or a torch.Tensor.")
        else:
            self.weights = torch.ones(num_classes) / num_classes  

    def _invert_weights(self, weights):
        weights = 1.0 / (weights + self.smooth)  
        weights = weights / weights.sum()  
        return weights

    def forward(self, predictions, ground_truths):
        ground_truth_oh = F.one_hot(ground_truths, num_classes=self.num_classes).float()
        predictions = F.softmax(predictions, dim=1)
        
        predictions = predictions.permute(0, 2, 3, 1)  
        if isinstance(self.weights, str) and self.weights == "auto":
            label_counts = ground_truth_oh.sum(dim=(0, 1, 2))
            weights = label_counts / (label_counts.sum() + self.smooth)  
            weights = self._invert_weights(weights)
        else:
            weights = self.weights
        
        if weights.sum() <= 1 - self.smooth or weights.sum() >= 1 + self.smooth:
            raise Exception(f'weights sum should be 1, got {weights.sum()}')
        intersection = (predictions * ground_truth_oh).sum(dim=(1, 2))
        summation = predictions.sum(dim=(1, 2)) + ground_truth_oh.sum(dim=(1, 2))
        weights = weights.to(predictions.device)
        weighted_dice_score = (2.0 * intersection + self.smooth) / (summation + self.smooth)
        # weighted_dice_score = weighted_dice_score
        # weighted_dice_score *= weights
        dice_score = weighted_dice_score.mean()
        
        return dice_score

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6, weights=None):
        super(DiceLoss, self).__init__()
        self.dice_score = DiceScore(num_classes, smooth, weights)

    def forward(self, predictions, ground_truths):
        dice_score = self.dice_score(predictions, ground_truths)
        dice_loss = 1 - dice_score
        return dice_loss
    
class SegmentModule(L.LightningModule):
    def __init__(self,  config=default_config):
        super().__init__()
        self.config = config
        self.used_classes = config['used_classes']
        if config['architecture'] == 'unet':
            self.model = UNet(config['in_channels'], len(self.used_classes))
        elif config['architecture'] == 'unet_conv_transpose':
            self.model = UNet_conv_transpose(config['in_channels'], len(self.used_classes))


        self.dice = torchmetrics.Dice(num_classes=len(self.used_classes), 
                                      average="macro",
                                      ignore_index=0)
        self.dice_default = torchmetrics.Dice(num_classes=len(self.used_classes))
        if config['loss'] == "dice_ce":
            dice_loss = DiceLoss(len(self.used_classes))
            ce_loss = nn.CrossEntropyLoss(reduce=False)
            self.loss_fn = lambda x, y: dice_loss(x, y) + ce_loss(x, y).mean()
        elif 'dice' in config['loss']:
            match config['loss'].replace('dice', '').replace('_', ''):
                case 'auto':
                    weights = 'auto'
                case 'precalculated':
                    weights = config['class_weights']
                case '':
                    weights = None
                case _:
                    incorrect_loss = config['loss']
                    raise ValueError(f'dice loss should be "dice", "dice_auto" or "dice_precalculated", got "{incorrect_loss}"')
            self.loss_fn = DiceLoss(len(self.used_classes))
        elif config['loss'] == 'tversky':
            self.loss_fn = TverskyLoss(len(self.used_classes),
                                       alpha=config['alpha'],
                                       beta=config['beta'],
                                       )
        elif config['loss'] == 'CE':
            if config["loss_weights"]:
                weights = torch.tensor(loss_weights[np.array(self.used_classes)], 
                                      dtype=torch.float32)
                
                weights = weights.to('cuda')
            else:
                weights = None
            self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        elif config['loss'] == "focal":
            self.loss_fn = FocalLoss(alpha=config.get('focal_alpha', 0.25), 
                                     gamma=config.get('focal_gamma', 2.0))
    def on_train_start(self):
        self.logger.log_hyperparams(params=self.config)

    def process_batch(self, batch, mode=None):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_fn(outputs, labels)
        if self.config['loss'] == 'CE':
            loss = loss.mean()
        preds = F.softmax(outputs, dim=1)
        dice = self.dice(outputs, labels)
        dice_default = self.dice_default(outputs, labels)
        if mode is not None:
            self.log(f'{mode}_loss', loss, on_epoch=True, logger=True)
            self.log(f'{mode}_dice', dice, on_epoch=True, logger=True, on_step=False) #TODO: проверить усреднение
            self.log(f'{mode}_dice_default', dice_default, on_epoch=True, logger=True, on_step=False) #TODO: проверить усреднение

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
        if self.config['optimiser'] == 'Adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                lr=self.config["learning_rate"],
                                # weight_decay=self.config["weight_decay"],
                                )
        elif self.config['optimiser'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                lr=self.config["learning_rate"],
                                weight_decay=self.config["weight_decay"],
                                )
        return_dict = {"optimizer": optimizer}
        scheduler = self.config['scheduler']
        if scheduler is not None:
            if 'exponential' in scheduler:
                gamma = float(scheduler.split('_')[-1])
                return_dict['lr_scheduler'] = ExponentialLR(optimizer, gamma=gamma)
            elif 'reduce_on_plateau':
                return_dict['lr_scheduler'] = ReduceLROnPlateau(optimizer,
                                                                factor=0.3,
                                                                patience=4,
                                                                mode='max'
                                                                )
                return_dict["monitor"] = "val_dice"
        return return_dict