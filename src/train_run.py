import sys
import os

from OrgansUtils import *

from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import albumentations as A

from SegmentModule import SegmentModule
from OrgansDataset import OrgansDataset
from MlflowMaintenance import check_mlflow_server
import datetime

config_file = sys.argv[1]
default_config_file = '../configs/default.json'


    
config = load_config(config_file)
default_config = load_config(default_config_file)
for key in default_config:
    if key not in config:
        config[key] = default_config[key]
        
config_diff = {key: value for key, value in config.items() 
               if default_config[key] != value}

config_part = "_".join([f'{key}-{config[key]}' for key in config_diff])

run_name = f"Run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{config_part}"

augs_path = os.path.join("../augs", config['augs'])
augs = A.load(augs_path, data_format='yaml')

dataset_path = os.path.join('../data', config['dataset_path'])
batch_size = config['batch_size']

model = SegmentModule(config)

train_dataset = OrgansDataset(
    os.path.join(dataset_path, 'train'), 
    img_size=config['img_size'],
    used_classes=config['used_classes'],
    augs=augs,
    clip_max=config['clip_max'],
    clip_min=config['clip_min'],
    )
val_dataset = OrgansDataset(
    os.path.join(dataset_path, 'val'),
    img_size=config['img_size'],
    used_classes=config['used_classes'],
    clip_max=config['clip_max'],
    clip_min=config['clip_min'],
    )


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=12, shuffle=True)

check_mlflow_server()
mlf_logger = MLFlowLogger(experiment_name="organs_segmentation", 
                          run_name=run_name,
                          tracking_uri="http://127.0.0.1:8081")
from lightning.pytorch import seed_everything
seed_everything(config['seed'], workers=True)


val_every_n_epochs = 1

checkpoint_callback = ModelCheckpoint(
        dirpath="../checkpoints",
        filename=os.path.join("../checkpoints", 
                              run_name,
                              "organs_segmentator_{epoch:02d}"),
        every_n_epochs=config['save_period'],
        save_top_k=-1,
    )

trainer = L.Trainer(max_epochs=config['epochs'], 
                    logger=mlf_logger, 
                    callbacks=[checkpoint_callback],     
                    check_val_every_n_epoch=config['save_period'],
                    deterministic=True)

trainer.fit(model, train_loader, val_loader, )

