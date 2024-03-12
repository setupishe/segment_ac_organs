import sys
import os
import json

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import albumentations as A

from SegmentModule import SegmentModule
from OrgansDataset import OrgansDataset
from MlflowMaintenance import check_mlflow_server
import datetime

config_file = sys.argv[1]
with open(os.path.join('../configs', config_file), 'r') as file:
    config = json.load(file)

config_part = "_".join([f'{key}-{config[key]}' for key in config])
run_name = f"Run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{config_part}"
model = SegmentModule(config)

augs_path = os.path.join("../augs", config['augs'])
augs = A.load(augs_path, data_format='yaml')

dataset_path = os.path.join('../data', config['dataset_path'])
batch_size = config['batch_size']

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
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=12)

check_mlflow_server()
print(run_name)
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
