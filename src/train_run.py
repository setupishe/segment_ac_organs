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


config_file = sys.argv[1]
with open(os.path.join('../configs', config_file), 'r') as file:
    config = json.load(file)

model = SegmentModule(config)

augs_path = os.path.join("../augs", config['augs'])
augs = A.load(augs_path, data_format='yaml')

dataset_path = os.path.join('../data', config['dataset_path'])
batch_size = config['batch_size']

train_dataset = OrgansDataset(
    os.path.join(dataset_path, 'train'), 
    img_size=config['img_size'],
    used_classes=config['used_classes'],
    augs=augs)
val_dataset = OrgansDataset(
    os.path.join(dataset_path, 'val'),
    img_size=config['img_size'],
    used_classes=config['used_classes'],
    )

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=12)

check_mlflow_server()

mlf_logger = MLFlowLogger(experiment_name="organs_segmentation", tracking_uri="http://127.0.0.1:8081")

from lightning.pytorch import seed_everything
seed_everything(config['seed'], workers=True)



val_every_n_epochs = 1

checkpoint_callback = ModelCheckpoint(
        filename="../checkpointsorgans_segmentator_{epoch:02d}",
        every_n_epochs=config['save_period'],
        save_top_k=-1,
    )

trainer = L.Trainer(max_epochs=config['epochs'], 
                    logger=mlf_logger, 
                    callbacks=[checkpoint_callback],     
                    check_val_every_n_epoch=config['save_period'],
                    deterministic=True)

trainer.fit(model, train_loader, val_loader, )

