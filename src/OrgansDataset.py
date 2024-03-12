from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from OrgansUtils import *

class OrgansDataset(Dataset):
    def __init__(self,
                dataset_path: str, 
                img_size: int,
                used_classes,
                augs: A.Compose | None = None,
                cache: bool = False,
                clip_min: int | None = None,
                clip_max: int | None = None,
                ):
        super().__init__()
        self.use_cache = cache
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.clip_min = clip_min
        self.clip_max = clip_max

        for img_path in glob.glob(dataset_path + '/**/*img.npy', recursive=True):
            lbl_path = img2label(img_path)
            self.images.append(load_npy(img_path) if self.use_cache else img_path)
            self.labels.append(load_npy(img_path) if self.use_cache else lbl_path)


        self.setup_transforms(augs)
        self.setup_mapping(used_classes)

    def setup_mapping(self, used_classes):
        self.class_mapping = np.zeros(len(labels_dict), dtype=int)
        for new_class, original_class in enumerate(used_classes):
            self.class_mapping[original_class] = new_class
        self.reverse_class_mapping = np.array(used_classes)
        
    def setup_transforms(self, augs):

        transforms = []
        if augs is not None:
            transforms.extend(augs.transforms)
        transforms.extend([
                        A.Resize(self.img_size, self.img_size, always_apply=True),
                        ToTensorV2(always_apply=True)
                    ])
        self.transforms = A.Compose(transforms)

    def map_classes(self, labels):
        return self.class_mapping[labels.astype(int)]
        
    def reverse_map_classes(self, labels):
        return self.reverse_class_mapping[labels.astype(int)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]

        if not self.use_cache:
            image = load_npy(image)
            label = load_npy(label)
        
        image = normalize(image, 
                          min_val=self.clip_min,
                          max_val=self.clip_max,
                          )
        image = np.expand_dims(image, 2)
        label = self.map_classes(label)
        transformed = self.transforms(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"].to(torch.long)
        return image, label