
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import cv2
import albumentations
import pandas as pd

class args:
    batch_size = 16
    image_size = 224


train_df = pd.read_csv('../input/petfinder-pawpularity-score/train.csv')

train_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

class PawPularityDataset(Dataset):
    def __init__(self,image_paths,train_df,augmentations):
        self.image_paths = image_paths
        self.train_df = train_df
        self.features = train_df.columns[1:-1]
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = cv2.imread(self.image_paths[idx])
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        
        target = train_df['Pawpularity'][idx]
        
        features = list(train_df.loc[0])[1:-1]
        
        return (
            torch.tensor(image,dtype = torch.float32),
            torch.tensor(target,dtype = torch.float32),
            torch.tensor(features,dtype = torch.float32)
        )

train_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in train_df["Id"].values]

train_loader = torch.utils.data.DataLoader(PawPularityDataset(train_img_paths,train_df,train_aug), batch_size = 2, shuffle=True)