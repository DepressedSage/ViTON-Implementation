import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content/drive/MyDrive/ViTON/DataViTON/train"
VAL_DIR = "/content/drive/MyDrive/ViTON/DataViTON/val"
EVAL_DIR = "/content/drive/MyDrive/ViTON/DataViTON/eval"
LEARNING_RATE = 2e-4
BATCH_SIZE = 20
NUM_WORKERS = 1
IMAGE_SIZE = 256
CHANNELS_IMG = 3
NUM_EPOCHS = 30
LOAD_MODEL = False
SAVE_MODEL = False
LAMBDA = 100
LAMBDA_GP = 10
CHECKPOINT_D = "/content/drive/MyDrive/ViTON/model/disc/dimg.pth.tar"
CHECKPOINT_G = "/content/drive/MyDrive/ViTON/model/gen/gimg.pth.tar"

all_transform = A.Compose(
    [A.Resize(width=256, height=256),], 
    additional_targets={
      "face_image": "image",
      "body_mask": "image",
      "target_cloth" : "image",
      "target_mask" : "image"
      }
)

transform_only_images = A.Compose(
    [
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],
    additional_targets={
      "face_image": "image",
      "target_cloth" : "image",
      }   
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ],
    additional_targets={
      "body_mask": "image",
      }
)