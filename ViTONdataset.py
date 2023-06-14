import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import utils.config as config
from skimage.transform import resize


class VirtualFashionDataset(Dataset):
    def __init__(self, root_dir,eval=False, transform=None):
      self.body_mask = os.path.join(root_dir,"body_mask/")
      self.cloth = os.path.join(root_dir,"cloth/")
      self.faces = os.path.join(root_dir,"faces/")
      self.heatmaps = os.path.join(root_dir,"heatmaps/")
      self.images = os.path.join(root_dir,"image/")
      self.target_cloth = os.path.join(root_dir,"original_cloth/")        
      self.target_mask = os.path.join(root_dir,"cloth_mask/")
      self.eval = eval
      self.image_list = os.listdir(self.images)
      self.image_list = [file for file in self.image_list if file != '.ipynb_checkpoints']

      if self.eval:
        import preprocess as p
        p.preprocess(root_dir,self.images,self.cloth,new=True,train=False)

        
    def __len__(self):
      return len(self.image_list)

    def __getitem__(self, idx):
        """These are X values for model"""
        body_mask = cv2.imread(self.body_mask+self.image_list[idx])
        body_mask = cv2.cvtColor(body_mask, cv2.COLOR_BGR2GRAY)

        face = cv2.imread(self.faces+self.image_list[idx])
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        og_heatmaps = np.transpose(np.load(self.heatmaps+self.image_list[idx][:-4]+".npy"),(2,0,1))
        heatmaps = np.zeros((17, 256, 256))
        for i in range(17):
          heatmaps[i] = resize(og_heatmaps[i], (256, 256), mode='constant')


        if self.eval:
          target_cloth = cv2.imread(self.target_cloth+self.image_list[idx][:-4]+".png")
          print(target_cloth)
        else:
          target_cloth = cv2.imread(self.target_cloth+self.image_list[idx][:-6]+"_1.png")
        target_cloth = cv2.cvtColor(target_cloth, cv2.COLOR_BGR2RGB)

        """These are Y values for model"""
        target_image = cv2.imread(self.images+self.image_list[idx])
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        target_mask = cv2.imread(self.target_mask+self.image_list[idx])
        target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)

        """Converting to Tensors and applying augumentations"""
        augmentaions = config.all_transform(
            image=target_image,
            face_image=face,
            body_mask=body_mask,
            target_cloth=target_cloth,
            target_mask = target_mask
        )

        transform_images = config.transform_only_images(
            image=augmentaions["image"],
            face_image=augmentaions["face_image"],
            target_cloth=augmentaions["target_cloth"],
        )

        transform_masks = config.transform_only_mask(
            image=augmentaions["target_mask"],
            body_mask=augmentaions["body_mask"]
        )

        body_mask_tensor = transform_masks["body_mask"].to(float)
        face_tensor = transform_images["face_image"].to(float)
        heatmap_tensor = torch.tensor(heatmaps).to(float)

        target_cloth_tensor = transform_images["target_cloth"].to(float)

        target_image_tensor = transform_images["image"].to(float)
        target_mask_tensor = transform_masks["image"].to(float)

        agnostic_person_representation_tensor = torch.cat(( face_tensor, body_mask_tensor, heatmap_tensor),dim = 0)

        return self.image_list[idx], agnostic_person_representation_tensor, target_cloth_tensor, target_image_tensor, target_mask_tensor

            