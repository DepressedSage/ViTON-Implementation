import os
import sys
sys.path.append(os.path.abspath("./"))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from utils.model_utils import save_checkpoint, load_checkpoint, save_some_examples
from torchvision.utils import save_image
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from ViTONdataset import VirtualFashionDataset
from PerceptualLoss import Perceptual_Loss
import utils.config as config

def evalViTON(model, loader, folder):
  for data in loader:
    image_name, agnostic, target_cloth, real_image, target_mask = data

    cloth_input_data = torch.cat((agnostic,target_cloth),dim=1)
    target_output = torch.cat((real_image,target_mask),dim=1)

    x, y = cloth_input_data.to(config.DEVICE), target_output.to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        y_fake = model(x.to(torch.float32))

        y_fake_image = y_fake[:,:3,:,:] * 0.5 + 0.5  # remove normalization#
        y_fake_mask = y_fake[:,3:,:,:] * 0.5 + 0.5  # remove normalization#

        name = image_name[0]
        save_image(y_fake_image, folder + f"/y_gen_{name}")
        save_image(y_fake_mask, folder + f"/y_gen_mask_{name}")
        print(f"Done with image {name}")

def collate_custom(batch):
    batch = [data for data in batch if data is not None]  # Skip None values in the batch
    return default_collate(batch)


def main():
    gen_image = Generator(in_channels = 24, out_channels = 4).to(config.DEVICE)

    gen_image_opt = optim.Adam(
        gen_image.parameters(),
        lr=config.LEARNING_RATE, 
        betas=(0.5, 0.999)
    )

    g_image_scaler = torch.cuda.amp.GradScaler()

    load_checkpoint(
        config.CHECKPOINT_G, gen_image, gen_image_opt, config.LEARNING_RATE
        )

    val_dataset = VirtualFashionDataset(config.VAL_DIR,eval=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_custom
    )

    evalViTON(gen_image, val_loader, folder=config.VAL_DIR+"/output")

if __name__ == "__main__":
    main()
