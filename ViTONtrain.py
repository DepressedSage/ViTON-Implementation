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

torch.backends.cudnn.benchmark = True

def trainViTON(
    disc_image, gen_image, loader, gen_image_opt,
    disc_image_opt, percep_loss, bce, g_image_scaler,
    d_image_scaler
):
    loop = tqdm(loader, leave=True)


    for idx, data in enumerate(loader):
        image_name, agnostic, target_cloth, real_image, target_mask = data
        agnostic = agnostic.to(config.DEVICE)           # p in the paper
        target_cloth = target_cloth.to(config.DEVICE)   # c in paper
        real_image = real_image.to(config.DEVICE)       # I in paper
        target_mask = target_mask.to(config.DEVICE)     # M0 in paper

        cloth_input_data = torch.cat((agnostic,target_cloth),dim=1)
        target_output = torch.cat((real_image,target_mask),dim=1)

        #Train Image discriminator
        with torch.cuda.amp.autocast():
            fake_image = gen_image(cloth_input_data)
            d_real_image = disc_image(cloth_input_data, target_output)
            d_real_loss = bce(d_real_image, torch.ones_like(d_real_image))
            d_fake_image = disc_image(cloth_input_data,fake_image.detach())
            d_fake_loss = bce(d_fake_image, torch.zeros_like(d_fake_image))
            d_loss = (d_real_loss + d_fake_loss)/2

        disc_image_opt.zero_grad()
        d_image_scaler.scale(d_loss).backward()
        d_image_scaler.step(disc_image_opt)
        d_image_scaler.update()  

        # Train generator
        with torch.cuda.amp.autocast():
            d_fake = disc_image(cloth_input_data, fake_image.detach())
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            g_percep_loss = percep_loss(fake_image[:,:3,:,:], real_image, fake_image[:,3:,:,:], target_mask)
            g_loss = g_fake_loss + g_percep_loss

        gen_image_opt.zero_grad()
        g_image_scaler.scale(g_loss).backward()
        g_image_scaler.step(gen_image_opt)
        g_image_scaler.update()  


        # memory_used = torch.cuda.memory_allocated()  # Get current GPU memory usage
        # print(f"\nIteration {idx + 1} - Memory used: {memory_used / (1024 ** 2)} MB")  # Convert to MB for readability

        if idx % 10 == 0:
            print(f"This is iteration: {idx}")
            loop.set_postfix(
                d_real_image=torch.sigmoid(d_real_image).mean().item(),
                d_fake_image=torch.sigmoid(d_fake_image).mean().item(),
            )
    return

def collate_custom(batch):
    batch = [data for data in batch if data is not None]  # Skip None values in the batch
    return default_collate(batch)


def main():
    disc_image = Discriminator(in_channels = 28).to(config.DEVICE)
    gen_image = Generator(in_channels = 24, out_channels = 4).to(config.DEVICE)
    perceptual_loss = Perceptual_Loss([1.0,1.0,1.0,1.0,1.0,1.0])
    BCE = nn.BCEWithLogitsLoss()
    gen_image_opt = optim.Adam(
        gen_image.parameters(),
        lr=config.LEARNING_RATE, 
        betas=(0.5, 0.999)
    )

    disc_image_opt = optim.Adam(
        disc_image.parameters(),
        lr=config.LEARNING_RATE, 
        betas=(0.5, 0.999)
    )

    g_image_scaler = torch.cuda.amp.GradScaler()
    d_image_scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_, disc_image, disc_image_opt, config.LEARNING_RATE,
        )

        load_checkpoint(
            config.CHECKPOINT_G, gen_image, gen_image_opt, config.LEARNING_RATE,
        )

    train_dataset = VirtualFashionDataset(config.TRAIN_DIR)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_custom
    )

    val_dataset = VirtualFashionDataset(config.VAL_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_custom
    )

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch} starting...")
        trainViTON(
            disc_image, gen_image, train_dataloader, gen_image_opt,
            disc_image_opt, perceptual_loss, BCE, g_image_scaler,
            d_image_scaler
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen_image, gen_image_opt, filename=config.CHECKPOINT_G)
            save_checkpoint(disc_image, disc_image_opt, filename=config.CHECKPOINT_D)

        save_some_examples(gen_image, val_loader, epoch, folder=config.VAL_DIR+"/output")

    print(f"Completed epoch: {epoch}!")
if __name__ == "__main__":
    main()

