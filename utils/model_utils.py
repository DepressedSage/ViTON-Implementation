import torch
import utils.config as config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    image_name, agnostic, target_cloth, real_image, target_mask = next(iter(val_loader))

    cloth_input_data = torch.cat((agnostic,target_cloth),dim=1)
    target_output = torch.cat((real_image,target_mask),dim=1)

    x, y = cloth_input_data.to(config.DEVICE), target_output.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)

        y_fake_image = y_fake[:,:3,:,:] * 0.5 + 0.5  # remove normalization#
        y_fake_mask = y_fake[:,3:,:,:] * 0.5 + 0.5  # remove normalization#

        save_image(y_fake_image, folder + f"/y_gen_{epoch}.png")
        save_image(y_fake_mask, folder + f"/y_gen_mask_{epoch}.png")

        if epoch == 1:
            save_image(y[:,:3,:,:] * 0.5 + 0.5, folder + f"/label_{epoch}.png")
            save_image(y[:,3:,:,:] * 0.5 + 0.5, folder + f"/label_mask_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

