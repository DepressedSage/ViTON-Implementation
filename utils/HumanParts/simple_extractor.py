import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.HumanPartsSegmentation.utils.transforms import transform_logits

import utils.HumanPartsSegmentation.networks as networks
from utils.HumanPartsSegmentation.datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
}

def put_palette(label_mat,parts_colors):
    shape = label_mat.shape
    rgb_image = np.zeros((shape[0],shape[1],3))
    for label_idx, label in enumerate(parts_colors):
        for i in range(len(label_mat)):
            for j in range(len(label_mat[0])):
                if rgb_image[i,j,0] + rgb_image[i,j,1] + rgb_image[i,j,2] == 0:
                    if label_mat[i][j] == label_idx:
                        rgb_image[i,j,0] = parts_colors[label][0]    
                        rgb_image[i,j,1] = parts_colors[label][1]
                        rgb_image[i,j,2] = parts_colors[label][2]                
    return rgb_image


def extract_parts(input_dir='/content/drive/MyDrive/ViTON/Data/train/inputs'
                ,output_dir='/content/drive/MyDrive/ViTON/Data/train/parts',
                model_path='/content/drive/MyDrive/ViTON/utils/HumanPartsSegmentation/checkpoints/final.pth',gpu='0'):
    gpus = [int(i) for i in gpu.split(',')]
    assert len(gpus) == 1
    if not gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    num_classes = dataset_settings['atr']['num_classes']
    input_size = dataset_settings['atr']['input_size']
    label = dataset_settings['atr']['label']

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(model_path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 129, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0]
    RGB_palette = list()

    for i in range(0,len(palette),3):
        RGB_palette.append(palette[i:i+3])

    parts_color = dict()
    for i, part in enumerate(dataset_settings['atr']['label']):
        parts_color[part] = RGB_palette[i]
    
    itr = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            out_list = os.listdir(output_dir)
            if img_name[:-4] + '.png' not in out_list:
              c = meta['center'].numpy()[0]
              s = meta['scale'].numpy()[0]
              w = meta['width'].numpy()[0]
              h = meta['height'].numpy()[0]

              output = model(image.cuda())
              upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
              upsample_output = upsample(output[0][-1][0].unsqueeze(0))
              upsample_output = upsample_output.squeeze()
              upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

              logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
              parsing_result = np.argmax(logits_result, axis=2)
              output_img = put_palette(np.asarray(parsing_result, dtype=np.uint8),parts_color)
              output_img = Image.fromarray(np.uint8(output_img))
              parsing_result_path = os.path.join(output_dir, img_name[:-4] + '.png')
              output_img.save(parsing_result_path)
              if itr > 1000:
                  return parts_color
              itr = itr+1