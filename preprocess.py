import os
import cv2
import torch
from PIL import Image, ImageOps
from skimage.transform import resize
from torchvision import transforms as T 
import numpy as np
from utils.scan_images import scan_image
from utils.downsampler import downsample_body_mask

def preprocess(root_dir,human_dir,cloth_dir,new=False,train=True):
    
    human_list = os.listdir(human_dir)
    human_list.sort()
    cloth_list = os.listdir(cloth_dir)
    input_dir = human_dir
    output_dir = os.path.join(root_dir,"parts/")

    if new and not train:
        from utils.HumanPartsSegmentation.simple_extractor import extract_parts
        extract_parts(input_dir = input_dir,output_dir = output_dir)
        extract_parts(input_dir = cloth_dir,output_dir = root_dir+"/cloth_parts")

    parts_color = {
        'Background': [0, 0, 0], 'Hat': [128, 0, 0], 'Hair': [0, 128, 0], 
        'Sunglasses': [128, 128, 0], 'Upper-clothes': [0, 0, 128], 'Skirt': [128, 0, 128], 
        'Pants': [0, 128, 128], 'Dress': [128, 128, 128], 'Belt': [64, 0, 0], 
        'Left-shoe': [192, 0, 0], 'Right-shoe': [64, 128, 0], 'Face': [192, 129, 0], 
        'Left-leg': [64, 0, 128], 'Right-leg': [192, 0, 128], 'Left-arm': [64, 128, 128],
         'Right-arm': [192, 128, 128], 'Bag': [0, 64, 0], 'Scarf': [128, 64, 0]
         }

    return_dict = dict()
    itr = 0
    for image_file in human_list:
      if image_file != '.ipynb_checkpoints':        
        image_name = os.path.splitext(image_file)[0]   
        extention = ".png"

        image_path = os.path.join(human_dir,image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        heatmaps = np.zeros((17,256,256))
        heatmap_list = os.listdir(root_dir+"/heatmaps")
        if image_name+".npy" not in heatmap_list:
          from utils.generate_heatmaps import generate_heatmaps
          heatmaps_path = os.path.join(root_dir,"heatmaps",image_name+".npy")
          heatmaps = np.transpose(generate_heatmaps(image_path),(1,2,0))
          np.save(heatmaps_path,heatmaps)

        segmented_image_path = os.path.join(root_dir,"parts",image_name)+extention
        segmented_image = cv2.imread(segmented_image_path)
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)


        face_list = os.listdir(root_dir+"/faces")
        if image_name+extention not in face_list:
            print(f"making face {image_name}...")
            """Extracting face and hair mask"""
            face_mask = scan_image(segmented_image, parts_color['Face'])
            hair_mask = scan_image(segmented_image, parts_color['Hair'])

            full_head_mask = face_mask + hair_mask
            image_np=np.asarray(image)
            full_head_image = cv2.bitwise_and(image_np, image_np, mask=full_head_mask)
            full_head_path = os.path.join(root_dir,"faces",image_name)+extention

            """Saving the face and hair image"""
            full_head_image = cv2.cvtColor(full_head_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(full_head_path,full_head_image)

        body_list = os.listdir(root_dir+"/body_mask")
        if image_name+extention not in body_list:
            print(f"making body mask {image_name}...")
            background = scan_image(segmented_image, parts_color['Background'])
            body_mask = np.zeros_like(full_head_mask)

            for i in range(len(full_head_mask)):
                for j in range(len(full_head_mask[0])):
                    if full_head_mask[i][j] == 0:
                        body_mask[i][j] = 1

            for i in range(len(background)):
                for j in range(len(background[0])):
                    if background[i][j] == 1:
                        body_mask[i][j] = 0

            body_mask = downsample_body_mask(body_mask*255)
            body_mask_path = os.path.join(root_dir,"body_mask",image_name)+extention
            cv2.imwrite(body_mask_path,body_mask)
            body_mask = np.expand_dims(body_mask, axis=-1)

        clmask_list = os.listdir(root_dir+"/cloth_mask")
        if image_name+extention not in clmask_list:
            cloth_mask_path = os.path.join(root_dir,"cloth_mask",image_name)+extention
            upper = scan_image(segmented_image, parts_color['Upper-clothes'])
            dress = scan_image(segmented_image, parts_color['Dress'])
            cloth_mask = upper + dress
            cv2.imwrite(cloth_mask_path,cloth_mask*255)
        else:
            og_cloth = os.listdir(root_dir+"/original_cloth")
            for cloth_file in cloth_list:
                if cloth_file != '.ipynb_checkpoints' and cloth_file[:-4]+".png" not in og_cloth:
                    cloth_name = os.path.splitext(cloth_file)[0]   
                    extention = ".png"
                    segmented_cloth_path = os.path.join(root_dir,"cloth_parts",cloth_name)+extention
                    segmented_cloth = cv2.imread(segmented_cloth_path)
                    segmented_cloth = cv2.cvtColor(segmented_cloth, cv2.COLOR_BGR2RGB)
                    
                    upper = scan_image(segmented_cloth, parts_color['Upper-clothes'])
                    dress = scan_image(segmented_cloth, parts_color['Dress'])
                    cloth_mask = upper + dress

                    cl_path = os.path.join(cloth_dir,cloth_file)
                    print(cl_path)
                    cloth = cv2.imread(cl_path)
                    cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)

                    cloth_np=np.asarray(cloth)
                    cloth_image = cv2.bitwise_and(cloth_np, cloth_np, mask=cloth_mask)
                    
                    """Saving the face and hair image"""
                    cloth_path = os.path.join(root_dir,"original_cloth",image_name)+extention
                    cloth_image = cv2.cvtColor(cloth_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cloth_path,cloth_image)



        itr = itr + 1
        print(itr)


        





