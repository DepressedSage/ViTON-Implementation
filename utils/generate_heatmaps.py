import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal
from torchvision.ops import distance_box_iou

def generate_keypoints(image_path):
# Load the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
# Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.unsqueeze(0)
# Perform inference
    model.eval()
    with torch.no_grad():
        output = model(img)[0]
    return output

def keypoints_per_person(image_path, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    img = cv2.imread(image_path)
    person_keypoints = dict()
    for person_id in range(len(all_keypoints)):
      if confs[person_id]>conf_threshold:
        keypoints = all_keypoints[person_id, ...]
        scoresTensor = all_scores[person_id, ...]
        scores = scoresTensor.numpy()
        key_points = dict()
        for kp in range(len(scores)):
          keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
          key_points[kp] = (scores[kp],keypoint)
        person_keypoints[person_id] = key_points
    return person_keypoints


def calc_dis(poi,img_shape):
    distance = np.zeros(img_shape)
    for i in range(len(distance)):
        for j in range(len(distance[0])):
            if j != poi[0] and i !=poi[1]:
                x = (poi[0] - j)**2
                y = (poi[1] - i)**2
                distance[i,j] = 1/((x + y)**0.5)

    min_val = np.min(distance)
    max_val = np.max(distance)
    distance = (distance - min_val) / (max_val - min_val)
    distance[poi[0],poi[1]] = 1
    for i in range(len(distance)):
        for j in range(len(distance[0])):
            if distance[i,j] < 0.8:
                distance[i,j]**8

    return distance

def generate_heatmaps(image_path, sigma=3.0):
    image = cv2.imread(image_path)
    target = generate_keypoints(image_path)
    target_keypoints = keypoints_per_person(image_path=image_path,
                                            all_keypoints=target["keypoints"],
                                            all_scores=target["keypoints_scores"],
                                            confs=target["scores"],
                                            keypoint_threshold=2)
    height, width ,_= image.shape
    coords = target_keypoints[0]
    n_keypoints = len(coords)
    heatmaps = np.zeros((n_keypoints, height, width), dtype=np.float32)

    for i in range(n_keypoints):
      if coords[i][0] > 2:
        x, y = coords[i][1]
        if x >= 0 and x < width and y >= 0 and y < height:
            distance = calc_dis((y,x),(height,width))

            heatmaps[i] = (distance*255).astype(np.uint8)
        #   kernel_size = 11
        #   gaussian_kernel = make_gaussian_kernel(sigma, size=kernel_size)
        #   left = min(int(np.round(x)) - (kernel_size // 2), width - 1 - kernel_size)
        #   right = left + kernel_size
        #   top = min(int(np.round(y)) - (kernel_size // 2), height - 1 - kernel_size)
        #   bottom = top + kernel_size
        #   heatmaps[i, top:bottom, left:right] = gaussian_kernel
    # for i in heatmaps:
    #     for j in i:
    #         for k in j
    #             if k > 0:
    #                 k = 1
    return heatmaps

