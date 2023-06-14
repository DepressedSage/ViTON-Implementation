# ViTON-Implementation

This is a PyTorch Implementation of the ViTON [paper link](https://arxiv.org/pdf/1711.08447.pdf)

Currently this has the generator part of the paper. I am working on the Refinement network.

### The Generator Network 

It uses a Unet architecture for the image + mask generation.
The input to the Generator Network is a 24 channel Tensor consisting of 
- Pose heatmaps of the human user that has the target pose. (17 Channels)
- A downsampled Body mask excluding the face and hair which will be used for the overall shape of the output image. (1 Channel) 
- Face and Hair extracted from the human image. This will be used for identiy in the target image. (3 Channels) 
- The target clothing that needs to be applied on the human picture. (3 Channels)

The output will be a 4 Channel Tensor
- The target ouput Image with the Human wearing the target clothing item. (3 Channels)
- The generator mask of the target clothing item from the output image.

The loss function is a combination of Perceptual Loss for checking the loss between the Target Output Image and the Generator Image by the generator network & An L1 loss for the target mask and the generated mask of the target clothing item on human.

For Perceptual Loss this implementation uses 5 layers of pretrained VGG16. 
( Conv1_2, Conv2_2, Conv3_2, Conv4_2, Conv5_2 )

These layers help by creating features maps using the trained parameters of the VGG16 model,

The generated and target image are passed through the VGG16 model and the feature maps from the specified layers are extracted. Then the loss between the two feature maps are calculated and added up to create the total loss for the generator.
This loss is added to the L1 loss calculated between the Masks to create the Total Loss for the Generator Network.

### The Discriminator Network:

It takes the input as X and Y and concatenates them
- X being the 24 channel input for the Generator Network
- Y being the target image to calculate `d_real_image` and Y being generated image to calculate `d_fake_image`

This is then passed into the discriminator network.

The loss function used for the Discriminator network is BCE. The loss is calculated for both `d_rea_image` and `d_fake_image` and then added to calcualte the total loss for the discriminator.

### Data Preparation:

The Data should be stored in the following format in the DataViTON directory.
```
- train
  ├─ parts
  ├─ outputs
  ├─ original_cloth_mask
  ├─ original_cloth
  ├─ body_mask
  ├─ cloth_mask
  ├─ faces
  ├─ heatmaps
  └─ image
 ```
  
```
- val
  ├─ parts
  ├─ outputs
  ├─ original_cloth_mask
  ├─ original_cloth
  ├─ body_mask
  ├─ cloth_mask
  ├─ faces
  ├─ heatmaps
  └─ image
  ```
 
The models should be stored in the following manner.
```
- model
  ├─ disc
  │   └─ dimg.pth.tar
  └─ gen
      └─ gimg.pth.tar
```

The link to the pretrained generator network [generator](https://drive.google.com/file/d/16UVHNGDgiurtHEexp68wCb_dB5N9BFcU/view?usp=sharing)

The link to the pretrained discriminator netowrk [discriminator](https://drive.google.com/file/d/167dBKAbzeVJ5rM1ABOzJy1tHhIHcJ6oB/view?usp=sharing)
