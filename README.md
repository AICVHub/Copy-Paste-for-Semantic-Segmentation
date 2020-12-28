---
typora-root-url: images
---

# Copy-Paste-for-Semantic-Segmentation
Unofficial implementation of Copy-Paste method: Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation.



### TODO:

- [ ] Use Large Scale Jittering when Copy-Paste;



### Steps:

1. choice source image and main image;
2. get annotations from source image;
3. rescale source image and it annotations;
4. paste source image and annotations to main image and annotations;
5. merge main annotations and source annotations;



### Usage:

`python copy_paste.py --input_dir dataset/VOCdevkit2012/VOC2012 --output_dir dataset/VOCdevkit2012/VOC2012_copy_paste`



### Results：

main image + source image:

![image](/image.jpg)

main anno + source anno:

![image](/anno.png)

visualization:

![vis](/vis.jpg)

