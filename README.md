---
typora-root-url: images
---

# Copy-Paste-for-Semantic-Segmentation
Unofficial implementation of Copy-Paste method: Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation.

### methods used in this repo:

1. Random Horizontal Flip
2. Large Scale Jittering
3. Copy-Paste



### Steps:

1. choice source image and main image;
2. get annotations from source image;
3. rescale source image and it annotations;
4. paste source image and annotations to main image and annotations;
5. merge main annotations and source annotations;



### Usage:

```
usage: copy_paste.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]
                     [--lsj LSJ]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        input annotated directory
  --output_dir OUTPUT_DIR
                        output dataset directory
  --lsj LSJ             if use Large Scale Jittering

```

for example:

`python copy_paste.py --input_dir ../dataset/VOCdevkit2012/VOC2012 --output_dir ../dataset/VOCdevkit2012/VOC2012_copy_paste --lsj True`



### Results：

main image + source image:

![image](/image.jpg)

main anno + source anno:

![image](/anno.png)

visualization:

![vis](/vis.jpg)

