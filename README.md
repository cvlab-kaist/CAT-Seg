# CAT-Seg:cat:: Cost Aggregation for Open-Vocabulary Semantic Segmentation

This is our official implementation of CAT-Seg!

[[arXiv](https://arxiv.org/abs/2303.11797)] [[Project](https://ku-cvlab.github.io/CAT-Seg/)] [[HuggingFace Demo](https://huggingface.co/spaces/hamacojr/CAT-Seg)]<br>
by [Seokju Cho](https://seokju-cho.github.io/)\*, [Heeseong Shin](https://github.com/hsshin98)\*, [Sunghwan Hong](https://sunghwanhong.github.io), Seungjun An, Seungjun Lee, [Anurag Arnab](https://anuragarnab.github.io), [Paul Hongsuck Seo](https://phseo.github.io), [Seungryong Kim](https://cvlab.korea.ac.kr)

## Introduction
![](assets/fig1.png)
We introduce cost aggregation to open-vocabulary semantic segmentation, which jointly aggregates both image and text modalities within the matching cost.

For further details and visualization results, please check out our [paper](#) and our [project page](https://ku-cvlab.github.io/CAT-Seg/).

## :fire:TODO
- [x] Train/Evaluation Code (Mar 21, 2023)
- [ ] Pre-trained weights
- [ ] Code of interactive demo

## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](datasets/README.md).

## Training
We provide shell scripts for training and evaluation. ```run.py``` trains the model in default configuration and evaluates the model after training. 

To train or evaluate the model in different environments, modify the given shell script and config files accordingly.

### Training script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For ViT-B variant
sh run.sh configs/vitb_r101_384.yaml 4 output/
# For ViT-L variant
sh run.sh configs/vitl_swinb_384.yaml 4 output/
# For ViT-H variant
sh run.sh configs/vitl_swinb_384.yaml 4 output/ MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED "ViT-H" MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM 1024
# For ViT-G variant
sh run.sh configs/vitl_swinb_384.yaml 4 output/ MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED "ViT-G" MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM 1280
```

## Evaluation
```eval.sh``` automatically evaluates the model following our evaluation protocol, with weights in the output directory if not specified.
To individually run the model in different datasets, please refer to the commands in ```eval.sh```.

### Evaluation script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

sh eval.sh configs/vitl_swinb_384.yaml 4 output/ MODEL.WEIGHTS path/to/weights.pth
```

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Zegformer](https://github.com/dingjiansw101/ZegFormer), whose code has been utilized in this repository.

## Citing CAT-Seg :cat::pray:

```BibTeX
@misc{cho2023catseg,
      title={CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation}, 
      author={Seokju Cho and Heeseong Shin and Sunghwan Hong and Seungjun An and Seungjun Lee and Anurag Arnab and Paul Hongsuck Seo and Seungryong Kim},
      year={2023},
      eprint={2303.11797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
