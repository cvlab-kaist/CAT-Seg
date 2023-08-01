[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cat-seg-cost-aggregation-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-3)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-3?p=cat-seg-cost-aggregation-for-open-vocabulary)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cat-seg-cost-aggregation-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-7)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-7?p=cat-seg-cost-aggregation-for-open-vocabulary)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cat-seg-cost-aggregation-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-2)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-2?p=cat-seg-cost-aggregation-for-open-vocabulary)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cat-seg-cost-aggregation-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-1)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-1?p=cat-seg-cost-aggregation-for-open-vocabulary)<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cat-seg-cost-aggregation-for-open-vocabulary/open-vocabulary-semantic-segmentation-on-5)](https://paperswithcode.com/sota/open-vocabulary-semantic-segmentation-on-5?p=cat-seg-cost-aggregation-for-open-vocabulary)

# CAT-Seg:cat:: Cost Aggregation for Open-Vocabulary Semantic Segmentation

This is our official implementation of CAT-Seg!

[[arXiv](https://arxiv.org/abs/2303.11797)] [[Project](https://ku-cvlab.github.io/CAT-Seg/)] [[HuggingFace Demo](https://huggingface.co/spaces/hamacojr/CAT-Seg)] [[Segment Anything with CAT-Seg](https://huggingface.co/spaces/hamacojr/SAM-CAT-Seg)]<br>

by [Seokju Cho](https://seokju-cho.github.io/)\*, [Heeseong Shin](https://github.com/hsshin98)\*, [Sunghwan Hong](https://sunghwanhong.github.io), Seungjun An, Seungjun Lee, [Anurag Arnab](https://anuragarnab.github.io), [Paul Hongsuck Seo](https://phseo.github.io), [Seungryong Kim](https://cvlab.korea.ac.kr)

## Introduction
![](assets/fig1.png)
We introduce cost aggregation to open-vocabulary semantic segmentation, which jointly aggregates both image and text modalities within the matching cost.

For further details and visualization results, please check out our [paper](https://arxiv.org/abs/2303.11797) and our [project page](https://ku-cvlab.github.io/CAT-Seg/).

**❗️Update:** We released a **[demo](https://huggingface.co/spaces/hamacojr/SAM-CAT-Seg)** for combining CAT-Seg and [Segment Anything](https://github.com/facebookresearch/segment-anything) for open-vocabulary semantic segmentation! We also released the code and installation guide in the demo branch for trying out the demo on your local devices! 

## :fire:TODO
- [x] Train/Evaluation Code (Mar 21, 2023)
- [x] Pre-trained weights (Mar 30, 2023)
- [x] Code of interactive demo (Jul 13, 2023)

## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](datasets/README.md).

## Demo
If you want to try your own images locally, please try [interactive demo](https://github.com/KU-CVLAB/CAT-Seg/tree/demo).

## Training
We provide shell scripts for training and evaluation. ```run.sh``` trains the model in default configuration and evaluates the model after training. 

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

## Pretrained Models
We provide pretrained weights for our models reported in the paper. All of the models were evaluated with 4 NVIDIA RTX 3090 GPUs, and can be reproduced with the evaluation script above.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">CLIP</th>
<th valign="bottom">A-847</th>
<th valign="bottom">PC-459</th>
<th valign="bottom">A-150</th>
<th valign="bottom">PC-59</th>
<th valign="bottom">PAS-20</th>
<th valign="bottom">PAS-20b</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: CAT-Seg (B) -->
<tr>
<td align="left">CAT-Seg (B)</a></td>
<td align="center">R101</td>
<td align="center">ViT-B/16</td>
<td align="center">8.9</td>
<td align="center">16.6</td>
<td align="center">27.2</td>
<td align="center">57.5</td>
<td align="center">93.7</td>
<td align="center">78.3</td>
<td align="center"><a href="https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_base.pth">ckpt</a>&nbsp;
</tr>
<!-- ROW: CAT-Seg (L) -->
<tr>
<td align="left">CAT-Seg (L)</a></td>
<td align="center">Swin-B</td>
<td align="center">ViT-L/14</td>
<td align="center">11.4</td>
<td align="center">20.4</td>
<td align="center">31.5</td>
<td align="center">62.0</td>
<td align="center">96.6</td>
<td align="center">81.8</td>
<td align="center"><a href="https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_large.pth">ckpt</a>&nbsp;
</tr>
<!-- ROW: CAT-Seg (H) -->
<tr>
<td align="left">CAT-Seg (H)</a></td>
<td align="center">Swin-B</td>
<td align="center">ViT-H/14</td>
<td align="center">13.1</td>
<td align="center">20.1</td>
<td align="center">34.4</td>
<td align="center">61.2</td>
<td align="center">96.7</td>
<td align="center">80.2</td>
<td align="center"><a href="https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_huge.pth">ckpt</a>&nbsp;
</tr>
<!-- ROW: CAT-Seg (G) -->
 <tr><td align="left">CAT-Seg (G)</a></td>
<td align="center">Swin-B</td>
<td align="center">ViT-G/14</td>
<td align="center">14.1</td>
<td align="center">21.4</td>
<td align="center">36.2</td>
<td align="center">61.5</td>
<td align="center">97.1</td>
<td align="center">81.4</td>
<td align="center"><a href="https://huggingface.co/hamacojr/CAT-Seg/blob/main/model_final_giant.pth">ckpt</a>&nbsp;
</tr>
</tbody></table>


## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Zegformer](https://github.com/dingjiansw101/ZegFormer), whose code has been utilized in this repository.
We also thank [Benedikt](mailto:benedikt.blumenstiel@student.kit.edu) for finding an error in our inference code and evaluating CAT-Seg over various datasets!
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
