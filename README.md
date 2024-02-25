# Variable Segment Length and Domain-Adapted Feature Optimization for Speaker Diarization


This project is associated with the recently released AIHSHELL-4 dataset for speaker diarization in conference scenario. The project is divided into five parts, named ***training the MSR network***, ***fine-tuning speaker embedding extractor***, and ***speaker diarization***. The task evaluates the ability of speaker diarization. The goal of this project is to optimize speaker embeddings for speaker diarization by introducing the Mixed Segment Recognition network and employing a data augmentation strategy and a new loss function to fine-tune the pre-trained speaker embedding extractor.

## Setup

```shell
git clone https://github.com/xiaoaaa2/Ada-sd.git
pip install -r requirements.txt
```
## Introduction

* [Train MSR](train_msr): Prepare the training and evaluation data.
* [Finetune Extractor](finetune_extractor): Train and evaluate the front end model. 
* [Speaker Diarization](sd): Generate the speaker diarization results. 

## General steps
1. Generate training data for MSR model and train the model.
2. Finetune the speaker embedding extractor by using data augmentation strategy and introducing a new loss function.
3. Intruduce the MSR network to the Speaker Diarization pipeline and replace the original speaker embedding extractor with the fine-tuned model.
4. Evaluate the Speaker Diarization project.

## Citation
If you use this challenge dataset and baseline system in a publication, please cite the following paper:
    @article{fu2021aishell,
             title={AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario},
             author={Fu, Yihui and Cheng, Luyao and Lv, Shubo and Jv, Yukai and Kong, Yuxiang and Chen, Zhuo and Hu, Yanxin and Xie, Lei and Wu, Jian and Bu, Hui and Xin, Xu and Jun, Du and Jingdong Chen},
             year={2021},
             conference={Interspeech2021, Brno, Czech Republic, Aug 30 - Sept 3, 2021}
             }
The paper is available at https://arxiv.org/abs/2104.03603
Dataset is available at http://www.openslr.org/111/ and http://www.aishelltech.com/aishell_4
    
## Code license 
[Apache 2.0](./LICENSE)
