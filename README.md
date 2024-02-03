# AISHELL-4

This project is associated with the recently-released AIHSHELL-4 dataset for speaker diarization in conference scenario. The project is divided into five parts, named ***training MSR network***, ***finetuning speaker embedding extractor***, and ***sd***. The task evaluates the ability of speaker diarization. The goal of this project is to The goal of this paper is to optimize speaker embeddings for speaker diarization by introducing the MSR network and employing a data augmentation strategy and a new loss function to fine-tune the pre-trained speaker embedding extractor.

## Setup

```shell
git clone https://github.com/xiaoaaa2/Ada-SD.git
pip install -r requirements.txt
```
## Introduction

* [Train MSR](model_preparation): Prepare the training and evaluation data.
* [Finetune extractor](model_preparation): Train and evaluate the front end model. 
* [Speaker Diarization](sd): Generate the speaker diarization results. 
* [Evaluation](eval): Evaluate the results of models above and generate the CERs for Speaker Independent and Speaker Dependent tasks respectively.

## General steps
1. Generate training data for fe and asr model and evaluation data for Speaker Independent task.
2. Do speaker diarization to generate rttm which includes vad and speaker diarization information.
3. Generate evaluation data for Speaker Dependent task with the results from step 2.
4. Train FE and ASR model respectively.
5. Generate the FE results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively.
6. Generate the ASR results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively with the results from step 2 and 3 for No FE results.
7. Generate the ASR results of evaluation data for Speaker Independent and Speaker Dependent tasks respectively with the results from step 5 for FE results.
8. Generate CER results for Speaker Independent and Speaker Dependent tasks of (No) FE with the results from step 6 and 7 respectively.




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
