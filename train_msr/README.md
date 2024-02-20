
# Mixed Segment Recognition Network implemented in Pytorch

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/AudioClassification-Pytorch)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/AudioClassification-Pytorch)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/AudioClassification-Pytorch)


# Introduction

This project is a classification network, with training data comprising mixed speech segments and clean speech segments. Mixed speech segments are created by cropping, concatenating, and smoothing segments from different individuals, while clean speech segments are directly cropped from the original speech data. To enhance the coherence between the two segments and simulate the alternation of speakers in real conversations, a fade-in and fade-out effect is applied at the crosspoints of the two speech segments during mixed speech processing. This effect ensures a smooth transition, making the overall processing more natural and enhancing experimental reliability. Subsequently, labels are generated for each speech segment based on the training data, typically indicating whether the segment is a mixed or clean segment.
The network training utilizes a ResNet model. After training the MSR network, its judgment on each segment will determine whether the re-segmentation is necessary.



## Preparing Data


`data_preparation.py` can be used to generate mixed and clean data. `create_data.py` can be used to generate a list of data sets. There are many ways to generate a list of data sets.
```shell
python data_preparation.py
python create_data.py
```

## Train Model




# Reference

1. https://github.com/yeyupiaoling/PPASR
2. https://github.com/alibaba-damo-academy/3D-Speaker
3. https://github.com/yeyupiaoling/AudioClassification-Pytorch
