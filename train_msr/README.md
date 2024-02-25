
# Mixed Segment Recognition Network implemented in Pytorch

# Introduction

This project is a classification network, with training data comprising mixed speech segments and pure speech segments. Mixed speech segments are created by cropping, concatenating, and smoothing segments from different individuals, while pure speech segments are directly cropped from the original speech data. To enhance the coherence between the two segments and simulate the alternation of speakers in real conversations, a fade-in and fade-out effect is applied at the crosspoints of the two speech segments during mixed speech processing. Subsequently, labels are generated for each speech segment based on the training data, typically indicating whether the segment is a mixed or clean segment.
The network training utilizes a ResNet model. After training the MSR network, its judgment on each segment will determine whether the re-segmentation is necessary.



## Preparing Data


`data_preparation.py` can be used to generate mixed and clean data. `create_data.py` can be used to generate a list of data sets. There are many ways to generate a list of data sets.
```shell
python data_preparation.py
python create_data.py
```

## Training Model

```shell
python train.py
```

## Evaluating Model

```shell
python eval.py
```

# Reference

1. https://github.com/yeyupiaoling/PPASR
2. https://github.com/alibaba-damo-academy/3D-Speaker
3. https://github.com/yeyupiaoling/AudioClassification-Pytorch
