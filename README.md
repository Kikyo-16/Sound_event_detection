# Sound event detection
## Introduction
This code aims at semi-supervised and weakly-supervised sound event detection. The dataset utilized in our experiments is from DCASE (IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events), more specifically, from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). The code embraces two methods we proposed to solve this task: [specialized decision surface (SDS) and disentangled feature (DF)](https://arxiv.org/abs/1905.10091) for weakly-supervised learning and [guided learning (GL)](https://arxiv.org/abs/1906.02517) for semi-supervised learning. We're so glad if you're interested in using it for research purpose or DCASE participation. Please don't hesitate to contact us should you have any question.
## Something about DCASE
The dataset utilized in our experiments is from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). We've encapsulated some interfaces to handle dataset of DCASE task4 and streams to read the data, so this code can be used directly for the future challenge (or as a basis for fine-tuning).
Actually, we exploited it to participate in DCASE2019 task4 and won the first price ([our technique report](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Lin_25.pdf) and [challenge results](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments-results)). After adding analysis of result and making some modifications, we submitted the new technical report (to be updated soon) to [DCASE2019 Workshop](http://dcase.community/workshop2019/). 
## Main ideas comprised in the code
### Specialized decision surface (SDS) and disentangled feature (DF)
We propose Specialized decision surface (SDS) and disentangled feature (DF) in paper [Specialized Decision Surface and DisentangledFeature for Weakly-Supervised Polyphonic Sound Event Detection](https://arxiv.org/abs/1905.10091).
There are mainly 2 contribution in our work:
1 The Multiple instance learning (MIL) framework with pooling module and the neural network is commonly utilized for the weakly-supervised learning task, base on which we compare the performances of different pooling modules including GMP (global max pooling), GAP (global average pooling), GSP (global softmax pooling) and cATP (class-wise attention pooling), and give an explanation about why cATP perform best from the perspective of the high-level feature space of the neural network encoder. This explanation enables us to focus on a potential decision surface which we term the specialized decision surface (SDS). SDS shows its power in frame-level prediction (event detection). Actually, similar to SDS, something like "attention output" or "attention mask" might have been exploited in other works. However, the detailed explanation remains explored. Therefore, we explore a detailed explanation for cATP and provide the detailed analysis of experimental results. We argue that the explanation of SDS is expected to promote the optimizing of the pooling module in weakly-supervised learning.
2 Disentangled feature (DF) is proposed to ease the problem caused by the co-occurrence of categories. We describe it in detail in the paper.

### Guided learning (GL)

## How to use
### Quick start
### Configure files
## Details of the code implement
### Feature extraction
### Source codes
## Contact us
Please don't hesitate to contact us should you have any question. You can email me at `linliwei17g@ict.ac.cn` or `1174436431@qq.com`.
