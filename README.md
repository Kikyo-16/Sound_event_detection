# (Note: 3 papers related to this code will be updated a few days later)
# Sound event detection
## Introduction
  This code aims at semi-supervised and weakly-supervised sound event detection. The dataset utilized in our experiments is from DCASE (IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events), more specifically, from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). The code embraces two methods we proposed to solve this task: [specialized decision surface (SDS) and disentangled feature (DF)](https://arxiv.org/abs/1905.10091v3) for weakly-supervised learning and [guided learning (GL)](https://arxiv.org/abs/1906.02517) for semi-supervised learning.  
  
  We're so glad if you're interested in using it for research purpose or DCASE participation. Please don't hesitate to contact us should you have any question.  
  
## Something about DCASE
  The dataset utilized in our experiments is from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). We've encapsulated some interfaces to handle dataset of DCASE task4 and streams to read the data, so this code can be used directly for the future challenge (or as a basis for fine-tuning).  
  
  Actually, we exploited it to participate in DCASE2019 task4 and won the first price ([our technique report](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Lin_25.pdf) and [challenge results](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments-results)). After adding analysis of result and making some modifications, we submitted the new technical report (to be updated soon) to [DCASE2019 Workshop](http://dcase.community/workshop2019/).  

## Main ideas comprised in the code
### Specialized decision surface (SDS) and disentangled feature (DF)
We propose specialized decision surface (SDS) and disentangled feature (DF) in paper [Specialized Decision Surface and DisentangledFeature for Weakly-Supervised Polyphonic Sound Event Detection](https://arxiv.org/abs/1905.10091v3).  

There are mainly 2 contribution in our work:  
  - The Multiple instance learning (MIL) framework with pooling module and the neural network is commonly utilized for the weakly-supervised learning task, base on which we compare the performances of different pooling modules including GMP (global max pooling), GAP (global average pooling), GSP (global softmax pooling) and cATP (class-wise attention pooling), and give an explanation about why cATP perform best from the perspective of the high-level feature space of the neural network encoder. This explanation enables us to focus on a potential decision surface which we term the specialized decision surface (SDS). SDS shows its power in frame-level prediction (event detection). Actually, similar to SDS, something like "attention output" or "attention mask" might have been exploited in other works. However, the detailed explanation remains explored. Therefore, we explore a detailed explanation for cATP and provide the detailed analysis of experimental results. We argue that the explanation of SDS is expected to promote the optimizing of the pooling module in weakly-supervised learning.  
  - Disentangled feature (DF) is proposed to ease the problem caused by the co-occurrence of categories. We describe it in detail in the paper.  
  
  The model architecture utilized in our experiments is the same as the PS-model discussed in the next section.  


### Guided learning (GL)
  We propose a semi-supervised learning method named guided learning (GL) fitting for weakly-supervised learning in paper [What you need is a more professional teacher](https://arxiv.org/abs/1906.02517).  

  Here are 2 model architectures utilized in our experiments:  
  ![image](https://github.com/Kikyo-16/Sound_event_detection/blob/master/image/fig1.png)

## How to use
### Quick start
Scripts in "Scripts" directory help quick start.  

(Before running scripts, make sure for the [data preparation](#user-content-data-preparation) )  

You can try to run  
#### sh scripts/cATP-2018.sh  
to train model with cATP-SDS-DF1 on the dataset of DCASE2018 task4;  

Run
#### sh scripts/semi-2018.sh  
to train model with GL on the dataset of DCASE2018 task4;  

Run
#### sh scripts/semi-2019.sh  
to train model on the dataset of DCASE2019 task4.  

### Configure files  
You can find details in example configure ctories (DCASE2018-task4, DCASE2018-task4_semi, DCASE2019-task4).  

## Details of the code implement
### Data preparation
  `scripts` provides some example scripts to help data preparation:  
    - `gen_feature-2018.sh` help extract feature for the dataset of DCASE2018 tas4  
    - `gen_feature-2019.sh` help extract feature for the dataset of DCASE2019 tas4  
    - `gen_label-2018.sh` help format labels of the dataset of DCASE2018 tas4  
    - `gen_label-2019.sh` help format labels of the dataset of DCASE2019 tas4  
    
  Before running any example script to extract feature, make sure there are a audio directory to storedoriginal audio files and a feature directory to store output feature. The example script requires `data/wav` as the audio directory and `data/feature` as the feature directory.  
  
  You can down load audio files from the website of DCASE and store all the audio files in the audio directory before running the scripts.  
  
  If you experience any trouble downloading the audio files, you can contact the organizers of DCASE task4. Or send me an E-mail, I'll be glad to help you.  
  
  Similarly, before running any example script to generate labels, make sure there is a label directory to store labels. The example script requires `data/label` as the label directory.  
  
  We provide data lists in `data/text`. The only difference from the original dataset from DCASE2018 task4 is that we provide a file list `data/text/all-2018.csv` with noisy annotations for the combination of weakly labeled training and unlabeled in domain training set. The noisy annotations, as mentioned in our paper [Specialized Decision Surface and DisentangledFeature for Weakly-Supervised Polyphonic Sound Event Detection](https://arxiv.org/abs/1905.10091v3) are obtained roughly by using a PT-model to tag unlabeled data. We release it for reproducing our experiments and we'll be so glad if it is helpful to your research.  
  
### Source codes
#### feature_extraction
Tools to help extract feature and generate labels.  
#### src
Source codes. See details in the source codes.  

## Contact us
Please don't hesitate to contact us should you have any question. You can email me at `linliwei17g@ict.ac.cn` or `1174436431@qq.com`.
