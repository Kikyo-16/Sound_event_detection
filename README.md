# Sound event detection
## Introduction
This code aims at semi-supervised and weakly-supervised sound event detection. The dataset utilized in our experiments is from DCASE (IEEE AASP Challenge on Detection and Classification of Acoustic Scenes and Events), more specifically, from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). The code embraces two methods we proposed to solve this task: [disentangled feature (DF)](https://arxiv.org/abs/1905.10091) for weakly-supervised learning and [guided learning (GL)](https://arxiv.org/abs/1906.02517) for semi-supervised learning. We're so glad if you're interested in using it for research purpose or DCASE participation. Please don't hesitate to contact us should you have any question.
## Something about DCASE
The dataset utilized in our experiments is from [DCASE2018 task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) and [DCASE2019 task4](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments). We've encapsulated some interfaces to handle dataset of DCASE task4 and streams to read the data, so this code can be used directly for the future challenge (or as a basis for fine-tuning).
Actually, we exploited it to participate in DCASE2019 task4 and won the first price ([our technique report](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Lin_25.pdf) and [challenge results](http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments-results)). After adding analysis of result and making some modifications, we submitted the new technical report (to be updated soon) to [DCASE2019 Workshop](http://dcase.community/workshop2019/). 
## Main ideas comprised in the code
### Disentangled feature
### Guided learning
## Details of the code implement
## How to use
### Quick start
### Configure files
### Feature extraction
### Source codes
## Contact us
Please don't hesitate to contact us should you have any question. You can email me at `linliwei17g@ict.ac.cn` or `1174436431@qq.com`.
