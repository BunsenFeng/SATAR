### SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection
This repository serves as a code listing for the paper 'SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection' to appear in CIKM 2021.

#### Introduction to SATAR
SATAR is a self-supervised Twitter user representation learning framework that is proposed to improve user representation learning and promote generalizable and adaptable bot detection.

The challenge of generalization in social media bot detection demands bot detectors to simultaneously identify bots that attack in many different ways and exploit diversified features on Twitter. However, previous bot detection methods fail to generalize since they only leverage limited user information and are trained on datasets with few types of bots. SATAR is designed to generalize by jointly leveraging all three aspects of user information, namely semantic, property and neighborhood information.

Apart from that, the challenge of adaptation in bot detection demands bot detectors to maintain desirable performance in different times and catch up with rapid bot evolution. However, previous bot detection measures rely heavily on feature engineering and are not designed to adapt to emerging trends in bot evolution. SATAR is designed to adapt by pre-training on mass self-supervised users and fine-tuning on specific bot detection scenarios.

#### Affiliated Paper
The affiliated paper, "SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection" is accepted at CIKM 2021.

```
@inproceedings{10.1145/3459637.3481949,
author = {Feng, Shangbin and Wan, Herun and Wang, Ningnan and Li, Jundong and Luo, Minnan},
title = {SATAR: A Self-Supervised Approach to Twitter Account Representation Learning and Its Application in Bot Detection},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3481949},
doi = {10.1145/3459637.3481949},
abstract = {Twitter has become a major social media platform since its launching in 2006, while
complaints about bot accounts have increased recently. Although extensive research
efforts have been made, the state-of-the-art bot detection methods fall short of generalizability
and adaptability. Specifically, previous bot detectors leverage only a small fraction
of user information and are often trained on datasets that only cover few types of
bots. As a result, they fail to generalize to real-world scenarios on the Twittersphere
where different types of bots co-exist. Additionally, bots in Twitter are constantly
evolving to evade detection. Previous efforts, although effective once in their context,
fail to adapt to new generations of Twitter bots. To address the two challenges of
Twitter bot detection, we propose SATAR, a self-supervised representation learning
framework of Twitter users, and apply it to the task of bot detection. In particular,
SATAR generalizes by jointly leveraging the semantics, property and neighborhood information
of a specific user. Meanwhile, SATAR adapts by pre-training on a massive number of
self-supervised users and fine-tuning on detailed bot detection scenarios. Extensive
experiments demonstrate that SATAR outperforms competitive baselines on different
bot detection datasets of varying information completeness and collection time. SATAR
is also proved to generalize in real-world scenarios and adapt to evolving generations
of social media bots.},
booktitle = {Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
pages = {3808–3817},
numpages = {10},
keywords = {self-supervised learning, representation learning, twitter bot detection, social media},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}

```

#### Code Listing
Codes of experiments on TwiBot-20 and two other datasets are listed in ./exp1_bot_detection/ and the name of the folders are correspondes to the methods respectively. Codes of ablation study and results are in ./exp2_ablation_study/ . Codes for data pre-processing are in ./data_preprocessing/ .

#### TwiBot-20 Dataset
TwiBot-20 is a comprehensive sample of the Twittersphere and it is representative of the current generation of Twitter bots and genuine users. It is naturally divided into four domains: politics, business, entertainment and sports and each user has semantics, property and neiborhood information. You can find more information about Twibot-20 in the paper 'TwiBot-20: A Novel Benchmark for Twitter Bot Detection' or visit https://github.com/BunsenFeng/TwiBot-20.



