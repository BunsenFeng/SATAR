### SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection
This repository serves as a code listing for the paper 'SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection' to appear in CIKM 2021.

#### Introduction to SATAR
SATAR is a self-supervised Twitter user representation learning framework that is proposed to improve user representation learning and promote generalizable and adaptable bot detection.

The challenge of generalization in social media bot detection demands bot detectors to simultaneously identify bots that attack in many different ways and exploit diversified features on Twitter. However, previous bot detection methods fail to generalize since they only leverage limited user information and are trained on datasets with few types of bots. SATAR is designed to generalize by jointly leveraging all three aspects of user information, namely semantic, property and neighborhood information.

Apart from that, the challenge of adaptation in bot detection demands bot detectors to maintain desirable performance in different times and catch up with rapid bot evolution. However, previous bot detection measures rely heavily on feature engineering and are not designed to adapt to emerging trends in bot evolution. SATAR is designed to adapt by pre-training on mass self-supervised users and fine-tuning on specific bot detection scenarios.

#### Affiliated Paper
The affiliated paper, "SATAR: A Self-supervised Approach to Twitter Account Representation Learning and its Application in Bot Detection" is accepted at CIKM 2021.

#### Code Listing
Codes of experiments on TwiBot-20 and two other datasets are listed in ./exp1_bot_detection/ and the name of the folders are correspondes to the methods respectively. Codes of ablation study and results are in ./exp2_ablation_study/ . Codes for data pre-processing are in ./data_preprocessing/ .

#### TwiBot-20 Dataset
TwiBot-20 is a comprehensive sample of the Twittersphere and it is representative of the current generation of Twitter bots and genuine users. It is naturally divided into four domains: politics, business, entertainment and sports and each user has semantics, property and neiborhood information. You can find more information about Twibot-20 in the paper 'TwiBot-20: A Novel Benchmark for Twitter Bot Detection' or visit https://github.com/BunsenFeng/TwiBot-20.



