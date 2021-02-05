from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB
import os
import torch
import math
import time
import numpy as np

X = []
y = []
labell = {}
screenName = {}
longevity = {}
followers_count = {}
friends_count = {}
statuses_count = {}
description = {}
devia1 = {}
devia2 = {}

f = open('./fianlList.txt', 'r', encoding='UTF-8')
for line in f:
    labell[line.split()[1]] = line.split()[2]
    screenName[line.split()[1]] = line.split()[0]
f.close()

for i in range(1, 41):
    files = os.listdir('../profile/' + str(i))
    for file in files:
        f = open('../profile/' + str(i) + '/' + file)
        flag = 0
        for line in f:
            if line == 'created_at\n':
                flag = 1
                continue
            if flag == 1:
                flag = 0
                longevity[file[:-8]] = int((time.time() - time.mktime(time.strptime(line[:-1], '%a %b %d %H:%M:%S +0000 %Y')))
                                           / 60 / 60 / 24)
                continue
            if line == 'followers_count\n':
                flag = 2
                continue
            if flag == 2:
                flag = 0
                followers_count[file[:-8]] = int(line[:-1])
            if line == 'friends_count\n':
                flag = 3
                continue
            if flag == 3:
                flag = 0
                friends_count[file[:-8]] = int(line[:-1])
            # +ID标准差
            if line == 'statuses_count\n':
                flag = 4
                continue
            if flag == 4:
                flag = 0
                statuses_count[file[:-8]] = int(line[:-1])
            if line == 'description\n':
                flag = 5
                continue
            if flag == 5:
                flag = 0
                description[file[:-8]] = line[:-1]
        f.close()
    files = os.listdir('../neighbor/' + str(i))
    for file in files:
        f = open('../neighbor/' + str(i) + '/' + file)
        flag = 0
        tmp1 = []
        tmp2 = []
        for line in f:
            if line == 'followingSet:\n':
                flag = 1
                continue
            if flag == 1 and line != 'followerSet:\n':
                tmp1.append(int(line[:-1]))
                continue
            if line == 'followerSet:\n':
                flag = 2
                continue
            if flag == 2:
                tmp2.append(int(line[:-1]))
                continue
        f.close()
        # print(np.isnan(np.std(tmp1)))
        # print(np.isnan(np.std(tmp2)))
        devia1[file[:-9]] = np.std(tmp1)
        devia2[file[:-9]] = np.std(tmp2)


files = open('./listTrain.txt', 'r', encoding='UTF-8')
for file in files:
    x = []
    x.append(len(screenName[file[:-1]]))  # length of screen name
    x.append(len(description[file[:-1]]))  # length of description
    x.append(longevity[file[:-1]])
    x.append(friends_count[file[:-1]])  # number of followings
    x.append(followers_count[file[:-1]])  # number of followers
    x.append(friends_count[file[:-1]] / (followers_count[file[:-1]] + .1))  # the ratio
    if np.isnan(devia1[file[:-1]]):
        x.append(0)
    else:
        x.append(devia1[file[:-1]])
    if np.isnan(devia2[file[:-1]]):
        x.append(0)
    else:
        x.append(devia2[file[:-1]])
    x.append(statuses_count[file[:-1]])  # number of posted tweets
    x.append(statuses_count[file[:-1]] / longevity[file[:-1]])  # posted tweets per day
    X.append(x)
    y.append(float(labell[file[:-1]]))
files.close()
clf = RandomForestRegressor()
clf.fit(X, y)

files = open('./listDev.txt', 'r', encoding='UTF-8')
cnt = 0
TP = 0
TN = 0
FP = 0
FN = 0
for file in files:
    cnt = cnt + 1
    x = []
    x.append(len(screenName[file[:-1]]))  # length of screen name
    x.append(len(description[file[:-1]]))  # length of description
    x.append(longevity[file[:-1]])
    x.append(friends_count[file[:-1]])  # number of followings
    x.append(followers_count[file[:-1]])  # number of followers
    x.append(friends_count[file[:-1]] / (followers_count[file[:-1]] + .1))  # the ratio
    if np.isnan(devia1[file[:-1]]):
        x.append(0)
    else:
        x.append(devia1[file[:-1]])
    if np.isnan(devia2[file[:-1]]):
        x.append(0)
    else:
        x.append(devia2[file[:-1]])
    x.append(statuses_count[file[:-1]])  # number of posted tweets
    x.append(statuses_count[file[:-1]] / longevity[file[:-1]])  # posted tweets per day
    pred = clf.predict([x])
    label = float(labell[file[:-1]])
    if pred <= 0.5 and label == 0:
        TN = TN + 1
    elif pred <= 0.5 and label == 1:
        FN = FN + 1
    elif pred >= 0.5 and label == 0:
        FP = FP + 1
    elif pred >= 0.5 and label == 1:
        TP = TP + 1
    else:
        print('ERROR!')

precision = TP / (TP + FP)
recall = TP / (TP + FN)
acc = (TP + TN) / cnt
F1 = TP / (TP + 0.5 * (FP + FN))
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print('val: ')
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('acc: ' + str(acc))
print('F1: ' + str(F1))
print('MCC: ' + str(MCC))




files = open('./listTest.txt', 'r', encoding='UTF-8')
cnt = 0
TP = 0
TN = 0
FP = 0
FN = 0
for file in files:
    cnt = cnt + 1
    x = []
    x.append(len(screenName[file[:-1]]))  # length of screen name
    x.append(len(description[file[:-1]]))  # length of description
    x.append(longevity[file[:-1]])
    x.append(friends_count[file[:-1]])  # number of followings
    x.append(followers_count[file[:-1]])  # number of followers
    x.append(friends_count[file[:-1]] / (followers_count[file[:-1]] + .1))  # the ratio
    if np.isnan(devia1[file[:-1]]):
        x.append(0)
    else:
        x.append(devia1[file[:-1]])
    if np.isnan(devia2[file[:-1]]):
        x.append(0)
    else:
        x.append(devia2[file[:-1]])
    x.append(statuses_count[file[:-1]])  # number of posted tweets
    x.append(statuses_count[file[:-1]] / longevity[file[:-1]])  # posted tweets per day
    pred = clf.predict([x])
    label = float(labell[file[:-1]])
    if pred <= 0.5 and label == 0:
        TN = TN + 1
    elif pred <= 0.5 and label == 1:
        FN = FN + 1
    elif pred >= 0.5 and label == 0:
        FP = FP + 1
    elif pred >= 0.5 and label == 1:
        TP = TP + 1
    else:
        print('ERROR!')

precision = TP / (TP + FP)
recall = TP / (TP + FN)
acc = (TP + TN) / cnt
F1 = TP / (TP + 0.5 * (FP + FN))
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print('test: ')
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('acc: ' + str(acc))
print('F1: ' + str(F1))
print('MCC: ' + str(MCC))

