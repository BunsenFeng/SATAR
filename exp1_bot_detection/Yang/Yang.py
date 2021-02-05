import os
from sklearn.ensemble import RandomForestClassifier
import joblib
import datetime
import time

IDList = []
labelList = []
with open('finalList.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        IDList.append(line.split()[1])
        labelList.append(line.split()[2])
print('load done')
featuretext = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
          'listed_count', 'default_profile', 'profile_use_background_image', 'verified']

def loaddata(file):
    data = []
    with open(file, 'r', encoding = 'utf-8') as f:
        for line in f:
            data.append(line.split()[0])
    return data
trainList = loaddata('listTrain.txt')
developList = loaddata('listDev.txt')
testList = loaddata('listTest.txt')
print('load done')
traindata = []
trainlabel = []
devdata = []
devlabel = []
testdata = []
testlabel = []

def calchours(data):
    dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5,
           'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 
            'Nov':11, 'Dec':12}
    tmp = data['created_at'].split()
    year = int(tmp[5])
    month = dict[tmp[1]]
    day = int(tmp[2])
    hour = int(tmp[3].split(':')[0])
    minute = int(tmp[3].split(':')[1])
    second = int(tmp[3].split(':')[2])
    date1 = datetime.datetime(year, month, day, hour, minute, second)
    date2 = datetime.datetime(2020, 8, 20, 17, 0, 0)
    return (date2 - date1).days * 24.0 + int((date2 - date1).seconds / 3600.0)

def calcnum(data):
    cnt = 0
    for index in range(len(data)):
        if data[index] >= '0' and data[index] <= '9':
            cnt += 1
    return cnt
def calclen(data):
    cnt = 0
    for word in data:
        cnt += len(word)
    return cnt

unigram = {}
with open('unigram.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        unigram[line.split()[0]] = int(line.split()[1])
bigram = {}
with open('bigram.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        word = (line.split()[0], line.split()[1])
        bigram[word] = int(line.split()[2])
print('gram load done')
def calclikely(data):
    ans = 1
    for index in range(len(data) - 1):
        word0 = data[index]
        word1 = data[index+1]
        try:
            tmp = bigram[(word0, word1)] / unigram[word0]
        except:
            tmp = 0
        ans = ans * tmp
    ans = ans ** (1 / (len(data) - 1))
    return ans
cnt = 0
for index in range(len(IDList)):
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
    ID = IDList[index]
    label = labelList[index]
    data = []
    with open('profile/' + ID + '_pro.txt', 'r', encoding = 'utf-8') as f:
        for line in f:
            data.append(line.strip())
    user = {}
    for i in range(int(len(data) / 2)):
        user[data[i*2]] = data[i*2+1]
    feature = []
    user_age = calchours(user)
    feature.append(int(user['statuses_count']) / user_age)
    feature.append(int(user['followers_count']) / user_age)
    feature.append(int(user['friends_count']) / user_age)
    feature.append(int(user['favourites_count']) / user_age)
    feature.append(int(user['listed_count']) / user_age)
    feature.append(int(user['followers_count']) / max(1, int(user['friends_count'])))
    feature.append(len(user['screen_name']))
    feature.append(calcnum(user['screen_name']))
    feature.append(len(user['name']))
    feature.append(calcnum(user['name']))
    feature.append(calclen(user['description']))
    feature.append(calclikely(user['screen_name'])) 
    for key in user:
        if user[key] == '' or user[key] == 'NULL' or user[key] == 'False':
            user[key] = '0'
        if user[key] == 'True':
            user[key] = '1'
    for key in featuretext:
        feature.append(int(user[key]))
    with open('feature.txt', 'a', encoding = 'utf-8') as f:
        f.write(ID + ' ')
        for key in feature:
            f.write(str(key) + ' ')
        f.write('\n')
    if ID in trainList:
        traindata.append(feature)
        trainlabel.append(label)
    elif ID in developList:
        devdata.append(feature)
        devlabel.append(label)
    elif ID in testList:
        testdata.append(feature)
        testlabel.append(label)  
print('load done')
print(len(testdata))
def Metric(truth, pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(truth)):
        if truth[i] == '1' and pred[i] == '1':
            TP += 1
        elif truth[i] == '0' and pred[i] == '1':
            FP += 1
        elif truth[i] == '0' and pred[i] == '0':
            TN += 1
        elif truth[i] == '1' and pred[i] == '0':
            FN += 1
    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    F1 = 2 / (recall ** -1 + precision ** -1)
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    print('precision:', precision)
    print('recall:', recall)
    print('specificity:', specificity)
    print('F1:', F1)
    print('MCC:', MCC)
    print('acc:', acc)
    return acc    
clf = RandomForestClassifier(random_state=0, max_depth = 10, verbose = True)
print('training start')
clf.fit(traindata, trainlabel)
print('train end')
devpred = clf.predict(devdata)
testpred = clf.predict(testdata)
print('dev data')
dev_acc = Metric(devlabel, devpred)
print('test data')
test_acc = Metric(testlabel, testpred)
joblib.dump(clf, 'models/model_' + str(int(dev_acc * 10000)) + '_' + str(int(test_acc * 10000)) + '.m')