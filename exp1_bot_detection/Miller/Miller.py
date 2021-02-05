from sklearn.cluster import KMeans
import numpy as np
import random
n_clusters = 5
cls = KMeans(n_clusters = n_clusters)
ID = []
label = {}
pred = {}
feature = {}
with open('stdF.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        user = line.split()[0]
        ID.append(user)
        label[user] = int(line.split()[-1])
        feature[user] = [eval(x) for x in line.split()[1:-1]]
core = []
data = []
with open('listTrain.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        data.append(line)
random.shuffle(data)
for line in data:
    tmp = line.strip()
    if label[tmp] == 1:
        core.append(tmp)
        if(len(core) == 200 * n_clusters):
            break
X = []
for tmp in core:
    pred[tmp] = 0
    X.append(feature[tmp])
Y = cls.fit_predict(X)

C = []
for i in range(n_clusters):
    C.append([])
for index in range(len(core)):
    C[Y[index]].append(core[index])
eps = 1.8
random.shuffle(ID)
def cald(x, y):
    return np.sum((np.array(x) - np.array(y)) ** 2) ** 0.5
def cal(C, x):
    ans = []
    for i in range(len(C)):
        tmp = 1e20
        for j in range(len(C[i])):
            tmp = min(tmp, cald(feature[C[i][j]], feature[x]))
        ans.append(tmp)
    return ans
cnt = 0
for tmp in ID:
    if tmp in core:
        continue
    flg = False
    dis = cal(C, tmp)
    if np.min(dis) <= eps:
        #C[np.argmin(dis)].append(tmp)
        pred[tmp] = 0
    else:
        pred[tmp] = 1
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
        
TP = 0
TN = 0
FP = 0
FN = 0
with open('listTest.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        user = line.strip()
        if label[user] == 1 and pred[user] == 1:
            TP += 1
        elif label[user] == 0 and pred[user] == 1:
            FP += 1
        elif label[user] == 0 and pred[user] == 0:
            TN += 1
        elif label[user] == 1 and pred[user] == 0:
            FN += 1
acc = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
F1 = 2 / (recall ** -1 + precision ** -1)
MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
print(TP, FP, TN, FN)
print('precision:', precision)
print('recall:', recall)
print('specificity:', specificity)
print('F1:', F1)
print('MCC:', MCC)
print('acc:', acc)