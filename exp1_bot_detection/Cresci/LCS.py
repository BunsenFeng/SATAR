import math
import time

beginn = time.time()
result = [1] * 1184
list = []
flag = ''


def issub(strrr):
    global list
    cnt = 0
    for dna in list:
        if strrr in dna:
            cnt = cnt + 1
    return cnt


def tryy(strrr):
    num = issub(strrr)
    if num > 1:
        global result
        if num == 50 and len(strrr) == 134:
            global flag
            flag = strrr
            print('num: ' + str(num))
            print('flag: ' + flag)
        for tmp in range(num):
            result[tmp] = max(result[tmp], len(strrr))
        tryy(strrr + 'A')
        tryy(strrr + 'C')
        tryy(strrr + 'T')

Sigma = ['A', 'C', 'T']

f = open('./tweet/finalTest1.txt', 'r', encoding='UTF-8')
for line in f:
    list.append(line[line.find(' ', 2) + 1: -1])
f.close()

for substring in Sigma:
    tryy(substring)

for i in result:
    print(i)

f = open('./tweet/finalTest1.txt', 'r', encoding='UTF-8')
TP = 0
FP = 0
TN = 0
FN = 0
for line in f:
    strr = line[line.find(' ', 2) + 1: -1]
    label = line[0]
    if flag in strr and label == '1':
        TP = TP + 1
    elif flag not in strr and label == '0':
        TN = TN + 1
    elif flag in strr and label == '0':
        FP = FP + 1
    else:
        FN = FN + 1
f.close()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
acc = (TP + TN) / 1183
specificity = TN / (TN + FP)
F1 = TP / (TP + 0.5 * (FP + FN))
MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

print('TP: ' + str(TP))
print('FP: ' + str(FP))
print('TN: ' + str(TN))
print('FN: ' + str(FN))
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('acc: ' + str(acc))
print('specificity: ' + str(specificity))
print('F1: ' + str(F1))
print('MCC: ' + str(MCC))
print('time: ' + str(time.time() - beginn))



