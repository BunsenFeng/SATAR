import math

result = [1] * 4355
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
        for tmp in range(num):
            result[tmp] = max(result[tmp], len(strrr))
        tryy(strrr + 'A')
        tryy(strrr + 'C')
        tryy(strrr + 'T')

Sigma = ['A', 'C', 'T']

f = open('./cresci_text/finalTest1.txt', 'r', encoding='UTF-8')
for line in f:
    list.append(line[line.find(' ', 2) + 1: -1])
f.close()

for substring in Sigma:
    tryy(substring)

for i in result:
    print(i)




