import os
import torch

path = './cresci_text'
final = open(path + '/finalTest1.txt', 'w', encoding='UTF-8')
for i in range(6):
    files = os.listdir(path + '/' + str(i))
    for file in files:
        check = open('./listTest_Cresci1.txt', 'r', encoding='UTF-8')
        if file[:-4] + '\n' not in check:
            check.close()
            continue
        check.close()
        f = open(path + '/' + str(i) + '/' + file, 'r', encoding='UTF-8')
        data = ''
        dna = ''
        for line in f:
            if line != '-----------------------\n':
                data = data + line
            else:
                if len(data) <= 4:
                    data = ''
                    continue
                if data[0: 4] == 'RT @':
                    dna = dna + 'C'  # C is for a retweet
                elif data[0] == '@':
                    dna = dna + 'T'  # T is for a reply
                else:
                    dna = dna + 'A'  # A is for a simple tweet
                data = ''
        f.close()
        label = torch.load('../whr_tmp/Cresci_tensor/' + file[:-4] + '.pt')['label']
        final.write(label + ' ' + file + ' ' + dna + '\n')
final.close()





