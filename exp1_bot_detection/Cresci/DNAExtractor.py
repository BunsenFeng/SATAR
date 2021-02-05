import os


path = './ours_text'
final = open(path + '/final.txt', 'w', encoding='UTF-8')
for i in range(40):
    files = os.listdir(path + '/' + str(i))
    for file in files:
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
                if data[0 : 4] == 'RT @':
                    dna = dna + 'C'  # C is for a retweet
                elif data[0] == '@':
                    dna = dna + 'T'  # T is for a reply
                else:
                    dna = dna + 'A'  # A is for a simple tweet
                data = ''
        f.close()
        final.write(file + ' ' + dna + '\n')
final.close()

