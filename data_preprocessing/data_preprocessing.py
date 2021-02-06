import os
import re
import emoji
import requests
import hashlib
from selenium import webdriver
import time

driver = webdriver.Chrome(executable_path="chromedriver.exe")
driver.get('https://translate.google.com/#view=home&op=translate&sl=auto&tl=en')


def translate(text):
    while True:
        try:
            tmp = driver.find_element_by_xpath(
                '/html/body/div[2]/div[2]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div/div/div')
            tmp.click()
        except:
            print(1)
        # else:
        # print(2)
        time.sleep(2)
        text_dummy = driver.find_element_by_class_name('tlid-source-text-input')
        text_dummy.clear()
        try:
            text_dummy.send_keys(text)
        except:
            return ''
        time.sleep(2)
        try:
            text_translation = driver.find_element_by_xpath(
                '/html/body/div[2]/div[2]/div[1]/div[2]/div[1]/div[1]/div[2]/div[3]/div[1]/div[2]/div/span[1]')
        except Exception as e:
            print('wait')
        else:
            break
    return text_translation.text

pppath = 'C:/Users/mrwan/Desktop/tweet'
index_to_trans = []
content_to_trans = ''
len_now = 0
for i in range(1, 41):
    files = os.listdir(pppath + '/' + str(i))
    data = ''
    for file in files:
        flg = True
        dataFinal = []
        if os.path.exists(pppath + '/new(' + str(i) + ')/' + file) == True:
            continue
        f = open(pppath + '/' + str(i) + '/' + file, 'r', encoding='utf-8')
        rr = open(pppath + '/new(' + str(i) + ')/' + 'remain.txt', 'a', encoding='utf-8')
        cnt = 0
        i_th = 0
        for line in f:
            # cnt += 1
            # if cnt % 100 == 0:
            # print(str(cnt/100))
            tmp = line
            if tmp.strip() != '-------------------------------------':
                data += tmp
            else:
                if len(data.strip()) == 0:
                    data = ''
                    continue
                # print(data)
                if data[0: 4] == "RT @":
                    data = ''
                    continue
                head = data.find('@')
                while head != -1:
                    tail1 = data[head:].find(' ')
                    tail2 = data[head:].find('\n')
                    if tail1 != -1 and tail2 != -1:
                        tail = min(tail1, tail2)
                    else:
                        tail = max(tail1, tail2)
                    data = data[0: head] + ' _bsta_ ' + data[head + tail:]
                    head = data.find('@')

                head = data.find('#')
                while head != -1:
                    tail1 = data[head:].find(' ')
                    tail2 = data[head:].find('\n')
                    if tail1 != -1 and tail2 != -1:
                        tail = min(tail1, tail2)
                    else:
                        tail = max(tail1, tail2)
                    data = data[0: head] + ' _gat_ ' + data[head + tail:]
                    head = data.find('#')

                head = data.find('https://t.co/')
                while head != -1:
                    data = data[0: head] + ' _knil_ ' + data[head + 23:]
                    head = data.find('https://t.co/')

                head = data.find('http://t.co/')
                while head != -1:
                    data = data[0: head] + ' _knil_ ' + data[head + 22:]
                    head = data.find('http://t.co/')

                head = data.find('<a href=')
                while head != -1:
                    tail = data.find('</a>')
                    data = data[0: head] + ' _knil_ ' + data[head + tail:]
                    head = data.find('<a href=')

                data = emoji.demojize(data)
                data = re.sub(r':', ' ', data)
                data = re.sub(r"['’]s", r" 's", data)
                data = re.sub(r"n['’]t", " n't", data)
                data = data.replace(r"&", " and ")
                data = data.replace("\n", " ")
                if len(data.strip()) == 0:
                    data = ''
                    continue
                    # áéíóúüñ¿¡ÁÉÍÓÚÜÑ
                if re.search(r"[^A-Za-z0-9'\n\s\!@#\$%\^&\*\(\)_\+-=\.,\{\}\|\?\\\[\]<>/`~\"…:“”‘’（）【】、—《！》；，。？–♫]",
                             data) != None:
                    # data = translate(text=data)
                    index_to_trans.append(i_th)
                    content_to_trans += (data + ' _b_ \n')
                    len_now = len(content_to_trans)
                    dataFinal.append('TBD\n')
                    data = ''
                    if len_now > 3000:
                        # print(index_to_trans)
                        temp = translate(text=content_to_trans)
                        # temp = translate(text=temp)
                        temp = temp.split('\n')
                        # print(temp)
                        # print(index_to_trans)
                        # print(len(dataFinal))
                        # print('tweet cnt: ' + str(len(index_to_trans)))
                        # print('_cUt_ split cnt: ' + str(len(temp)))
                        for j in range(len(index_to_trans)):
                            if j >= len(temp):
                                flg = False
                                rr.write(file + '\n')
                                print(file)
                                break
                            mystr = re.sub(r"[^A-Za-z0-9'_]", " ", temp[j])
                            mystr = mystr + ' '
                            mystr = re.sub(r"\s{2,}", " ", mystr) + '\n'
                            dataFinal[index_to_trans[j]] = mystr
                        len_now = 0
                        content_to_trans = ''
                        index_to_trans = []
                    continue

                data = re.sub(r"[^A-Za-z0-9'_]", " ", data)
                data = data + ' '
                data = re.sub(r"\s{2,}", " ", data)
                data = data + '\n'
                dataFinal.append(data)
                data = ''
                i_th += 1

        if len_now != 0:
            temp = translate(text=content_to_trans)
            # temp = translate(text=temp)
            temp = temp.split('\n')
            # print(temp)
            # print(index_to_trans)
            # print(dataFinal)
            # print('tweet cnt: ' + str(len(index_to_trans)))
            # print('_cUt_ split cnt: ' + str(len(temp)))
            for j in range(len(index_to_trans)):
                if j >= len(temp):
                    flg = False
                    rr.write(file + '\n')
                    print(file)
                    break
                mystr = re.sub(r"[^A-Za-z0-9'_]", " ", temp[j])
                mystr = mystr + ' '
                mystr = re.sub(r"\s{2,}", " ", mystr)
                dataFinal[index_to_trans[j]] = mystr
            len_now = 0
            content_to_trans = ''
            index_to_trans = []
        f.close()
        if flg == False:
            continue
        f = open(pppath + '/new(' + str(i) + ')/' + file, 'w', encoding='utf-8')  

        for line in dataFinal:
            if line == 'TBD\n':
                continue
            f.write(line)
        f.close()
        rr.close()
