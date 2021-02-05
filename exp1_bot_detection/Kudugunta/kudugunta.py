import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(8)


# torch.manual_seed(4)
# random.seed()


class TweetDataset(Dataset):

    def __init__(self, file):
        self.file = file.replace('.txt', '')
        self.IDList = []
        with open(file, 'r', encoding='utf-8') as f:
            for ID in f:
                self.IDList.append(ID.split()[0])

    def __len__(self):
        return len(self.IDList)

    def __getitem__(self, index):
        ID = self.IDList[index]
        user = {}
        pathP = './property/' + ID + '.pth'
        user['property'] = torch.load(pathP)
        pathT = './ours_tensor_lstm/' + ID + '.pt'
        tweet = torch.load(pathT)
        user['tweet'] = tweet['tweet']
        user['label'] = torch.tensor(int(tweet['label']))
        return user


class LSTMClassifier(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 property_size,
                 linear_dim=20,
                 label_size=2,
                 batch_size=16,
                 ):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(num_embeddings=len(embed_mat), embedding_dim=len(embed_mat[0]))
        self.word_embeddings.weight.data = embed_mat
        self.word_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.hidden2label = nn.Linear(linear_dim, label_size)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_dim + property_size, linear_dim)
        self.relu1 = nn.ReLU()

    def init_hidden(self):
        return (
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        )

    def forward(self, sentence, property):
        self.batch_size = len(sentence)
        sentence = sentence.transpose(0, 1)
        self.hidden = self.init_hidden()
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.linear(torch.cat((torch.mean(lstm_out[-1], dim=0).float(), property.squeeze().float())))
        y = self.relu1(y)
        y = self.hidden2label(y)
        log_probs = F.log_softmax(y, dim=0)
        return log_probs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / (len(truth) + 1)


def train():
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    EPOCH = 50
    BATCH_SIZE = 1
    WEIGHT_DECAY = 1e-2
    SMALL_STEP_EPOCH = 5

    # fullDataset = TweetDataset('./PTH/accountList.txt')
    # trSize = int(len(fullDataset) * 0.7)
    # devSize = int(len(fullDataset) * 0.2)
    # testSize = len(fullDataset) - trSize - devSize
    # trainDataset, devDataset, testDataset = torch.utils.data.random_split(fullDataset, [trSize, devSize, testSize])
    trainLoader = DataLoader(dataset=TweetDataset('./listTrain.txt'),
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             )
    devLoader = DataLoader(dataset=TweetDataset('./listDev.txt'),
                           batch_size=BATCH_SIZE,
                           shuffle=True)
    testLoader = DataLoader(dataset=TweetDataset('./listTest.txt'),
                            batch_size=BATCH_SIZE,
                            shuffle=True)

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM,
                           label_size=2,
                           batch_size=BATCH_SIZE,
                           property_size=10,
                           )
    device_ids = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    bestTillNow = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, trainLoader, loss_function, optimizer, i)
        dev_acc = evaluate_epoch(model, devLoader, loss_function, i, 1)
        test_acc = evaluate_epoch(model, testLoader, loss_function, i, 2)
        if dev_acc > bestTillNow:
            bestTillNow = dev_acc
            print('New best model! Acc = ' + str(dev_acc))
            torch.save(model.module.state_dict(), './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')
        if i == SMALL_STEP_EPOCH:
            optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay = WEIGHT_DECAY)


def train_epoch(model,
                train_data,
                loss_function,
                optimizer,
                i):
    model.train()
    avg_loss = 0
    acc = 0
    cnt = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    sumLoss = 0
    numToBackward = 16
    for batch in train_data:
        cnt += 1
        # if cnt % 100 == 0:
        #     print('Begin ' + str(int(cnt / 100)) + ' batch in an epoch!')
        label = batch['label']
        sent = batch['tweet'].squeeze(0)
        property = batch['property']
        label = label.cuda()
        sent = sent.cuda()
        property = property.cuda()
        pred = model(sent, property).unsqueeze(0)
        if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][1] and label == 1:
            acc = acc + 1
        if pred[0][0] > pred[0][1] and label == 0:
            TP = TP + 1
        if pred[0][0] > pred[0][1] and label == 1:
            FP = FP + 1
        if pred[0][0] < pred[0][1] and label == 1:
            TN = TN + 1
        if pred[0][0] < pred[0][1] and label == 0:
            FN = FN + 1

        loss = loss_function(pred, label)
        avg_loss += loss.item()
        sumLoss = sumLoss + loss
        model.zero_grad()
        if cnt % numToBackward == 0:
            # print('Acc now: ' + str(acc / cnt))

            sumLoss = sumLoss / float(numToBackward)
            sumLoss.backward()
            sumLoss = 0
            optimizer.step()
    avg_loss = avg_loss / cnt
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = acc / float(cnt)
    specificity = TN / (TN + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print('epoch ' + str(i))
    print('train: ')
    print('loss: ' + str(avg_loss))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('acc: ' + str(acc))
    print('specificity: ' + str(specificity))
    print('F1: ' + str(F1))
    print('MCC: ' + str(MCC))


def evaluate_epoch(model,
                   train_data,
                   loss_function,
                   i,
                   ii):
    model.eval()
    avg_loss = 0
    acc = 0
    cnt = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for batch in train_data:
        cnt = cnt + 1
        label = batch['label']
        sent = batch['tweet'].squeeze(0)
        property = batch['property']
        label = label.cuda()
        sent = sent.cuda()
        property = property.cuda()
        pred = model(sent, property).unsqueeze(0)
        if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][
            1] and label == 1:  # 0 for bot and 1 for human
            acc = acc + 1
        if pred[0][0] > pred[0][1] and label == 0:
            TP = TP + 1
        if pred[0][0] > pred[0][1] and label == 1:
            FP = FP + 1
        if pred[0][0] < pred[0][1] and label == 1:
            TN = TN + 1
        if pred[0][0] < pred[0][1] and label == 0:
            FN = FN + 1
        model.zero_grad()

        loss = loss_function(pred, label)
        avg_loss += loss.item()
    avg_loss = avg_loss / cnt
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = acc / float(cnt)
    specificity = TN / (TN + FP)
    F1 = TP / (TP + 0.5 * (FP + FN))
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if ii == 1:
        print('val: ')
    else:
        print('test: ')
    print('loss: ' + str(avg_loss))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('acc: ' + str(acc))
    print('specificity: ' + str(specificity))
    print('F1: ' + str(F1))
    print('MCC: ' + str(MCC))

    return acc


embed_mat = torch.load('./embedding.pt')
train()
