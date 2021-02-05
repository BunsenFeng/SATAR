from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import argparse

B_BID = 267040
class TweetDataset(Dataset):
    def __init__(self, file):
        self.IDList = []
        with open(file, 'r', encoding = 'utf-8') as f:
            for account in f:
                self.IDList.append(account.split()[0])
    def __len__(self):
        return len(self.IDList)
    def __getitem__(self, index):
        ID = self.IDList[index]
        data = torch.load('accountTensor/' + ID + '.pt')
        if data['tweet'].size() == torch.Size([1, 0]):
            print(ID)
        return data
class TweetLSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers, drop_prob, batch_size):
        super(TweetLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size
        
        self.embeddinglayer = nn.Embedding(fullEmbeds.size(0), 300)
        self.embeddinglayer.weight.data = fullEmbeds
        self.embeddinglayer.weight.requires_grad = False
        
        self.lstm = nn.LSTM(300, hidden_dim, n_layers,
                           dropout = drop_prob, batch_first = True,
                           bidirectional = True)
        
        self.fc = nn.Linear(2 * hidden_dim, 2)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).cuda()),
               autograd.Variable(torch.zeros(2 * self.n_layers, self.batch_size, self.hidden_dim).cuda()))
    def forward(self, sentence):
        self.batch_size = len(sentence.data)
        self.hidden = self.init_hidden()
        embeds = self.embeddinglayer(sentence)
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        y = self.fc(lstm_out)
        log_probs = F.log_softmax(y, dim = 2)
        log_probs = log_probs[:, -1]
        out = (log_probs.sum(0) / len(sentence.data)).unsqueeze(0)
        return out
def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
            right += 1.0
    return right/len(truth)
def train():
    device_ids = [0]
    EPOCH = 30
    dropout = 0.5
    model = TweetLSTM(hidden_dim = HIDDEN_DIM, n_layers = N_LAYERS,
                     drop_prob = dropout, batch_size = BATCH_SIZE)
    model = nn.DataParallel(model, device_ids = device_ids)
    model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.module.parameters(), lr = 0.01, betas = (0.9, 0.99))
    best_dev_acc = 0.0
    for i in range(EPOCH):
        dropout = 0.5 - i * (0.4) / EPOCH
        model.module.lstm.dropout = dropout
        print('epoch: %d start!' % i)
        train_epoch(model, trainDataset, loss_function, optimizer, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model, devDataset, loss_function, optimizer, i, 'dev')
        test_acc = evaluate(model, testDataset, loss_function, optimizer, i, 'test')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm best_models/mr_best_model_minibatch_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.module.state_dict(), 'best_models/mr_best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
def Metric(truth, pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(truth)):
        if truth[i] == 1 and pred[i] == 1:
            TP += 1
        elif truth[i] == 0 and pred[i] == 1:
            FP += 1
        elif truth[i] == 0 and pred[i] == 0:
            TN += 1
        else:
            FN += 1
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
def evaluate(model, dataset, loss_function, optimizer, i, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    tweetcount = 0
    for user in dataset:
        sent, label = user['tweet'], user['label']
        tweetcount += len(sent)
        label = torch.tensor([int(label)])
        sent = sent.cuda()
        label = label.cuda()
        truth_res += list(label.data)
        pred = model(sent)
        pred_label = pred.cpu().data.max(1)[1].numpy()
        pred = pred.cuda()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.item()    
    avg_loss /= tweetcount
    acc = get_accuracy(truth_res, pred_res)
    Metric(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc
def train_epoch(model, dataset, loss_function, optimizer, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    tweetcount = 0
    tmp = 0
    for user in dataset:
        sent, label = user['tweet'], user['label']
        tweetcount += len(sent)
        label = torch.tensor([int(label)])
        sent = sent.cuda()
        label = label.cuda()
        truth_res += list(label.data)
        pred = model(sent)
        pred_label = pred.cpu().data.max(1)[1].numpy()
        pred = pred.cuda()
        pred_res += [x for x in pred_label]
        if tmp == 0:
            pred_tot = pred
            label_tot = label
            tmp += 1
            continue
        elif tmp == 16:
            tmp = 0
        else:
            pred_tot = torch.cat((pred_tot, pred), 0)
            label_tot = torch.cat((label_tot, label), 0)
            tmp += 1
            continue
        loss = loss_function(pred_tot, label_tot)
        avg_loss += loss.item() 
        count += 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count * 16, loss.item()))
        model.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= tweetcount
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))
        
if __name__ == '__main__':
    HIDDEN_DIM = 200
    N_LAYERS = 3
    BATCH_SIZE = 64
    
    trainDataset = TweetDataset('listTrain.txt')
    devDataset = TweetDataset('listDev.txt')
    testDataset = TweetDataset('listTest.txt')
    print('load done')
    fullEmbeds = torch.load('embedding.pt')
    train()
