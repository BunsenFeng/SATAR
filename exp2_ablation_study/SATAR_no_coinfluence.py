import torch
import numpy
import math
import pandas as pd
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.distributed as dist
import time


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~model subclass~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Attention Mechanism in a nutshell
# Input : impending vectors of size vector_num * vector_size
# Output : Attentioned representation of size vector_size
class Attention(nn.Module):

    def __init__(self, vector_size):
        super(Attention, self).__init__()
        self.vector_size = vector_size

        self.fc = nn.Linear(vector_size, vector_size)
        # self.fc.bias.data.fill_(0)
        self.weightparam = nn.Parameter(torch.randn(vector_size, 1))

    def forward(self, vectors):
        # hidden = torch.tanh(self.fc(vectors))
        # print(hidden)
        weight = torch.tanh(self.fc(vectors)).matmul(self.weightparam)
        # del hidden
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        weight = F.softmax(weight, dim=0)
        # print(weight)
        rep = vectors.mul(weight)
        # del weight
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rep = rep.sum(dim=0)
        return rep


# Model for word-level semantics extraction (2 initializations: 1 WordLevel(batch_size of 1) 1 TweetLevelLow)
# Input : sequence of size seq_len * batch_size (maybe a result of rnn.pad_sequence?)
# Output : seq_len * batch_size * rep_size
class SemanticWord(nn.Module):

    def __init__(self, embedding_dim, rep_size, batch_size, num_layer, embed_layer, p):
        super(SemanticWord, self).__init__()
        self.hidden_dim = int(rep_size / 2)
        self.embedding_dim = embedding_dim
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.num_layer = num_layer

        self.word_embeddings = embed_layer
        self.word_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, dropout=p, num_layers=num_layer)
        # self.hidden = self.init_hidden()
        # self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(non_blocking=True), self.hidden[1].type(torch.FloatTensor).cuda(non_blocking=True))
        # self.register_buffer('hidden', self.hidden)

    def init_hidden(self, batch_size):
        # torch.cuda.FloatTensor(1000, 1000).fill_(0)
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0))
        return temp
        # return (temp[0].type(torch.FloatTensor).cuda(non_blocking=True), temp[1].type(torch.FloatTensor).cuda(non_blocking=True))

    def forward(self, text):
        sim_batch_size = 8
        batch_size = len(text[0])
        if batch_size <= sim_batch_size:
            self.hidden = self.init_hidden(batch_size)
            # text = text[0:text.detach().tolist().index(tdict['b_b'])]
            tmp = {i for i in range(len(text)) if text[i].item() == tdict['b_b']}
            tmp = list(set(range(len(text))) - tmp)
            if not len(tmp):
                tmp = [0]
            text = text[tmp]
            result = self.word_embeddings(text)
            result = result.clone().view(len(text), batch_size, -1).cuda(non_blocking=True)
            result, _ = self.lstm(result, self.hidden)
            del self.hidden
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return result
        else:
            now = 0
            tmp = []
            # print('batch_size: ' + str(batch_size))
            while True:
                now_text = text[:, now:min(now + sim_batch_size, batch_size)]
                now_batch_size = len(now_text[0])
                # print('now batch size: ' + str(now_batch_size))
                self.hidden = self.init_hidden(now_batch_size)
                result = self.word_embeddings(now_text)
                result = result.clone().view(len(now_text), now_batch_size, -1).cuda(non_blocking=True)
                result, _ = self.lstm(result, self.hidden)
                # del self.hidden
                del now_text
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tmp.append(result)
                if now + sim_batch_size >= batch_size:
                    break
                now += sim_batch_size
            tmp = torch.cat(tmp, dim=1)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print('before attention: ', tmp.size())
            return tmp

        # batch_size = len(text[0])
        # self.hidden = self.init_hidden(batch_size)
        # text = text[0:text.detach().tolist().index(tdict['b_b'])]
        # if batch_size == 1:
        # tmp = {i for i in range(len(text)) if text[i].item() == tdict['b_b']}
        # tmp = list(set(range(len(text))) - tmp)
        # if not len(tmp):
        # tmp = [0]
        # text = text[tmp]
        # result = self.word_embeddings(text)
        # result = result.clone().view(len(text), batch_size, -1).cuda(non_blocking=True)
        # result, _ = self.lstm(result, self.hidden)
        # del self.hidden
        # return result


# Model for tweet-level semantics extraction from tweet vectors
# Input : sequence of tweet vectors of a single user of size vector_num * 1 * tweet_vec_size
# Output : vector_num * rep_size
class SemanticTweet(nn.Module):

    def __init__(self, input_dim, rep_size, num_layer, p):
        super(SemanticTweet, self).__init__()
        self.hidden_dim = int(rep_size / 2)
        self.input_dim = input_dim
        self.rep_size = rep_size
        self.batch_size = 1
        self.num_layer = num_layer

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True, dropout=p, num_layers=num_layer)
        self.hidden = self.init_hidden()
        # self.hidden = (self.hidden[0].type(torch.FloatTensor).cuda(non_blocking=True), self.hidden[1].type(torch.FloatTensor).cuda(non_blocking=True))
        # self.register_buffer('hidden', self.hidden)

    def init_hidden(self):
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0))
        return temp
        # return (temp[0].type(torch.FloatTensor).cuda(non_blocking=True), temp[1].type(torch.FloatTensor).cuda(non_blocking=True))

    def forward(self, vectors):
        self.hidden = self.init_hidden()
        # result = vectors.clone().view(len(vectors), self.batch_size, -1)
        # vectors = vectors.cuda(non_blocking = True)
        result, _ = self.lstm(vectors, self.hidden)
        result = result.squeeze(1)
        # del self.hidden
        return result


# Aggregated semantic model
# Input : user dict {'word' : torch.tensor([1,2,3,...]), 'tweet': tensor of size tweet_cnt * tweet_len}
# Output : overall semantic representation of size rep_size
class SemanticVector(nn.Module):

    def __init__(self, embedding_dim, rep_size, num_layer, dropout, embed_layer):
        super(SemanticVector, self).__init__()
        self.embedding_dim = embedding_dim
        self.rep_size = rep_size
        self.num_layer = num_layer
        self.dropout = dropout
        # self.embed_layer = embed_layer
        # self.embed_layer.weight.requires_grad = False

        self.WordLevelModel = SemanticWord(embedding_dim=self.embedding_dim, rep_size=int(self.rep_size / 2),
                                           batch_size=1,
                                           num_layer=self.num_layer, embed_layer=embed_layer, p=self.dropout)
        self.TweetLowModel = SemanticWord(embedding_dim=self.embedding_dim, rep_size=int(self.rep_size / 2),
                                          batch_size=1,
                                          num_layer=self.num_layer, embed_layer=embed_layer, p=self.dropout)
        self.TweetHighModel = SemanticTweet(input_dim=int(self.rep_size / 2), rep_size=int(self.rep_size / 2),
                                            num_layer=self.num_layer, p=dropout)
        self.WordAttention = Attention(vector_size=int(self.rep_size / 2))
        self.TweetLowAttention = Attention(vector_size=int(self.rep_size / 2))
        self.TweetHighAttention = Attention(vector_size=int(self.rep_size / 2))

    def forward(self, user):
        text_word = user['word']
        # text_word = text_word.unsqueeze(1)
        WordLevelRep = self.WordAttention(self.WordLevelModel(text_word.unsqueeze(1)).squeeze(1))
        del text_word
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        text_tweet = user['tweet']  # one tweet each row
        # TweetRep = []
        # for i in range(len(text_tweet)):
        #    TweetRep.append(
        #        self.TweetLowAttention(self.TweetLowModel(text_tweet[i, :].unsqueeze(1)).squeeze(1)).tolist())
        TweetRep = self.TweetLowAttention(self.TweetLowModel(text_tweet.transpose(0, 1)))
        del text_tweet
        # print('user tweet low finish')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # vec_tweet = torch.tensor(TweetRep).unsqueeze(1)
        TweetLevelRep = self.TweetHighAttention(self.TweetHighModel(TweetRep.unsqueeze(1)))
        # del TweetRep
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # print('a user semantic finish')
        return torch.cat((WordLevelRep, TweetLevelRep))


# Model for transforming the properties vector
# Input : property vectors of size vector_num * input_size
# Output : representations of size vector_num * rep_size
class Properties(nn.Module):

    def __init__(self, input_size, rep_size, dropout):
        super(Properties, self).__init__()
        self.input_size = input_size
        self.rep_size = rep_size

        # self.fc1 = nn.Linear(self.input_size, self.input_size)
        # self.fc2 = nn.Linear(self.input_size, self.rep_size)
        # self.fc3 = nn.Linear(self.rep_size, self.rep_size)
        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # self.fc3.bias.data.fill_(0)
        # self.act1 = nn.ReLU()
        # self.act2 = nn.ReLU()
        # self.act3 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout)
        # self.dropout2 = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.input_size, self.rep_size)
        self.act = nn.ReLU()

    def forward(self, vectors):
        # vectors = self.dropout1(self.act1(self.fc1(vectors)))
        # vectors = self.dropout2(self.act2(self.fc2(vectors)))
        # vectors = self.act3(self.fc3(vectors))
        vectors = self.act(self.fc(vectors))
        return vectors


# Co attention model
# Input : user {'semantic' : batch_size * 1 * vec_size, 'property' : batch_size * 1 * vec_size, 'neighbor' : batch_size * 1 * (2 * vec_size)}
# Output : final representation of size batch_size * 1 * vec_size
class CoAttention(nn.Module):

    def __init__(self, vec_size):
        super(CoAttention, self).__init__()
        self.vec_size = vec_size
        #self.Wsp = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        #self.Wpn = nn.Parameter(torch.randn(self.vec_size, 2 * self.vec_size))
        #self.Wns = nn.Parameter(torch.randn(2 * self.vec_size, self.vec_size))
        #self.Ws = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        #self.Wp = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        #self.Wn = nn.Parameter(torch.randn(self.vec_size, 2 * self.vec_size))
        #self.Wh = nn.Parameter(torch.randn(3 * self.vec_size, self.vec_size))
        self.fc = nn.Linear(2 * vec_size, 2 * vec_size)
        self.act = nn.ReLU()

    def forward(self, user):
        # Vs Vp Vn tensor of size vec_size x 1
        Vs = torch.transpose(user['semantic'], 1, 2)
        Vp = torch.transpose(user['property'], 1, 2)
        Vn = torch.transpose(self.act(self.fc(user['neighbor'])), 1, 2)
        
        #print('Vs size ', Vs.size())
        #print('Vp size ', Vp.size())
        #print('Vn size ', Vn.size())
        #print('Wsp size ', Wsp.size())
        #Fsp = torch.tanh(torch.transpose(Vs, 1, 2).matmul(self.Wsp).matmul(Vp))
        #Fpn = torch.tanh(torch.transpose(Vp, 1, 2).matmul(self.Wpn).matmul(Vn))
        #Fns = torch.tanh(torch.transpose(Vn, 1, 2).matmul(self.Wns).matmul(Vs))

        #Hs = torch.tanh(self.Ws.matmul(Vs) + self.Wp.matmul(Vp) * Fsp + self.Wn.matmul(Vn) * Fns)
        #Hp = torch.tanh(self.Wp.matmul(Vp) + self.Ws.matmul(Vs) * Fsp + self.Wn.matmul(Vn) * Fpn)
        #Hn = torch.tanh(self.Wn.matmul(Vn) + self.Ws.matmul(Vs) * Fns + self.Wp.matmul(Vp) * Fpn)

        # V = torch.cat((torch.cat((Vs, Vp)), Vn))
        V = torch.cat((torch.cat((Vs, Vp), dim = 1), Vn), dim = 1)
        # rep = torch.cat((V, H))
        result = torch.tanh(torch.transpose(V, 1, 2))
        #del Vs,Vp,Vn,Fsp,Fpn,Fns,Hs,Hp,Hn,H
        #print('result size ', result.size())
        return result


# Model to predict follower from vector representation
# Input : user representation of size vector_num * vec_size
# Output : log softmax of size vector_num * label_size
class FollowerClassification(nn.Module):

    def __init__(self, vec_size, label_size, dropout):
        super(FollowerClassification, self).__init__()
        self.vec_size = vec_size
        self.label_size = label_size

        # self.fc1 = nn.Linear(self.vec_size, self.vec_size)
        self.fc2 = nn.Linear(4 * self.vec_size, self.label_size)
        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # self.act1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, vector):
        # result = self.dropout1(self.act1(self.fc1(vector)))
        # print(result.size())
        result = F.log_softmax(self.fc2(vector), dim=1)
        return result


# Model to predict a batch of users' followers
# Input : a batch of users via DataLoader, user information specified in forward() annotation
# Output : a batch of users' follower classification(log_softmax) of size batch_size * label_size ;
#          semantic repersentation of list size batch_size * rep_size
#          property representation of tensor size batch_size * rep_size
class ModelBatch(nn.Module):

    def __init__(self, EMBEDDING_DIM, REP_SIZE, NUM_LAYER, DROPOUT, EMBED_LAYER, PROPERTY_SIZE, LABEL_SIZE):
        super(ModelBatch, self).__init__()

        self.SemanticModel = SemanticVector(embedding_dim=EMBEDDING_DIM, rep_size=REP_SIZE, num_layer=NUM_LAYER,
                                            dropout=DROPOUT, embed_layer=EMBED_LAYER)
        self.PropertyModel = Properties(input_size=PROPERTY_SIZE, dropout=DROPOUT, rep_size=REP_SIZE)
        self.CoAttentionModel = CoAttention(vec_size=REP_SIZE)
        self.FollowerPredictModel = FollowerClassification(vec_size=REP_SIZE, label_size=LABEL_SIZE, dropout=DROPOUT)

    def forward(self, user_batch):
        # each user shall contain
        # 'word' : torch.tensor([1,2,3,..])
        # 'tweet' : tensor of size tweet_cnt * tweet_len
        # 'property' : tensor of size 1 * PROPERTY_SIZE
        # 'neighbor' : tensor of size 1 * REP_SIZE

        # semantic vector extraction
        semantic_reps = []
        for i in range(len(user_batch['word'])):
            semantic_reps.append(self.SemanticModel({'word': user_batch['word'][i], 'tweet': user_batch['tweet'][i]}))
        # print('semantic finish')

        # property vector extraction
        property_reps = self.PropertyModel(user_batch['property'])

        # vector representation
        # vector_reps = []
        # for i in range(len(user_batch['word'])):
        #    vector_reps.append(self.CoAttentionModel({'semantic': semantic_reps[i].unsqueeze(0),
        #                                              'property': property_reps[i],
        #                                              'neighbor': user_batch['neighbor'][i]}).squeeze(0).tolist())
        vector_reps = self.CoAttentionModel({'semantic': torch.stack(semantic_reps).unsqueeze(1),
                                             'property': property_reps,
                                             'neighbor': user_batch['neighbor']}).squeeze(1)

        # follower prediction
        # vector_reps = torch.tensor(vector_reps).cuda(non_blocking=True)
        result = self.FollowerPredictModel(vector_reps)
        # del vector_reps
        return result, semantic_reps, property_reps, vector_reps


# Dataset Class(code by whr)
class TweetDataset(Dataset):
    def __init__(self, file):
        self.file = file.replace('.txt', '')
        self.IDList = []
        with open(file, 'r', encoding='utf-8') as f:
            for ID in f:
                self.IDList.append(ID.split()[0])
        #zeros = torch.zeros([1,600], dtype = torch.float)
        #for ID in self.IDList:
            #path = 'neighbor_tensor/' + ID + '.pt'
            #torch.save(zeros, path)

    def __len__(self):
        return len(self.IDList)

    def __getitem__(self, index):
        ID = self.IDList[index]
        path = 'tensor/' + ID + '.pt'
        user = torch.load(path)
        path = 'neighbor_tensor/' + ID + 'PartData.pt'
        user['neighbor'] = torch.load(path)
        return user

    def check(self, ID):
        return ID in self.IDList

    def update(self, ID, val):
        path = 'neighbor_tensor/' + ID + self.file + '.pt'
        torch.save(val, path)


# SATAR + FC + fine tune for botection
class SATARbot(nn.Module):
    def __init__(self, EMBEDDING_DIM, REP_SIZE, NUM_LAYER, DROPOUT, EMBED_LAYER, PROPERTY_SIZE, LABEL_SIZE):
        super(SATARbot, self).__init__()

        self.SATAR = ModelBatch(EMBEDDING_DIM = EMBEDDING_DIM, REP_SIZE = REP_SIZE, NUM_LAYER = NUM_LAYER, DROPOUT = DROPOUT,
                                EMBED_LAYER = EMBED_LAYER, PROPERTY_SIZE = PROPERTY_SIZE, LABEL_SIZE = LABEL_SIZE)

        self.fc = nn.Linear(REP_SIZE * 4, 2)

    def forward(self, user_batch):
        _, _, _, reps = self.SATAR(user_batch)
        result = F.log_softmax(self.fc(reps), dim=1)
        return result


# padding for each batch in DataLoader
def pad_collate(batch):
    final = {}
    tmp = []
    for user in batch:
        tmp.append(user['word'])
    tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    # final['word'] = tmp.cuda(non_blocking=True)
    final['word'] = tmp
    tmp = []
    for user in batch:
        tmp.append(user['id'])
    # tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    # final['id'] = torch.stack(tmp).cuda(non_blocking=True)
    final['id'] = torch.stack(tmp)
    tmp = []
    for user in batch:
        tmp.append(user['target'])
    # tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    # final['target'] = torch.stack(tmp).cuda(non_blocking=True)
    final['target'] = torch.stack(tmp)

    tmp = []
    for user in batch:
        tmp.append(user['neighbor'])
    # tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    # final['neighbor'] = torch.stack(tmp).cuda(non_blocking=True)
    final['neighbor'] = torch.stack(tmp)

    tmp = []
    for user in batch:
        tmp.append(user['property'])
    # tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    # final['property'] = torch.stack(tmp).cuda(non_blocking=True)
    final['property'] = torch.stack(tmp)
    mxH = 0
    mxL = 0
    for user in batch:
        mxH = max(mxH, user['tweet'].size()[0])
        mxL = max(mxL, user['tweet'].size()[1])
    empty = [tdict['b_b']] * mxL
    tmp = []
    for user in batch:
        T = []
        tweet = user['tweet'].numpy().tolist()
        for i in tweet:
            i = i + [tdict['b_b']] * (mxL - len(i))
            T.append(i)
        for i in range(mxH - len(tweet)):
            T.append(empty)
        tmp.append(T)
    # final['tweet'] = torch.tensor(tmp).cuda(non_blocking=True)
    final['tweet'] = torch.tensor(tmp)
    return final


# The main function of training
def train():
    # loading word embedding (to be modified)
    VOCAB_SIZE = len(embed_mat)
    EMBEDDING_DIM = len(embed_mat[0])
    embed_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    embed_layer.weight.data = embed_mat
    embed_layer.weight.requires_grad = False

    # DataLoader intializations (to be modified)
    #print('Loading Full Dataset')
    # data = TweetDataset()
    # broad_data = TweetDataset()

    # TrainDataLoader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
    # DevDataLoader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
    # TestDataLoader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
    # BigDataLoader = DataLoader(broad_data, batch_size=32, shuffle=True, num_workers=0)

    #FullData = TweetDataset('FullData.txt')
    print('Loading Part Dataset')
    #PartData = TweetDataset('cresci_IDList.txt')
    #print('A total of ' + str(len(PartData)) + ' users are used for training')
    #train_set, validate_set, test_set = torch.utils.data.random_split(PartData, [int(len(PartData) * 0.8),
    #                                                                             int(len(PartData) * 0.1),
    #                                                                             len(PartData) - int(
    #                                                                                 len(PartData) * 0.8) - int(
    #                                                                                 len(PartData) * 0.1)])
    train_set = TweetDataset('listTrain.txt')
    validate_set = TweetDataset('listDev.txt')
    test_set = TweetDataset('listTest.txt')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(validate_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    #full_sampler = torch.utils.data.distributed.DistributedSampler(FullData)

    # print('TrainDataLoader')
    TrainDataLoader = DataLoader(train_set, batch_size=12, num_workers=0, collate_fn=pad_collate,
                                 pin_memory=True, sampler=train_sampler)
    # print('DevDataLoader')
    DevDataLoader = DataLoader(validate_set, batch_size=12, num_workers=0, collate_fn=pad_collate,
                               pin_memory=True, sampler=dev_sampler)
    # print('TestDataLoader')
    TestDataLoader = DataLoader(test_set, batch_size=12, num_workers=0, collate_fn=pad_collate,
                                pin_memory=True, sampler=test_sampler)
    # print('BigDataLoader')
    #BigDataLoader = DataLoader(FullData, batch_size=28, num_workers=0, collate_fn=pad_collate,
    #                           pin_memory=True, sampler=full_sampler)

    # Model Declaration
    print('Declaring Model')
    REP_SIZE = 300
    DROPOUT = 0.6
    NUM_LAYER = 1
    PROPERTY_SIZE = 29
    LABEL_SIZE = 2

    model = SATARbot(EMBEDDING_DIM=EMBEDDING_DIM, REP_SIZE=REP_SIZE, DROPOUT=DROPOUT, EMBED_LAYER=embed_layer,
                       NUM_LAYER=NUM_LAYER, PROPERTY_SIZE=PROPERTY_SIZE, LABEL_SIZE=LABEL_SIZE)
    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=1e-2)
    

    # param_list(model)
    #print('There is a total of ' + str(count_parameters(model)) + ' parameters!')

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.cuda()
    #model.SATAR.load_state_dict(torch.load('best_post_model.model'))
    if torch.cuda.device_count() > 1:
        print('Lets use', torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model.module.SATAR.load_state_dict(torch.load('best_post_model.model'), strict = False)
    

    # print('moving model to GPU!!!')
    # model = model.to('cuda')
    # print('model moved to GPU!!!')

    # Training
    print('training starts now')
    best_dev_acc = 0
    no_up = 0
    EPOCH = 300
    warmup_epoch = 2
    WEIGHT_DECAY = 0
    smaller_step_epoch = -1
    optimizer = optim.SGD([param for param in model.module.fc.parameters()], lr=1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    for i in range(EPOCH):
        print('epoch: ' + str(i) + ' start at ' + time.ctime(time.time()))
        train_epoch(model, TrainDataLoader, train_set, loss_function, optimizer, i)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('epoch: ' + str(i) + ' train end at ' + time.ctime(time.time()))
        print('now best dev acc:', best_dev_acc)
        dev_acc = evaluate(model, DevDataLoader, validate_set, loss_function, 'dev')
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print('epoch: ' + str(i) + ' dev end at ' + time.ctime(time.time()))
        test_acc = evaluate(model, TestDataLoader, test_set, loss_function, 'test')
        print('epoch: ' + str(i) + ' test end at ' + time.ctime(time.time()))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('New Best Dev at epoch ' + str(i))
            #torch.save(model.state_dict(), 'best_models/SATARtune_epoch' + str(i) + '_acc_' + str(test_acc) + '.model')
            #no_up = 0
        # warmup ends and unfreezes SATAR after warmup_epoch epochs
        
        if args.local_rank == 0:
            torch.save(model.state_dict(), 'best_models/SATARtune_epoch' + str(i) + '_acc_' + str(test_acc) + '.model')
        
        if i == warmup_epoch:
            optimizer = optim.SGD([
                {'params': model.module.fc.parameters()},
                {'params': model.module.SATAR.parameters(), 'lr': 1e-2}
            ], lr=1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
            
        if i == smaller_step_epoch:
            optimizer = optim.SGD([
                {'params': model.module.fc.parameters()},
                {'params': model.module.SATAR.parameters(), 'lr': 1e-3}
            ], lr=1e-3, momentum = 0.9, weight_decay = WEIGHT_DECAY)
        # else:
        # no_up += 1
        # if no_up >= 10:
        # exit()
        # print('epoch: ' + str(i) + ' dev end at ' + time.ctime(time.time()))
        # print('Updating Vs Vp')
        # info_update(model, BigDataLoader)
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # print('epoch: ' + str(i) + ' info update end at ' + time.ctime(time.time()))
        # print('Updating Vn')
        # neighbor_update(model, TrainDataLoader, train_set, REP_SIZE)
        # neighbor_update(model, DevDataLoader, validate_set, REP_SIZE)
        # neighbor_update(model, TestDataLoader, test_set, REP_SIZE)
        # print('epoch: ' + str(i) + ' neighbor update end at ' + time.ctime(time.time()))
        # print(userinfo['2345678'])


# Return the classification accuracy on the given dataset
def evaluate(model, eval_set, eval_data, loss_function, name):
    model.eval()
    avg_loss = 0
    correct = 0
    total_user = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i_batch, data in enumerate(eval_set):
            sample_batch = {}
            for elem in data.items():
                sample_batch[elem[0]] = elem[1].cuda(non_blocking=True)
            pred = model(sample_batch)
            # del trash1, trash2
            # print('batch ' + str(i_batch))
            # print(pred.size())
            # print(sample_batch['target'].size())
            for i in range(len(sample_batch['id'])):
                total_user += 1
                pred_label = pred[i].detach().cpu().numpy().argmax()
                true_label = sample_batch['target'][i].item()
                if pred_label == true_label:
                    correct += 1
                    if true_label == 1:
                        TP += 1
                    if true_label == 0:
                        TN += 1
                else:
                    if true_label == 1:
                        FN += 1
                    if true_label == 0:
                        FP += 1
                
            avg_loss = avg_loss + loss_function(pred, sample_batch['target'].squeeze(1)).item()
            # del pred
            del sample_batch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        avg_loss = avg_loss / len(eval_set)
        acc = correct / total_user
        # acc = correct
        #print(name + ' avg_loss:%g, acc:%g, TP %g TN %g FP %g FN %g' % (avg_loss, acc, TP, TN, FP, FN))
        #f.write(name + ' avg_loss:%g, acc:%g, TP %g TN %g FP %g FN %g' % (avg_loss, acc, TP, TN, FP, FN))
        f=open('log.txt', 'a')
        f.write(name + ' avg_loss: '+str(avg_loss)+', acc: '+str(acc)+', TP '+str(TP)+' TN '+str(TN)+' FP '+str(FP)+' FN '+str(FN) + '\n')
        f.close()
        return acc


# Training of a single epoch
def train_epoch(model, train_set, train_data, loss_function, optimizer, i_th):
    model.train()
    avg_loss = 0.0
    correct = 0
    total_user = 0
    count = 0
    count_report = int(len(train_set) / 5)
    for i_batch, data in enumerate(train_set):
        sample_batch = {}
        for elem in data.items():
            sample_batch[elem[0]] = elem[1].cuda(non_blocking=True)
        pred = model(sample_batch)
        # del trash1, trash2
        # print('batch ' + str(i_batch))
        for i in range(len(sample_batch['id'])):
            total_user += 1
            if pred[i].detach().cpu().numpy().argmax() == sample_batch['target'][i].item():
                correct += 1
        model.zero_grad()
        loss = loss_function(pred, sample_batch['target'].squeeze(1))
        avg_loss = avg_loss + loss.item()
        count = count + 1
        #if count % count_report == 0:
            #print('epoch: %d batch: %d loss :%g' % (i_th, count, loss.item()))
        loss = loss + 0 * sum(p.sum() for p in model.parameters())
        loss.backward(retain_graph=False)
        # print('loss backward!!!')
        optimizer.step()
        # del pred, loss
        del sample_batch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    avg_loss = avg_loss / len(train_set)
    acc = correct / total_user
    # acc = correct
    #print('epoch: %d done!\ntrain avg_loss:%g, acc:%g' % (i_th, avg_loss, acc))
    f=open('log.txt', 'a')
    f.write('epoch: '+str(i_th)+' done!\ntrain avg_loss:'+str(avg_loss)+', acc:'+str(acc) + '\n')
    f.close()


# calcualte the total amount of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


f = open('log.txt', 'w')
f.write(str(time.ctime(time.time())) + '\n')
f.close()
os.system('export CUDA_VISIBLE_DEVICES=0,1,2,3')
device = 0
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# os.system('export CUDA_VISIBLE_DEVICES=0,1,2,3')

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)
#print(args.local_rank)

#print('Loading Word Embedding')
# tdict, embed_mat = loadDict('embed_file.txt')
# torch.save(embed_mat, 'embedding.pt')
tdict = {}
tdict['b_b'] = 267040  # b_b token id
user_info = {}
embed_mat = torch.load('embedding.pt')
train()