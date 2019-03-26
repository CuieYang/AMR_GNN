#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for concept identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from modules.helper_module import data_dropout
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
import numpy as np
from torch.autograd import Variable

class SentenceEncoder(nn.Module):
    def __init__(self, opt, embs):
        self.layers = opt.txt_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.txt_rnn_size // self.num_directions
    #    inputSize = opt.word_dim*2 + opt.lemma_dim + opt.pos_dim +opt.ner_dim
        inputSize = embs["word_fix_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim\
                    +embs["pos_lut"].embedding_dim + embs["ner_lut"].embedding_dim

        super(SentenceEncoder, self).__init__()
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)


        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]

        self.pos_lut = embs["pos_lut"]

        self.ner_lut = embs["ner_lut"]

        self.drop_emb = nn.Dropout(opt.dropout)
        self.alpha = opt.alpha

        if opt.cuda:
            self.rnn.cuda()

    def forward(self, packed_input: PackedSequence,hidden=None):
    #input: pack(data x n_feature ,batch_size)
        input = packed_input.data
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)

        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])


        emb = self.drop_emb(torch.cat([lemma_emb,pos_emb,ner_emb],1))#  data,embed
        emb = torch.cat([word_fix_embed,emb],1)#  data,embed
        emb =  PackedSequence(emb, packed_input.batch_sizes)
        outputs, hidden_t = self.rnn(emb, hidden)

        return  outputs

class Node_emblayer(nn.Module):  # 继承nn.Module  yang 20190205
    def __init__(self,opt,in_features,out_features):
        super(Node_emblayer, self).__init__()  # 等价于nn.Module.__init__(self)

        self.inputfeature = in_features
        self.outfeature = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.bc = nn.Linear(in_features, out_features)
        self.edge = nn.Linear(3*out_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        # self.transform = nn.Sequential(
        #     nn.Dropout(opt.dropout),
        #     nn.Linear(self.outfeature,self.outfeature,bias = self.outfeature))
        self.K = opt.Node_K

    def forward(self, fw_input,bw_input, edge_embs, fw_adj, bw_adj,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep):

        fw_adjmask = fw_adj.ge(0)
        bw_adjmask = bw_adj.ge(0)
        fw_edgmask = fw_edgeid.ge(0)
        bw_edgmask = bw_edgeid.ge(0)
        fw_edgdepmask = fw_edgedep.ge(0)
        bw_edgdepmask = bw_edgedep.ge(0)
        edgout = torch.masked_select(fw_edgedep[:, 0], fw_edgdepmask[:, 0])

        for k in range(self.K):
            alpha = 1-0.1*k
            fw_hidden = Variable(torch.zeros(len(bw_input), self.inputfeature))
            bw_hidden = Variable(torch.zeros(len(bw_input), self.inputfeature))

            if self.cuda:
                fw_hidden = fw_hidden.cuda()
                bw_hidden = bw_hidden.cuda()

            for i in range(len(fw_input)):
                indices = Variable(torch.masked_select(fw_adj[i,:], fw_adjmask[i,:]).cuda())
                # indices = Variable(fw_adj[i,0:fw_end[i]].cuda())
                adj_x = torch.index_select(fw_input, 0, indices)

                out = torch.masked_select(fw_edgeid[i,:], fw_edgmask[i,:])
                if len(out)>0:
                    # print("out",out)
                    indices = Variable(out.cuda())
                    adj_x1 = torch.index_select(edge_embs, 0, indices)
                    adj_x = torch.cat((adj_x1, adj_x), 0)
                adj_x = torch.mean(adj_x, dim=0)
                # if k < self.K - 1:
                #     adj_x = torch.mean(adj_x, dim=0)
                # else:
                #     adj_x = self.transform(adj_x)
                #     adj_x = self.relu(adj_x)
                #     adj_x = torch.max(adj_x, dim=0)[0]  # torch.mean(adj_x, dim=0)
                fw_hidden[i,:] = torch.cat((fw_input[i, :], adj_x), -1).view(1, -1)

                indices = Variable(torch.masked_select(bw_adj[i,:], bw_adjmask[i,:]).cuda())
                adj_x = torch.index_select(bw_input, 0, indices)

                out = torch.masked_select(bw_edgeid[i, :], bw_edgmask[i, :])
                if len(out)>0:
                    indices = Variable(out.cuda())
                    adj_x1 = torch.index_select(edge_embs, 0, indices)
                    adj_x = torch.cat((adj_x1, adj_x), 0)
                adj_x = torch.mean(adj_x, dim=0)
                # if k < self.K - 1:
                #     adj_x = torch.mean(adj_x, dim=0)
                # else:
                #     adj_x = self.transform(adj_x)
                #     adj_x = self.relu(adj_x)
                #     adj_x = torch.max(adj_x, dim=0)[0]  # torch.mean(adj_x, dim=0)
                bw_hidden[i,:] = torch.cat((bw_input[i, :], adj_x), -1).view(1, -1)


            fw_hidden = self.fc(fw_hidden)
            bw_hidden = self.bc(bw_hidden)

            if k < self.K - 1:
                bw_hidden = self.relu(bw_hidden)
                fw_hidden = self.relu(fw_hidden)

            fw_input = fw_hidden
            bw_input = bw_hidden


            if len(edgout)>0:
                edge_hidden = Variable(torch.zeros(len(edgout), 3*self.outfeature))
                if self.cuda:
                    edge_hidden = edge_hidden.cuda()

                kk = 0
                for i in range(len(bw_edgedep)):
                    edge_dep = edge_embs[i,:].clone()

                    out = torch.masked_select(fw_edgedep[i, :], fw_edgdepmask[i, :])
                    if len(out)>0:

                        indices = Variable(out.cuda())
                        fadj_x = torch.index_select(fw_input, 0, indices)
                        fadj_x = torch.mean(fadj_x, dim=0)
                        # if k<self.K-1:
                        #     fadj_x = torch.mean(fadj_x, dim=0)
                        # else:
                        #     fadj_x = self.transform(fadj_x)
                        #     fadj_x = self.relu(fadj_x)
                        #     fadj_x = torch.max(fadj_x, dim=0)[0]  # torch.mean(adj_x, dim=0)
                        edge_dep = torch.cat((edge_dep, fadj_x), -1)

                        out = torch.masked_select(bw_edgedep[i, :], bw_edgdepmask[i, :])
                        indices = Variable(out.cuda())
                        badj_x = torch.index_select(bw_input, 0, indices)
                        badj_x = torch.mean(badj_x, dim=0)
                        # if k<self.K-1:
                        #     badj_x = torch.mean(badj_x, dim=0)
                        # else:
                        #     badj_x = self.transform(badj_x)
                        #     badj_x = self.relu(badj_x)
                        #     badj_x = torch.max(badj_x, dim=0)[0]  # torch.mean(adj_x, dim=0)
                        edge_dep = torch.cat((edge_dep, badj_x), -1).view(1,-1)
                        # print("edge_dep",edge_dep)
                        edge_hidden[kk, :] = edge_dep.view(1, -1)
                        kk = kk+1

                edge_hidden = self.edge(edge_hidden)
                edge_hidden = self.relu(edge_hidden)

            if k<self.K-1:
                out_edge_hidden = Variable(torch.zeros(len(bw_edgedep), self.outfeature))
                if self.cuda:
                    out_edge_hidden = out_edge_hidden.cuda()
                kk = 0
                for i in range (len(bw_edgedep)):
                    if len(torch.masked_select(fw_edgedep[i, :], fw_edgdepmask[i, :])) > 0:
                        out_edge_hidden[i, :]= edge_hidden[kk,:].clone()
                        kk = kk+1
                    else:
                        out_edge_hidden[i, :] = edge_embs[i, :].clone()
                edge_embs = out_edge_hidden
        return fw_input, bw_input  # fw_output,bw_output

class Node_embedding(nn.Module):  # 继承nn.Module  yang 20190205
    def __init__(self, opt,embs):
        super(Node_embedding, self).__init__()  # 等价于nn.Module.__init__(self)
        self.out_features = opt.txt_rnn_size
        # self.edge_embs = nn.Parameter(torch.randn(opt.dep_len, opt.txt_rnn_size))
        self.gc1 = Node_emblayer(opt,2 * opt.txt_rnn_size,opt.txt_rnn_size)
        self.edge_embs = embs["dep_lut"]
        self.dep_len = opt.dep_len
        # self.gc2 = Node_emblayer2(2 * opt.txt_rnn_size, opt.txt_rnn_size)
        if opt.cuda:
            self.cuda()

    # def node_sence(self, fw_input,bw_input, edge_embs, fw_adj, bw_adj,bw_end,fw_end,fw_edgeid,bw_edgeid,bw_edgeend,fw_edgeend):
    #
    #     fw_hidden = Variable(torch.zeros(len(bw_input), self.out_features))
    #     bw_hidden = Variable(torch.zeros(len(bw_input), self.out_features))
    #
    #     if self.cuda:
    #         fw_hidden = fw_hidden.cuda()
    #         bw_hidden = bw_hidden.cuda()
    #
    #     for i in range(len(fw_input)):
    #         indices = Variable(fw_adj[i, 0:fw_end[i]].cuda())
    #         adj_x = torch.index_select(fw_input, 0, indices)
    #         if fw_edgeend[i]>0:
    #             indices = Variable(fw_edgeid[i, 0:fw_edgeend[i]].cuda())
    #             adj_x1 = torch.index_select(edge_embs, 0, indices)
    #             adj_x = torch.cat((adj_x1, adj_x), 0)
    #
    #         adj_x = torch.cat((fw_input[i,:].view(1,-1), adj_x), 0)
    #         # adj_x = torch.max(adj_x, dim=0)[0]
    #         adj_x = torch.mean(adj_x, dim=0)
    #         fw_hidden[i,:] = adj_x.view(1, -1)
    #         # fw_hidden[i, :] = fw_input[i, :]
    #
    #         indices = Variable(bw_adj[i, 0:bw_end[i]].cuda())
    #         adj_x = torch.index_select(bw_input, 0, indices)
    #
    #         if fw_edgeend[i]>0:
    #             indices = Variable(bw_edgeid[i, 0:bw_edgeend[i]].cuda())
    #             adj_x1 = torch.index_select(edge_embs, 0, indices)
    #             adj_x = torch.cat((adj_x1, adj_x), 0)
    #         adj_x = torch.cat((bw_input[i,:].view(1,-1), adj_x), 0)
    #         # adj_x = torch.max(adj_x, dim=0)[0]
    #         adj_x = torch.mean(adj_x, dim=0)
    #         bw_hidden[i,:] = adj_x.view(1, -1)
    #
    #     return fw_hidden,bw_hidden

    def forward(self, packed_input: PackedSequence,fw_adj,bw_adj,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep):

        fw_input = packed_input.data
        bw_input = packed_input.data
        embs_ind = Variable(torch.linspace(0,self.dep_len-1,self.dep_len).long())
        if self.cuda:
            embs_ind = embs_ind.cuda()
        edge_embs = self.edge_embs(embs_ind)
        fw_output, bw_output = self.gc1(fw_input, bw_input, edge_embs, fw_adj, bw_adj,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep)

        node_embedding = PackedSequence(torch.cat([fw_output, bw_output], dim=1), packed_input.batch_sizes)
        return node_embedding

class Concept_Classifier(nn.Module):

    def __init__(self, opt, embs):
        super(Concept_Classifier, self).__init__()
        self.txt_rnn_size = 2*opt.txt_rnn_size

        self.n_cat = embs["cat_lut"].num_embeddings
        self.n_high = embs["high_lut"].num_embeddings
        self.n_aux = embs["aux_lut"].num_embeddings

        self.cat_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_cat,bias = opt.cat_bias))

        self.le_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_high+1,bias = opt.lemma_bias))

        self.ner_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_rnn_size,self.n_aux,bias = opt.cat_bias))

        self.t = 1
        self.sm =  nn.Softmax(dim=1)
        if opt.cuda:
            self.cuda()

    def forward(self, src_enc ):
        '''
            src_enc: pack(data x txt_rnn_size ,batch_size)
           src_le:  pack(data x 1 ,batch_size)

           out:  (datax n_cat, batch_size),    (data x n_high+1,batch_size)
        '''

        assert isinstance(src_enc,PackedSequence)


     #   high_embs = self.high_lut.weight.expand(le_score.size(0),self.n_high,self.dim)
      #  le_self_embs = self.lemma_lut(src_le.data).unsqueeze(1)
      #  le_emb = torch.cat([high_embs,le_self_embs],dim=1) #data x high+1 x dim

        pre_enc =src_enc.data

        cat_score = self.cat_score(pre_enc) #  n_data x n_cat
        ner_score = self.ner_score(pre_enc)#  n_data x n_cat
        le_score = self.le_score (src_enc.data)
        le_prob = self.sm(le_score)
        cat_prob = self.sm(cat_score)
        ner_prob = self.sm(ner_score)
        batch_sizes = src_enc.batch_sizes
        return   PackedSequence(cat_prob,batch_sizes),PackedSequence(le_prob,batch_sizes),PackedSequence(ner_prob,batch_sizes)

class ConceptIdentifier(nn.Module):
    #could share encoder with other model
    def __init__(self, opt,embs,encoder = None):
        super(ConceptIdentifier, self).__init__()
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = SentenceEncoder( opt, embs)
        self.generator = Concept_Classifier( opt, embs)
        self.node_embedding = Node_embedding(opt,embs)  # yang

    def forward(self, srcBatch,fw_adjs,bw_adjs,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep):
        src_enc = self.encoder(srcBatch)
        node_embedding = self.node_embedding(src_enc,fw_adjs,bw_adjs,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep)

        probBatch = self.generator(node_embedding)
        return probBatch,node_embedding
