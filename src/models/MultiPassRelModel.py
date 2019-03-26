#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for relation identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.helper_module import mypack ,myunpack,MyPackedSequence,MyDoublePackedSequence,mydoubleunpack,mydoublepack,DoublePackedSequence,doubleunpack,data_dropout
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from utility.constants import *
import numpy as np



#sentence encoder for root identification
class RootSentenceEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.root_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.rel_rnn_size // self.num_directions
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim

        super(RootSentenceEncoder, self).__init__()


        self.rnn =nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn,
                           batch_first=True)

        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]

        self.pos_lut = embs["pos_lut"]


        self.ner_lut = embs["ner_lut"]

        self.alpha = opt.alpha
        if opt.cuda:
            self.rnn.cuda()



    def forward(self, packed_input,hidden=None):
    #input: pack(data x n_feature ,batch_size)
    #posterior: pack(data x src_len ,batch_size)
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)

        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([word_fix_embed,lemma_emb,pos_emb,ner_emb],1)#  data,embed

        emb = PackedSequence(emb, packed_input.batch_sizes)


        outputs = self.rnn(emb, hidden)[0]

        return  outputs

#combine amr node embedding and aligned sentence token embedding
class RootEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.amr_enlayers
        #share hyper parameter with relation model
        self.size = opt.rel_dim
        inputSize = embs["cat_lut"].embedding_dim + embs["lemma_lut"].embedding_dim+opt.rel_rnn_size
        super(RootEncoder, self).__init__()

        self.cat_lut = embs["cat_lut"]

        self.lemma_lut  = embs["lemma_lut"]

        self.root = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size ),
            nn.ReLU()
        )


        self.alpha = opt.alpha
        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        head_emb,lengths = [],[]
        src_enc = myunpack(*src_enc)  #  pre_amr_l/src_l  x batch x dim
        for i, index in enumerate(indexes):
            enc = src_enc[i]  #src_l x  dim
            head_emb.append(enc[index])  #var(amr_l  x dim)
            lengths.append(len(index))
        return mypack(head_emb,lengths)

    #input: all_data x n_feature, lengths
    #index: batch_size x var(amr_len)
    #src_enc   (batch x amr_len) x src_len x txt_rnn_size

    #head: batch   x var( amr_len x txt_rnn_size )

    #dep : batch x var( amr_len x amr_len x txt_rnn_size )

    #heads: [var(len),rel_dim]
    #deps: [var(len)**2,rel_dim]
    def forward(self, input, index,src_enc):
        assert isinstance(input, MyPackedSequence),input
        input,lengths = input
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        cat_embed = self.cat_lut(input[:,AMR_CAT])
        lemma_embed = self.lemma_lut(input[:,AMR_LE])

        amr_emb = torch.cat([cat_embed,lemma_embed],1)
    #    print (input,lengths)

        head_emb = self.getEmb(index,src_enc)  #packed, mydoublepacked


        root_emb = torch.cat([amr_emb,head_emb.data],1)
        root_emb = self.root(root_emb)

        return MyPackedSequence(root_emb,lengths)

class RNode_emblayer(nn.Module):  # 继承nn.Module  yang 20190205
    def __init__(self,opt,in_features,out_features):
        super(RNode_emblayer, self).__init__()  # 等价于nn.Module.__init__(self)

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
class RNode_embedding(nn.Module):  # 继承nn.Module  yang 20190205
    def __init__(self, opt, embs):
        super(RNode_embedding, self).__init__()  # 等价于nn.Module.__init__(self)
        self.out_features = opt.txt_rnn_size
        # self.edge_embs = nn.Parameter(torch.randn(opt.dep_len, opt.txt_rnn_size))
        self.gc1 = RNode_emblayer(opt, 2 * opt.txt_rnn_size, opt.txt_rnn_size)
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

    def forward(self, packed_input: PackedSequence, fw_adj, bw_adj, fw_edgeid, bw_edgeid, fw_edgedep, bw_edgedep):

        fw_input = packed_input.data
        bw_input = packed_input.data
        embs_ind = Variable(torch.linspace(0, self.dep_len - 1, self.dep_len).long())
        if self.cuda:
            embs_ind = embs_ind.cuda()
        edge_embs = self.edge_embs(embs_ind)
        fw_output, bw_output = self.gc1(fw_input, bw_input, edge_embs, fw_adj, bw_adj, fw_edgeid, bw_edgeid,
                                    fw_edgedep, bw_edgedep)

        node_embedding = PackedSequence(torch.cat([fw_output, bw_output], dim=1), packed_input.batch_sizes)
        return node_embedding

#multi pass sentence encoder for relation identification
class RelSentenceEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.rel_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.rel_rnn_size // self.num_directions
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim

        self.Rnnsize = 2*opt.rel_rnn_size+1
        super(RelSentenceEncoder, self).__init__()

        self.node_embedding = RNode_embedding(opt, embs)
        self.sencrnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        self.rnn =nn.LSTM(self.Rnnsize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn,
                           batch_first=True)   #first is for root

        self.lemma_lut = embs["lemma_lut"]

        self.word_fix_lut  = embs["word_fix_lut"]

        self.pos_lut = embs["pos_lut"]


        self.ner_lut = embs["ner_lut"]

        self.alpha = opt.alpha
        if opt.cuda:
            self.rnn.cuda()

    def posteriorIndictedEmb(self,embs,posterior):
        #real alignment is sent in as list of index
        #variational relaxed posterior is sent in as MyPackedSequence

        #out   (batch x amr_len) x src_len x (dim+1)
        embs,src_len = unpack(embs)

        if isinstance(posterior,MyPackedSequence):
       #     print ("posterior is packed")
            posterior = myunpack(*posterior)
            embs = embs.transpose(0,1)
            out = []
            lengths = []
            amr_len = [len(p) for p in posterior]
            for i,emb in enumerate(embs):
                expanded_emb = emb.unsqueeze(0).expand([amr_len[i]]+[i for i in emb.size()]) # amr_len x src_len x dim
                indicator = posterior[i].unsqueeze(2)  # amr_len x src_len x 1
                out.append(torch.cat([expanded_emb,indicator],2))  # amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_len[i]
            data = torch.cat(out,dim=0)

            return pack(data, lengths, batch_first=True), amr_len

        elif isinstance(posterior,list):
            embs = embs.transpose(0,1)
            src_l = embs.size(1)
            amr_len = [len(i) for i in posterior]
            out = []
            lengths = []
            for i,emb in enumerate(embs):
                amr_l = len(posterior[i])
                expanded_emb = emb.unsqueeze(0).expand([amr_l]+[i for i in emb.size()]) # amr_len x src_len x dim
                indicator = emb.data.new(amr_l,src_l).zero_()
                indicator.scatter_(1, posterior[i].data.unsqueeze(1), 1.0) # amr_len x src_len x 1
                indicator = Variable(indicator.unsqueeze(2))
                out.append(torch.cat([expanded_emb,indicator],2))  # amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_l
            data = torch.cat(out,dim=0)

            return pack(data,lengths,batch_first=True),amr_len


    def forward(self, packed_input, packed_posterior,fw_adjs,bw_adjs,fw_edgeid,bw_edgeid,fw_edgedep,bw_edgedep,hidden=None):
    #input: pack(data x n_feature ,batch_size)
    #posterior: pack(data x src_len ,batch_size)
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([word_fix_embed,lemma_emb,pos_emb,ner_emb],1)#  data,embed

        emb = PackedSequence(emb, packed_input.batch_sizes)

        outputs, hidden_t = self.sencrnn(emb, hidden)

        # print("outputs",outputs)
        # print("self.Rnnsize",self.Rnnsize)
        node_embedding = self.node_embedding(outputs, fw_adjs, bw_adjs, fw_edgeid, bw_edgeid, fw_edgedep, bw_edgedep)
        poster_emb,amr_len = self.posteriorIndictedEmb(node_embedding,packed_posterior)

        # print("poster_emb", poster_emb)
        if hidden == None:

           Outputs = self.rnn(poster_emb, hidden)[0]   # h layer*batch_size*hidden_size

        else:
           print("yang hidden")
           Outputs = self.rnn(poster_emb, hidden)[0]

        return  DoublePackedSequence(Outputs,amr_len,Outputs.data)


#combine amr node embedding and aligned sentence token embedding
class RelEncoder(nn.Module):

    def __init__(self, opt, embs):
        super(RelEncoder, self).__init__()

        self.layers = opt.amr_enlayers

        self.size = opt.rel_dim
        inputSize = embs["cat_lut"].embedding_dim + embs["lemma_lut"].embedding_dim+opt.rel_rnn_size

        self.head = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size )
        )

        self.dep = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(inputSize,self.size )
        )

        self.cat_lut = embs["cat_lut"]

        self.lemma_lut  = embs["lemma_lut"]
        self.alpha = opt.alpha

        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        head_emb,dep_emb = [],[]
        src_enc,src_l = doubleunpack(src_enc)  # batch x var(amr_l x src_l x dim)
        length_pairs = []
        for i, index in enumerate(indexes):
            enc = src_enc[i]  #amr_l src_l dim
            dep_emb.append(enc.index_select(1,index))  #var(amr_l x amr_l x dim)
            head_index = index.unsqueeze(1).unsqueeze(2).expand(enc.size(0),1,enc.size(-1))
       #     print ("getEmb",enc.size(),dep_index.size(),head_index.size())
            head_emb.append(enc.gather(1,head_index).squeeze(1))  #var(amr_l  x dim)
            length_pairs.append([len(index),len(index)])
        return mypack(head_emb,[ls[0] for ls in length_pairs]),mydoublepack(dep_emb,length_pairs),length_pairs

    #input: all_data x n_feature, lengths
    #index: batch_size x var(amr_len)
    #src_enc   (batch x amr_len) x src_len x txt_rnn_size

    #head: batch   x var( amr_len x txt_rnn_size )

    #dep : batch x var( amr_len x amr_len x txt_rnn_size )

    #heads: [var(len),rel_dim]
    #deps: [var(len)**2,rel_dim]
    def forward(self, input, index,src_enc):
        assert isinstance(input, MyPackedSequence),input
        input,lengths = input
        if self.alpha and self.training:
            input = data_dropout(input,self.alpha)
        cat_embed = self.cat_lut(input[:,AMR_CAT])
        lemma_embed = self.lemma_lut(input[:,AMR_LE])

        amr_emb = torch.cat([cat_embed,lemma_embed],1)
    #    print (input,lengths)

        head_emb_t,dep_emb_t,length_pairs = self.getEmb(index,src_enc)  #packed, mydoublepacked


        head_emb = torch.cat([amr_emb,head_emb_t.data],1)

        dep_amr_emb_t = myunpack(*MyPackedSequence(amr_emb,lengths))
        dep_amr_emb = [ emb.unsqueeze(0).expand(emb.size(0),emb.size(0),emb.size(-1))      for emb in dep_amr_emb_t]

        mydouble_amr_emb = mydoublepack(dep_amr_emb,length_pairs)

    #    print ("rel_encoder",mydouble_amr_emb.data.size(),dep_emb_t.data.size())
        dep_emb = torch.cat([mydouble_amr_emb.data,dep_emb_t.data],-1)

       # emb_unpacked = myunpack(emb,lengths)

        head_packed = MyPackedSequence(self.head(head_emb),lengths) #  total,rel_dim
        head_amr_packed = MyPackedSequence(amr_emb,lengths) #  total,rel_dim

   #     print ("dep_emb",dep_emb.size())
        size = dep_emb.size()
        dep = self.dep(dep_emb.view(-1,size[-1])).view(size[0],size[1],-1)

        dep_packed  = MyDoublePackedSequence(MyPackedSequence(dep,mydouble_amr_emb[0][1]),mydouble_amr_emb[1],dep)

        return  head_amr_packed,head_packed,dep_packed  #,MyPackedSequence(emb,lengths)


class RelModel(nn.Module):
    def __init__(self, opt,embs):
        super(RelModel, self).__init__()
        self.root_encoder = RootEncoder(opt,embs)
        self.encoder = RelEncoder( opt, embs)
        self.generator = RelCalssifierBiLinear( opt, embs,embs["rel_lut"].num_embeddings)

        self.root = nn.Linear(opt.rel_dim,1)
        self.LogSoftmax = nn.LogSoftmax(dim=0)


    def root_score(self,mypackedhead):
        heads = myunpack(*mypackedhead)
        output = []
        for head in heads:
            score = self.root(head).squeeze(1)
            output.append(self.LogSoftmax(score))
        return output

    def forward(self, srlBatch, index,src_enc,root_enc):
        mypacked_root_enc = self.root_encoder(srlBatch, index,root_enc) #with information from le cat enc
        roots = self.root_score(mypacked_root_enc)

        encoded= self.encoder(srlBatch, index,src_enc)
        score_packed = self.generator(*encoded)

        return score_packed,roots #,arg_logit_packed


class RelCalssifierBiLinear(nn.Module):

    def __init__(self, opt, embs,n_rel):
        super(RelCalssifierBiLinear, self).__init__()
        self.n_rel = n_rel
        self.cat_lut = embs["cat_lut"]
        self.inputSize = opt.rel_dim


        self.bilinear = nn.Sequential(nn.Dropout(opt.dropout),
                                  nn.Linear(self.inputSize,self.inputSize* self.n_rel))
        self.head_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                   nn.Linear(self.inputSize,self.n_rel))
        self.dep_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                      nn.Linear(self.inputSize,self.n_rel))
        self.bias = nn.Parameter(torch.normal(torch.zeros(self.n_rel)).cuda())


     #   self.lsm = nn.LogSoftmax()
        self.cat_lut = embs["cat_lut"]
        self.lemma_lut  = embs["lemma_lut"]
        if opt.cuda:
            self.cuda()

    def bilinearForParallel(self,inputs,length_pairs):
        output = []
        ls = []
        for i,input in enumerate(inputs):

            #head_t : amr_l x (  rel_dim x n_rel)
            #dep_t : amr_l x amr_l x rel_dim
            #head_bias : amr_l  x n_rel
            #dep_bias : amr_l  x   amr_l  x n_rel
            head_t,dep_t,head_bias,dep_bias = input
            l = len(head_t)
            ls.append(l)
            head_t = head_t.view(l,-1,self.n_rel)
            score =dep_t[:,:length_pairs[i][1]].bmm( head_t.view(l,-1,self.n_rel)).view(l,l,self.n_rel).transpose(0,1)

            dep_bias =  dep_bias[:,:length_pairs[i][1]]
            score = score + dep_bias

            score = score + head_bias.unsqueeze(1).expand_as(score)
            score = score+self.bias.unsqueeze(0).unsqueeze(1).expand_as(score)
            score = F.log_softmax(score.view(ls[-1]*ls[-1],self.n_rel),dim=1) # - score.exp().sum(2,keepdim=True).log().expand_as(score)
            
            output.append(score.view(ls[-1]*ls[-1],self.n_rel))
        return output,[l**2 for l in ls]


    def forward(self, _,heads,deps):
        '''heads.data: mypacked        amr_l x rel_dim
            deps.data: mydoublepacked     amr_l x amr_l x rel_dim
        '''
        heads_data = heads.data
        deps_data = deps.data

        head_bilinear_transformed = self.bilinear (heads_data)  #all_data x (    n_rel x inputsize)

        head_bias_unpacked = myunpack(self.head_bias(heads_data),heads.lengths) #[len x n_rel]

        size = deps_data.size()
        dep_bias =  self.dep_bias(deps_data.view(-1,size[-1])).view(size[0],size[1],-1)

        dep_bias_unpacked,length_pairs = mydoubleunpack(MyDoublePackedSequence(MyPackedSequence( dep_bias,deps[0][1]),deps[1],dep_bias) ) #[len x n_rel]

        bilinear_unpacked = myunpack(head_bilinear_transformed,heads.lengths)

        deps_unpacked,length_pairs = mydoubleunpack(deps)
        output,l = self.bilinearForParallel( zip(bilinear_unpacked,deps_unpacked,head_bias_unpacked,dep_bias_unpacked),length_pairs)
        myscore_packed = mypack(output,l)

      #  prob_packed = MyPackedSequence(myscore_packed.data,l)
        return myscore_packed