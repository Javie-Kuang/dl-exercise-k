import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

#  Xavier 均匀初始化权重
def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]   # 输入维度（前一层神经元数）
        fan_out = weight_shape[0]  # 输出维度（当前层神经元数）
        '''
        W ~ U[-bound, bound]
        bound = sqrt(6 / (fan_in + fan_out))
        '''
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)       # 偏置初始化为 0
        print("inital  linear weight ")

# 创建一个词嵌入层，使用 均匀分布 U(-1,1) 初始化权重，然后将输入的词索引转换为词向量
class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #########################################
        # here you need to define the "self.rnn_lstm"  the input size is "embedding_dim" and the output size is "lstm_hidden_dim"
        # the lstm should have two layers, and the  input and output tensors are provided as (batch, seq, feature)
        # ???
        self.rnn_lstm=nn.LSTM(
            input_size=embedding_dim,     # 输入维度：词向量维度
            hidden_size=lstm_hidden_dim,  # 隐藏状态维度
            num_layers=2,                 # 两层 LSTM
            batch_first=True              # 输入格式：(batch, seq, feature)
        )

        ##########################################
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len )
        self.apply(weights_init) # call the weights initial function.

        self.softmax = nn.LogSoftmax() # the activation function.
        # self.tanh = nn.Tanh()
        
    def forward(self,sentence,is_test = False):
        # 词嵌入，形状：(1, batch_size, embedding_dim) → 需要调整
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        # print(batch_input.size()) # print the size of the input
        ################################################
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        
        # 调用 LSTM
        # 初始化隐藏状态和细胞状态为 0
        # (num_layers, batch, hidden)
        h0=torch.zeros(2,self.batch_size,self.lstm_dim).to(batch_input.device)
        c0=torch.zeros(2,self.batch_size,self.lstm_dim).to(batch_input.device)

        # LSTM 前向传播
        # output 形状：(batch, seq_len, lstm_hidden_dim)
        output,(hn,cn)=self.rnn_lstm(batch_input,(h0,c0))

        ################################################
        out = output.contiguous().view(-1,self.lstm_dim)

        out =  F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[ -1, : ].view(1,-1)
            output = prediction
        else:
           output = out
        # print(out)
        return output

