import numpy as np
import os
import torch
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import ToTensor


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Linear(embedding_size, hidden_size)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, num_layers = num_stacked_layers, batch_first = True, dropout = 0.5)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, x, hidden_in1, mem_in1):
        # batch_size = x.size(0)
        # embeds = self.embedding(x)
        lstm_out, (hidden_out1, mem_out1) = self.lstm(x, (hidden_in1, mem_in1))
        lstm_out = self.relu(lstm_out)
        out_arr = self.dense1(lstm_out)
        out_arr = self.dense2(out_arr)
        output = self.dense3(out_arr)
        # output = self.softmax(out_arr)

        return output, hidden_out1, mem_out1
    


    
class LSTM_divided(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 64, num_stacked_layers_pose = 4, num_stacked_layers_hands = 4):
        super().__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Linear(embedding_size, hidden_size)
        self.lstm_pose = nn.LSTM(input_size = 132, hidden_size=hidden_size, num_layers = num_stacked_layers_pose, batch_first = True, dropout = 0.5)
        self.lstm_hands = nn.LSTM(input_size = input_size-132, hidden_size=hidden_size, num_layers = num_stacked_layers_hands, batch_first = True, dropout = 0.5)
        self.dense1 = nn.Linear(hidden_size, hidden_size)

        self.dense2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.dense_out = nn.Linear(hidden_size*2, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, x, hidden_in1, mem_in1, hidden_in2, mem_in2):
        # batch_size = x.size(0)
        # embeds = self.embedding(x)
        lstm1_out, (hidden_out1, mem_out1) = self.lstm_pose(x[:,:,:132], (hidden_in1, mem_in1))
        lstm1_out = self.relu(lstm1_out)
        out1_arr = self.dense1(lstm1_out)
        # print(out1_arr.shape)

        lstm2_out, (hidden_out2, mem_out2) = self.lstm_hands(x[:,:,132:], (hidden_in2, mem_in2))
        lstm2_out = self.relu(lstm2_out)
        out2_arr = self.dense1(lstm2_out)
        # print(out2_arr.shape)

        out_arr = torch.cat((out1_arr, out2_arr), dim = 2)
        # print(out_arr.shape)


        out_arr = self.dense2(out_arr)
        out_arr = self.dense2(out_arr)
        output = self.dense_out(out_arr)
        # output = self.softmax(out_arr)

        return output, hidden_out1, mem_out1, hidden_out2, mem_out2
    








# model = LSTM(258, 1743, 128, 4)
# dummy_input = torch.randn(1,30, 258)
# dummy_hidden1 = torch.randn(4, 1, 128)
# dummy_mem1 = torch.randn(4, 1, 128)
# dummy_hidden2 = torch.randn(4, 1, 128)
# dummy_mem2 = torch.randn(4, 1, 128)
# dummy_hidden3 = torch.randn(4, 1, 128)
# dummy_mem3 = torch.randn(4, 1, 128)
# output, a,b,c,d,e,f = model(dummy_input, dummy_hidden1, dummy_mem1, dummy_hidden2, dummy_mem2, dummy_hidden3, dummy_mem3)

# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.format = 'png'
# dot.render('model_graph')
# print("Visualization saved as 'model_graph.png'")
    

