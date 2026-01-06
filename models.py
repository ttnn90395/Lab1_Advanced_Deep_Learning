import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):

        embedded_x = self.embedding(x)
        fc1_output = self.tanh(self.fc1(embedded_x))
        sum = torch.sum(fc1_output, dim=1)
        x = self.fc2(sum)

        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        embedded_x = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(embedded_x)
        x = self.fc(h_n[-1])
        
        return x.squeeze()
