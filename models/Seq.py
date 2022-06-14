import torch, os

class AttentionLSTM(torch.nn.Module):
    
    def __init__(self, neurons, dimension):
        super(AttentionLSTM, self).__init__()
        self.neurons = neurons
        self.Wx = torch.nn.Linear(dimension, neurons)
        self.Wxhat = torch.nn.Linear(dimension, neurons)
        self.Att = torch.nn.Sequential(torch.nn.Linear(neurons, 1), torch.nn.Sigmoid())
        

    def forward(self, X):
        Wx = self.Wx(X)
        Wthat = torch.repeat_interleave(torch.unsqueeze(X, dim=1), Wx.shape[1], dim=1)
        Wxhat = self.Wxhat(Wthat)
        Wx = torch.unsqueeze(Wx, dim=2)
        A = self.Att(torch.tanh(Wxhat + Wx))
        A = Wthat*A
        return torch.sum(A, axis=-2)

class LSTMAtt_Classifier(torch.nn.Module):

  def __init__(self, language, hidden_size=32,  **kwargs):

    super(LSTMAtt_Classifier, self).__init__()

    self.best_acc = -1
    self.language = language
    self.attention_neurons = kwargs['features_nodes']
    self.lstm_size = kwargs['interm_layer_size']
    self.att = AttentionLSTM(neurons=self.attention_neurons, dimension=hidden_size)
    self.bilstm = torch.nn.LSTM(batch_first=True, input_size=hidden_size, hidden_size=self.lstm_size, bidirectional=True, proj_size=0)
    self.lstm = torch.nn.LSTM(batch_first=True, input_size=hidden_size, hidden_size=self.lstm_size, proj_size=0)
    self.dense = torch.nn.Linear(in_features=self.lstm_size, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss() 

    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, data, encode=False):
    
    X = self.att(data['text'].to(device=self.device))
    # X, _ = self.bilstm(X)
    X, _  = self.lstm(X)

    if encode == True:
        return X[:,-1]

    return  self.dense(X[:,-1])


  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):
    return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=decay)

  def computeLoss(self, outputs, data):
    return self.loss_criterion(outputs, data['labels'].to(self.device) )

