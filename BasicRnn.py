import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(torch.zeros(module.num_layers * 2, batch_size, module.hidden)), \
           Variable(torch.zeros(module.num_layers * 2, batch_size, module.hidden))

class BasicRnn(nn.Module):

    def __init__(self, input_size=82, hidden=128, num_layers=2):
        super(BasicRnn, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

        ## *2 for bidirectional. +1 for adding the last y value explictly into the FC layer
        self.fc = nn.Linear((hidden * 2) + 1, 1)

        self.criterion = nn.MSELoss()


    def forward(self,batch):

        unpack_batch = pad_packed_sequence(batch,batch_first=True)
        init_s = zero_state(self,batch_size=batch.batch_sizes[0])

        packed_output,(h_0,c_0) = self.lstm(batch,init_s)

        ##TODO: variable initialize - to calc grad or not

        unPacked_output = pad_packed_sequence(packed_output,batch_first=True) #batchSize x seqLength x FeatureSize
        # TODO: check that -1 is indeed the last timestamp (and not 0)
        last_y_values = unpack_batch[0][:, -1, -1].data.contiguous().view(batch.batch_sizes[0], 1)
        x = self.fc( Variable( torch.cat( (unPacked_output[0].data[:,-1,:],last_y_values),1)  ))

        return x


def create():
    return BasicRnn()