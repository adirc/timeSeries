import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from utils import maybe_cuda


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(torch.zeros(module.num_layers * 2, batch_size, module.hidden)), \
           Variable(torch.zeros(module.num_layers * 2, batch_size, module.hidden))

class BasicRnn(nn.Module):

    def __init__(self, input_size=82, hidden=128, num_layers=2,biderctional = True,isCuda=False):
        super(BasicRnn, self).__init__()
        self.isCuda = isCuda
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0,
                            bidirectional=biderctional)

        biderctionalMult = 2 if biderctional else 1

        ## *2 for bidirectional. +1 for adding the last y value explictly into the FC layer
        self.fc = nn.Linear((hidden * biderctionalMult ), 1)

        self.criterion = nn.MSELoss()


    def forward(self,batch):

        batch_to_rnn = [b[:, :] for b in batch]
        big_tensor = Variable(maybe_cuda(torch.FloatTensor(batch_to_rnn), self.isCuda))
        lengths = [big_tensor.size(1) for i in range(0, big_tensor.size(0))]
        packed_batch = pack_padded_sequence(big_tensor, lengths, batch_first=True)
        init_s = zero_state(self,batch_size=len((batch)))

        packed_output,(h_0,c_0) = self.lstm(packed_batch,init_s)

        ##TODO: variable initialize - to calc grad or not

        unPacked_output = pad_packed_sequence(packed_output,batch_first=True) #batchSize x seqLength x FeatureSize
        # TODO: check that -1 is indeed the last timestamp (and not 0)
        # last_y_values = unpack_batch[0][:, -1, -1].data.contiguous().view(batch.batch_sizes[0], 1)
        #x = self.fc( Variable( torch.cat( (unPacked_output[0].data[:,-1,:],last_y_values),1)  ))
        x = self.fc(Variable(unPacked_output[0].data[:, -1, :]))

        return x


def create():
    return BasicRnn(biderctional = False)