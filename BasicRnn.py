import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from utils import maybe_cuda,zero_state


class BasicRnn(nn.Module):

    def __init__(self, input_size=1, hidden=128, num_layers=2,biderctional = True,isCuda=False):
        super(BasicRnn, self).__init__()
        self.isCuda = isCuda
        self.bidrectional = biderctional
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
        self.states = 0
        self.counter = 0

    def forward(self,batch,reset_state = True):
        self.counter = self.counter  + 1

        #TODO: changed it to supprt electricty data
        #batch_to_rnn = [b[:, :] for b in batch]
        big_tensor = torch.FloatTensor(batch)
        big_tensor = big_tensor.view(len(batch), big_tensor.shape[1], 1)
        lengths = [big_tensor.size(1) for i in range(0, big_tensor.size(0))]
        big_tensor = Variable(maybe_cuda(big_tensor, self.isCuda))
        packed_batch = pack_padded_sequence(big_tensor, lengths, batch_first=True)

        #packed_output,(h_0,c_0) = self.lstm(packed_batch,init_s)
        if reset_state:
            self.states = zero_state(self,batch_size=len((batch)),bidrectional= self.bidrectional)

        ###TODO: Important
        ###TODO: how to set calc_grad as false for self.states
        packed_output, self.states = self.lstm(packed_batch, self.states)

        ##TODO: variable initialize - to calc grad or not

        unPacked_output = pad_packed_sequence(packed_output,batch_first=True) #batchSize x seqLength x FeatureSize
        # TODO: check that -1 is indeed the last timestamp (and not 0)
        # last_y_values = unpack_batch[0][:, -1, -1].data.contiguous().view(batch.batch_sizes[0], 1)
        #x = self.fc( Variable( torch.cat( (unPacked_output[0].data[:,-1,:],last_y_values),1)  ))
        x = self.fc(Variable(maybe_cuda(unPacked_output[0].data[:, -1, :],self.isCuda) ))

        return x





def create(isCuda):
    return BasicRnn(biderctional = False,isCuda = isCuda)