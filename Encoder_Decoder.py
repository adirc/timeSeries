import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
from utils import maybe_cuda

def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable( maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden),module.isCuda) ), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden),module.isCuda) )

class Encoder(nn.Module):

    def __init__(self,is_cuda, input_size=82, hidden=128, num_layers=2):
        super(Encoder, self).__init__()
        self.isCuda = is_cuda
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

    def forward(self,batch):

        init_s = zero_state(self,batch_size=batch.batch_sizes[0])
        packed_output,(h_0,c_0) = self.lstm(batch,init_s)

        ##TODO: variable initialize - to calc grad or not

        ## TODO: can I get part of the packed_seq without unpaked it first ?
        unPacked_output = pad_packed_sequence(packed_output,batch_first=True) #batchSize x seqLength x FeatureSize
        # TODO: check that -1 is indeed the kast timestamp (and not 0)
        output = Variable(maybe_cuda(unPacked_output[0].data[:,-1,:],self.isCuda) ) # batchSize * inputSize*2
        #lengths = [output.size(1) for i in range(output.size(0))]
        #packed_output = pack_padded_sequence(output,lengths,batch_first=True)
        return output


class Decoder(nn.Module):

    def __init__(self,is_cuda ,input_size, hidden=128, num_layers=2):
        super(Decoder, self).__init__()
        self.isCuda = is_cuda
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)


    def forward(self,batch):
        init_s = zero_state(self,batch_size=batch.batch_sizes[0])
        packed_output,(h_0,c_0) = self.lstm(batch,init_s)
        unPacked_output = pad_packed_sequence(packed_output,batch_first=True) #batchSize x seqLength x FeatureSize
        # TODO: check that -1 is indeed the last timestamp (and not 0)
        return Variable(maybe_cuda(unPacked_output[0].data[:,-1,:],self.isCuda) )

class EncoderDecoder(nn.Module):
    def __init__(self,is_cuda):
        super(EncoderDecoder,self).__init__()
        self.isCuda = is_cuda
        self.encoder = Encoder(self.isCuda )
        # *2 for bidirectional. +1 for previous ys
        self.decoder = Decoder(self.isCuda ,self.encoder.hidden* 2 + 1)
        self.fc = nn.Linear(self.decoder.hidden * 2 + 1,1)
        self.criterion = nn.MSELoss()

    def forward(self,batch):

        batchSize = len(batch)
        historySize = batch[0].shape[0]

        # remove last column - the previous y_s
        batch_to_encoder = [b[:,:-1] for b in batch]
        big_tensor = Variable(maybe_cuda(torch.FloatTensor(batch_to_encoder),self.isCuda) )
        lengths = [big_tensor.size(1) for i in range(0, big_tensor.size(0))]
        packed_batch_to_encoder = pack_padded_sequence(big_tensor, lengths  , batch_first=True)

        c = self.encoder(packed_batch_to_encoder) # return c vectors unpacked. tensor batchSize * (encoderHiddenSize*2)
        c = c.view(c.size(0),1,c.size(1))
        c = c.expand(c.size(0),historySize,c.size(2))

        # concat c with the previoues y's
        batch_y_values = [b[:,-1] for b in batch]
        batch_y_values = torch.FloatTensor(batch_y_values)
        c_and_y = Variable(maybe_cuda(torch.cat((c.data, batch_y_values), 2),self.isCuda) )

        lengths = [c_and_y.data.size(1) for i in range(c_and_y.data.size(0))]
        packed_c_and_y = pack_padded_sequence(c_and_y,lengths,batch_first=True)

        output = self.decoder(packed_c_and_y) # batchSize * (2*decoderHiddenSize)

        #last_y_values = unpack_batch[0][:, -1, -1].data.contiguous().view(batch.batch_sizes[0], 1)
        last_y_values = [b[-1,-1] for b in batch]
        last_y_values = torch.FloatTensor(last_y_values)

        x = torch.cat((output.data, last_y_values), 1)
        predicted_value = self.fc(Variable(maybe_cuda(x,self.isCuda) ))
        return predicted_value


def create(isCuda):
    return EncoderDecoder(isCuda)