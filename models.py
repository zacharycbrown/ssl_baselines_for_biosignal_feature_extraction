# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# stager net
class StagerNet(nn.Module):
    def __init__(self, channels, dropout_rate=0.5, embed_dim=100):
        super(StagerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, (1, channels), stride=(1, 1))
        self.conv2 = nn.Conv2d(1, 16, (50, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, (50, 1), stride=(1, 1))
        self.linear1 = nn.Linear(208*channels, embed_dim)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        # input assumed to be of shape (C,T)=(2,3000)
        x = torch.unsqueeze(x, 1)

        # convolve x with C filters to 1 by T by C
        x = self.conv1(x)
        # permute to (C, T, I)
        x = x.permute(0, 3, 2, 1)

        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, (13, 1)))
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, (13, 1)))
        x = self.batchnorm2(x)

        x = torch.flatten(x, 1) # flatten all but batch dim
        x = F.dropout(x, p=self.dropout_rate)
        x = self.linear1(x)
        return x

# rp net for Relative Positioning Task
class RPNet(nn.Module):
    def __init__(self, channels, dropout_rate=0.5, embed_dim=100):
        super(RPNet, self).__init__()
        self.embed_model = StagerNet(channels, dropout_rate=dropout_rate, embed_dim=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
    
    def forward(self, x1, x2):
        x1_embedded = self.embed_model(x1)
        x2_embedded = self.embed_model(x2)

        # the torch.abs() is able to emulate the grp function in RP
        out = self.linear(torch.abs(x1_embedded - x2_embedded))
        return out

# ts net for Temporal Shuffling Task
class TSNet(nn.Module):
    def __init__(self, channels, dropout_rate=0.5, embed_dim=100):
        super(TSNet, self).__init__()
        self.embed_model = StagerNet(channels, dropout_rate=dropout_rate, embed_dim=embed_dim)
        self.linear = nn.Linear(2*embed_dim, 1)

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
    
    def forward(self, x1, x2, x3):
        x1_embedded = self.embed_model(x1)
        x2_embedded = self.embed_model(x2)
        x3_embedded = self.embed_model(x3)

        # the torch.abs() is able to emulate the grp function in RP
        # print("TSNet.forward: x1_embedded shape == ", x1_embedded.shape)
        # print("TSNet.forward: x2_embedded shape == ", x2_embedded.shape)
        # print("TSNet.forward: x3_embedded shape == ", x3_embedded.shape)
        # print("TSNet.forward: torch.abs(x1_embedded - x2_embedded) shape == ", torch.abs(x1_embedded - x2_embedded).shape)
        # print("TSNet.forward: torch.abs(x2_embedded - x3_embedded) shape == ", torch.abs(x2_embedded - x3_embedded).shape)
        # print("TSNet.forward: torch.cat((torch.abs(x1_embedded - x2_embedded), torch.abs(x2_embedded - x3_embedded)), dim=-1) shape == ", torch.cat((torch.abs(x1_embedded - x2_embedded), torch.abs(x2_embedded - x3_embedded)), dim=-1).shape)
        # raise Exception()
        out = self.linear(torch.cat((torch.abs(x1_embedded - x2_embedded), torch.abs(x2_embedded - x3_embedded)), dim=-1))
        return out


# cpc net for Contrastive Predictive Coding Task
class CPCNet(nn.Module):
    def __init__(self, Np, channels, ct_dim=100, h_dim=100, dropout_rate=0.5, embed_dim=100):
        super(CPCNet, self).__init__()

        self.BATCH_DIM = 0
        self.ENTRY_DIM = 1
        self.PRED_VAL_DIM = 2
        self.NUM_ENTRIES = 16
        self.NUM_PREDS = 11

        self.Np = Np
        self.channels = channels
        self.ct_dim = ct_dim
        self.h_dim = h_dim
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim

        self.embed_model = StagerNet(channels, dropout_rate=dropout_rate, embed_dim=embed_dim)
        self.gru = nn.GRU(ct_dim, h_dim, 1, batch_first=True)
        self.bilinear_list = nn.ModuleList()

        for _ in range(Np):
            self.bilinear_list.append(nn.Bilinear(in1_features=h_dim, in2_features=ct_dim, out_features=1, bias=False))

    def forward(self, Xc, Xp, Xb):
        """
        This function involves:
            1) Some very convoluted reshaping of the inputs:
                Essentially, we need to get a 3d tensor with the indices as (batch, entry, predicted value) for 
                positive and negative samples where the positive samples are always in the 0th index of the last 
                dimension of the output
        *** Note: in jstranne's original repo, the for-loop had the sample and predicted indices swapped - therefore 
                  the naming convention may be off (it may be inappropriate to view 'prediction' indices as actual
                  predictions, for example). Unfortunately, I am unable to clarify this with jstranne at the moment
        """

        # embed and reshape Xb
        Xb = [[self.embed_model(torch.squeeze(Xb[:,i,j,:,:])) for i in range(Xb.shape[1])] for j in range(Xb.shape[2])]
        for i in range(len(Xb)):
            Xb[i] = torch.stack(Xb[i])
        Xb = torch.stack(Xb).permute(2, 1, 0, 3)

        # embed and reshape Xc
        Xc = [self.embed_model(torch.squeeze(Xc[:,i,:,:])) for i in range(Xc.shape[1])]
        Xc = torch.stack(Xc).permute(1, 0, 2)

        # embed and reshape Xp
        Xp = [self.embed_model(torch.squeeze(Xp[:,i,:,:])) for i in range(Xp.shape[1])]
        Xp = torch.stack(Xp).permute(1, 0, 2).unsqueeze(2)

        # combine Xp and Xb tensors
        Xp = torch.cat((Xp, Xb), 2)

        # initialize output tensor
        out = torch.empty([Xb.shape[self.BATCH_DIM], self.NUM_ENTRIES, self.NUM_PREDS], dtype=Xp.dtype, device=Xp.device)

        # process the inputs to make the final output
        _, hidden = self.gru(Xc)
        hidden = torch.squeeze(hidden)

        for batch in range(Xp.shape[self.BATCH_DIM]):
            for sample in range(Xp.shape[self.ENTRY_DIM]):
                for predicted in range(Xp.shape[self.PRED_VAL_DIM]):
                    out[batch, sample, predicted] = self.bilinear_list[sample](hidden[batch, :], Xp[batch, sample, predicted, :])
        
        return out
    
    def custom_cpc_loss(self, input):
        """
        Runs a negative log softmax on the first column of the last index using the knowledge that this is where 
        all the positive samples are

        Input should be in the shape [batch, np, nb+1], the first index of nb+1 being the 'correct' one
        """
        NB_DIM = 2
        CORRECT_INDEX = 0
        loss_func = nn.LogSoftmax(dim=NB_DIM)
        log_soft = loss_func(input)[:,:,CORRECT_INDEX]
        return -torch.sum(log_soft)


class DownstreamNet(nn.Module):
    def __init__(self, embedders, classes, embed_dim=100):
        """
        Network for downstream prediction/classification tasks. Simply the embeder(s) and a final linear 
        layer.

        embedders: list of embedding models (trained/untrained and trainable/frozen)
        classes: the total number of classes to be used in prediction/classification task
        """
        super(DownstreamNet, self).__init__()
        self.BATCH_DIM_INDEX = 0
        self.EMBED_DIM_INDEX = 1
        self.embedders = nn.ModuleList()
        for embedder in embedders:
            self.embedders.append(embedder)
        self.num_embedders = len(embedders)
        self.linear = nn.Linear(self.num_embedders*embed_dim, classes)
    
    def forward(self, x):
        if self.num_embedders == 1:
            x = self.embedders[0](x)
        else:
            x_embeds = [self.embedders[i](x) for i in range(self.num_embedders)]
            x = torch.cat(tuple(x_embeds), dim=self.EMBED_DIM_INDEX)
        return self.linear(x)