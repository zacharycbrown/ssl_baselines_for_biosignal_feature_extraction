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

        self.embedder_types = []
        self.embedders = nn.ModuleList()
        for embedder_type, embedder in embedders:
            # print("DownstreamNet.__init__ : embedder_type == ", embedder_type)
            if embedder_type not in ["RP", "TS", "CPC", "PS", "SQ", "SA"]:
                raise ValueError("Embedder type "+str(embedder_type)+" not supported")
            self.embedder_types.append(embedder_type)
            self.embedders.append(embedder)

        self.num_embedders = len(embedders)
        # print("DownstreamNet.__init__ : num_embedders == ", self.num_embedders)
        self.linear = nn.Linear(self.num_embedders*embed_dim, classes)
    
    def forward(self, x):
        # print("DownstreamNet.forward : x == ", x)
        # if self.num_embedders == 1:
        #     print("DownstreamNet.forward : RUNNING SINGLE EMBEDDER")
        #     x = self.embedders[0](x)
        # else:
        # print("DownstreamNet.forward : RUNNING MULTIPLE EMBEDDERS")
        x_embeds = []
        for i in range(self.num_embedders):
            embedded_x = None
            if self.embedder_types[i] in ["RP", "TS", "CPC", "SA"]:
                # print("\tDownstreamNet.forward : RUNNING RS, TS, CPC, OR SA MODEL")
                embedded_x = self.embedders[i](x)
            else:
                # print("\tDownstreamNet.forward : RUNNING PS OR SQ MODEL")
                _, embedded_x = self.embedders[i](x)
            # print("DownstreamNet.forward : embedded_x shape == ", embedded_x.shape)
            x_embeds.append(embedded_x)
        x = torch.cat(tuple(x_embeds), dim=self.EMBED_DIM_INDEX)
        # print("DownstreamNet.forward : x shape == ", x.shape)
        return self.linear(x)


class SACLResBlock(torch.nn.Module):
    """
    see appendix Figure A.1 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels_in, num_channels_out, kernel_size, dropout_rate=0.5):
        super(SACLResBlock, self).__init__()
        self.batch_norm_1 = torch.nn.BatchNorm1d(num_channels_in, track_running_stats=False)
        self.elu_1 = torch.nn.ELU()
        self.conv1d_residual = torch.nn.Conv1d(num_channels_in, num_channels_out, 1)
        self.conv1d_1 = torch.nn.Conv1d(num_channels_in, num_channels_out, kernel_size, padding=kernel_size-1)
        self.batch_norm_2 = torch.nn.BatchNorm1d(num_channels_out, track_running_stats=False)
        self.elu_2 = torch.nn.ELU()
        self.conv1d_2 = torch.nn.Conv1d(num_channels_out, num_channels_out, kernel_size)
        pass
    
    def forward(self, x):
        # print("\nSACLResBlock: x.shape == ", x.shape)
        x = self.batch_norm_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.elu_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x_resid = self.conv1d_residual(x)
        # print("SACLResBlock: x_resid.shape == ", x_resid.shape)
        x = self.conv1d_1(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.batch_norm_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.elu_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        x = self.conv1d_2(x)
        # print("SACLResBlock: x.shape == ", x.shape)
        out = x + x_resid
        # print("SACLResBlock: out.shape == ", out.shape)
        return out

class SACLFlatten(torch.nn.Module):
    """
    see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
    """
    def __init__(self):
        super(SACLFlatten, self).__init__()
        pass
    
    def forward(self, x):
        return x.view(x.size(0), -1)

class SACLEncoder(torch.nn.Module):
    """
    NOTE: IF YOU ARE GETTING SHAPE ERRORS, CHANGE THE SHAPE OF THE FINAL LINEAR LAYER IN THE EMBEDDER FIRST (BY CHANGING THE INTEGER-DIVISION DENOMINATOR)
    see  appendix Figure A.2 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(SACLEncoder, self).__init__()
        self.sequential_process = torch.nn.Sequential(torch.nn.Conv1d(num_channels, num_channels//2, temporal_len//32), 
                                               SACLResBlock(num_channels//2, num_channels//2, temporal_len//16), 
                                               torch.nn.MaxPool1d(4),  
                                               SACLResBlock(num_channels//2, num_channels, temporal_len//16), 
                                               torch.nn.MaxPool1d(4),  
                                               SACLResBlock(num_channels, num_channels*2, temporal_len//32), 
                                               torch.nn.ELU(), 
                                               SACLFlatten(), # see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
                                               torch.nn.Linear(num_channels*(2**1)*(int(temporal_len/16.5)), embed_dim) # added to make it easier for different data sets with different shapes to be run through model
        )

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.sequential_process(x)
        return out

class SACLNet(torch.nn.Module):
    """
    NOTE: IF YOU ARE GETTING SHAPE ERRORS, CHANGE THE SHAPE OF THE FINAL LINEAR LAYER IN THE EMBEDDER FIRST (BY CHANGING THE INTEGER-DIVISION DENOMINATOR)
    see  appendix Figure A.2 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100, num_upstream_decode_features=20):
        super(SACLNet, self).__init__()
        self.embed_model = SACLEncoder(num_channels, temporal_len, dropout_rate=0.5, embed_dim=100)
        # self.embed_model = torch.nn.Sequential(torch.nn.Conv1d(num_channels, num_channels//2, temporal_len//32), 
        #                                        SACLResBlock(num_channels//2, num_channels//2, temporal_len//16), 
        #                                        torch.nn.MaxPool1d(4),  
        #                                        SACLResBlock(num_channels//2, num_channels, temporal_len//16), 
        #                                        torch.nn.MaxPool1d(4),  
        #                                        SACLResBlock(num_channels, num_channels*2, temporal_len//32), 
        #                                        torch.nn.ELU(), 
        #                                        SACLFlatten(), # see https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
        #                                        torch.nn.Linear(num_channels*(2**1)*(int(temporal_len/16.5)), embed_dim) # added to make it easier for different data sets with different shapes to be run through model
        # )
        self.decode_model = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, num_upstream_decode_features)
        )

        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        # print("\nSACLNet: x.shape == ", x.shape, "\n")
        # x = x.permute(0, 2, 1)
        x_embedded = self.embed_model(x)
        # print("\nSACLNet: x_embedded.shape == ", x_embedded.shape, "\n")
        out = self.decode_model(x_embedded)
        # print("\nSACLNet: out.shape == ", out.shape, "\n")
        return out

class SACLAdversary(torch.nn.Module):
    """
    see  Figure 1 in arxiv.org/pdf/2007.04871.pdf for diagram
    """
    def __init__(self, embed_dim, num_subjects, dropout_rate=0.5):
        super(SACLAdversary, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, embed_dim//2), 
                                                torch.nn.ReLU(), 
                                                torch.nn.Linear(embed_dim//2, num_subjects), 
                                                torch.nn.Sigmoid() # ADDED BY ZAC TO ADDRESS NANs IN ADVERSARIAL LOSS
        )
        pass
    
    def forward(self, x):
        return self.model(x)


# class SeqCLRRecurrentEncoder(torch.nn.Module):
#     """
#     see Figure 4.A of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
#     """
#     def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
#         super(SeqCLRRecurrentEncoder, self).__init__()
#         self.TEMPORAL_DIM = 2
#         self.BATCH_DIM = 0
#         self.CHANNEL_DIM = 1

#         self.gru_1 = torch.nn.GRU(num_channels, 256)
#         self.gru_2 = torch.nn.GRU(num_channels, 128)
#         self.gru_3 = torch.nn.GRU(num_channels, 64)

#         self.x_linear_1 = torch.nn.Linear(256+128+64, 128)
#         self.x_relu_1 = torch.nn.ReLU()
#         self.h_linear_1 = torch.nn.Linear(256+128+64, 128)
#         self.h_relu_1 = torch.nn.ReLU()

#         self.x_layer_norm_1 = torch.nn.LayerNorm(128)
#         self.h_layer_norm_1 = torch.nn.LayerNorm(128)
#         self.gru_4 = torch.nn.GRU(128, 128)
        
#         self.x_layer_norm_2 = torch.nn.LayerNorm(128)
#         self.h_layer_norm_2 = torch.nn.LayerNorm(128)
#         self.gru_5 = torch.nn.GRU(128, 128)

#         self.final_linear = torch.nn.Linear(temporal_len*128, embed_dim)

#         self.num_channels = num_channels
#         self.temporal_len =temporal_len
#         self.dropout_rate = dropout_rate
#         self.embed_dim = embed_dim
    
#     def forward(self, x):
#         orig_temporal_len = x.size(self.TEMPORAL_DIM)
#         orig_batch_num = x.size(self.BATCH_DIM)
        
#         # prepare x for processing
#         # see https://discuss.pytorch.org/t/how-to-downsample-a-time-series/78485
#         # and https://discuss.pytorch.org/t/f-interpolate-weird-behaviour/36088
#         x_down_1 = torch.nn.functional.interpolate(x, size=(x.size(self.TEMPORAL_DIM)//2))
#         intermed_temporal_len = x_down_1.size(self.TEMPORAL_DIM)
#         x_down_2 = torch.nn.functional.interpolate(x_down_1, size=(x_down_1.size(self.TEMPORAL_DIM)//2))
        
#         x = x.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
#         x_down_1 = x_down_1.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
#         x_down_2 = x_down_2.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)

#         # embed x using various gru modules
#         x_embed_1, hidden_1 = self.gru_1(x) # TO-DO: do we need to remember this hidden, or not?
#         x_embed_2, hidden_2 = self.gru_2(x_down_1)# TO-DO: do we need to remember this hidden, or not?
#         x_embed_3, hidden_3 = self.gru_3(x_down_2)# TO-DO: do we need to remember this hidden, or not?

#         # upsample the two smaller embeddings, with x_embed_3 requiring two upsamples to reach the appropriate size ***Note: https://pytorch.org/docs/stable/notes/randomness.html
#         x_embed_2 = torch.nn.functional.interpolate(x_embed_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
#         # hidden_2 = torch.nn.functional.interpolate(hidden_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

#         x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)
#         x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
#         # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?
#         # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

#         # combine embeddings
#         x = torch.cat(tuple([x_embed_1, x_embed_2, x_embed_3]), dim=2)
#         h = torch.cat(tuple([hidden_1, hidden_2, hidden_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?

#         x = self.x_linear_1(x)
#         h = self.h_linear_1(h)
#         x = self.x_relu_1(x)
#         h = self.h_relu_1(h)

#         # first residual block pass
#         x_hat = self.x_layer_norm_1(x)
#         h_hat = self.h_layer_norm_1(h)# TO-DO: do we need to remember this hidden, or not?
#         x_hat, h_hat = self.gru_4(x, h)# TO-DO: do we need to remember this hidden, or not?
#         x = x + x_hat
#         h = h + h_hat# TO-DO: do we need to remember this hidden, or not?

#         # second residual block pass
#         x_hat = self.x_layer_norm_2(x)
#         h_hat = self.h_layer_norm_2(h)# TO-DO: do we need to remember this hidden, or not?
#         x_hat, _ = self.gru_5(x, h) # x_hat, h_hat = self.gru_5(x, h)# TO-DO: do we need to remember this hidden, or not?
#         x = x + x_hat
#         # h = h + h_hat
        
#         # final output generation
#         x = x.permute(1, 2, 0).contiguous().view(orig_batch_num, -1) # see https://github.com/cezannec/capsule_net_pytorch/issues/4 and https://stackoverflow.com/questions/48915810/pytorch-contiguous 
#         out = self.final_linear(x) # TO-DO: should I include the final h in this as well (via concatenation)??? if so, need to change final_linear and preceding code in forward pass
#         return out
class SeqCLRRecurrentEncoder(torch.nn.Module):
    """
    see Figure 4.A of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(SeqCLRRecurrentEncoder, self).__init__()
        self.TEMPORAL_DIM = 2
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1

        self.gru_1 = torch.nn.GRU(num_channels, 256)
        self.gru_2 = torch.nn.GRU(num_channels, 128)
        self.gru_3 = torch.nn.GRU(num_channels, 64)

        self.x_linear_1 = torch.nn.Linear(256+128+64, 128)
        self.x_relu_1 = torch.nn.ReLU()
        self.h_linear_1 = torch.nn.Linear(256+128+64, 128)
        self.h_relu_1 = torch.nn.ReLU()

        self.x_layer_norm_1 = torch.nn.LayerNorm(128)
        self.h_layer_norm_1 = torch.nn.LayerNorm(128)

        self.gru_4 = torch.nn.GRU(128, 128)
        
        self.x_layer_norm_2 = torch.nn.LayerNorm(128)
        self.h_layer_norm_2 = torch.nn.LayerNorm(128)
        
        self.gru_5 = torch.nn.GRU(128, 128)

        self.final_linear = torch.nn.Linear(128, embed_dim)

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
    
    def forward(self, x):
        orig_temporal_len = x.size(self.TEMPORAL_DIM)
        orig_batch_num = x.size(self.BATCH_DIM)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)
        
        # prepare x for processing
        # see https://discuss.pytorch.org/t/how-to-downsample-a-time-series/78485
        # and https://discuss.pytorch.org/t/f-interpolate-weird-behaviour/36088
        x_down_1 = torch.nn.functional.interpolate(x, size=(x.size(self.TEMPORAL_DIM)//2))
        intermed_temporal_len = x_down_1.size(self.TEMPORAL_DIM)
        x_down_2 = torch.nn.functional.interpolate(x_down_1, size=(x_down_1.size(self.TEMPORAL_DIM)//2))
        
        x = x.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        x_down_1 = x_down_1.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        x_down_2 = x_down_2.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)

        # embed x using various gru modules
        x_embed_1, hidden_1 = self.gru_1(x) # TO-DO: do we need to remember this hidden, or not?
        x_embed_2, hidden_2 = self.gru_2(x_down_1)# TO-DO: do we need to remember this hidden, or not?
        x_embed_3, hidden_3 = self.gru_3(x_down_2)# TO-DO: do we need to remember this hidden, or not?

        # upsample the two smaller embeddings, with x_embed_3 requiring two upsamples to reach the appropriate size ***Note: https://pytorch.org/docs/stable/notes/randomness.html
        x_embed_2 = torch.nn.functional.interpolate(x_embed_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
        # hidden_2 = torch.nn.functional.interpolate(hidden_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

        x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)
        x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
        # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?
        # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

        # combine embeddings
        x = torch.cat(tuple([x_embed_1, x_embed_2, x_embed_3]), dim=2)
        h = torch.cat(tuple([hidden_1, hidden_2, hidden_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?

        x = self.x_linear_1(x)
        h = self.h_linear_1(h)
        x = self.x_relu_1(x)
        h = self.h_relu_1(h)

        # first residual block pass
        x_hat = self.x_layer_norm_1(x)
        h_hat = self.h_layer_norm_1(h)# TO-DO: do we need to remember this hidden, or not?
        x_hat, h_hat = self.gru_4(x, h)# TO-DO: do we need to remember this hidden, or not?
        x = x + x_hat
        h = h + h_hat# TO-DO: do we need to remember this hidden, or not?

        # second residual block pass
        x_hat = self.x_layer_norm_2(x)
        h_hat = self.h_layer_norm_2(h)# TO-DO: do we need to remember this hidden, or not?
        x_hat, _ = self.gru_5(x, h) # x_hat, h_hat = self.gru_5(x, h)# TO-DO: do we need to remember this hidden, or not?
        x = x + x_hat
        # h = h + h_hat
        
        # final output generation
        x = self.final_linear(x) # TO-DO: should I include the final h in this as well (via concatenation)??? if so, need to change final_linear and preceding code in forward pass
        x = x.permute(1, 2, 0)
        return x, torch.mean(x, dim=2) # x is used for upstream task and torch.mean(x, dim=1) for downstream

class SeqCLRConvolutionalResidualBlock(torch.nn.Module):
    """
    see Figure 4.B of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, D=250, K=64, dropout_rate=0.5):
        super(SeqCLRConvolutionalResidualBlock, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1
        self.TEMPORAL_DIM = 2

        self.linear = torch.nn.Linear(D, D)
        self.sequential = torch.nn.Sequential(
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(D), 
            torch.nn.ReflectionPad1d(((K//2)-1, K//2)), 
            torch.nn.Conv1d(D, D, K)
        )
        pass
    
    def forward(self, x):
        x_hat = self.linear(x.permute(self.BATCH_DIM, 2, 1)).permute(self.BATCH_DIM, 2, 1)
        x_hat = self.sequential(x_hat)
        return x + x_hat

class SeqCLRConvolutionalEncoder(torch.nn.Module):
    """
    see Figure 4.B of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(SeqCLRConvolutionalEncoder, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 2 # 1
        self.TEMPORAL_DIM = 1 # 2

        self.K128 = 64 # 128 # 64 # 32 # 128 # 64 # 128 # 32 # 16 # 32 # 128
        self.K64 = 32 # 64 # 32 # 16 # 64 # 32 # 64 # 16 # 8 # 16 # 64
        self.K16 = 8 # 16 # 8 # 4 # 4 # 8 # 16 # 4 # 2 # 4 # 16

        self.D_INTERNAL_100 = 20 # 100 # 10 # 20 # 100 # 10 # 5 # 10 # 100
        self.D_INTERNAL_50 = 10 # 50 # 5 # 10 # 50 # 5 # 5 # 50
        self.D_INTERNAL_250 = 50 # 250 # 25 # 50 # 250 # 25 # 15 # 25 # 250
        self.D_OUT = embed_dim

        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K128//2)-1, self.K128//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K128)
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K64)
        )
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K16//2)-1, self.K16//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_50, self.K16)
        )

        self.res_block_1 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_2 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_3 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_4 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)

        self.final_relu = torch.nn.ReLU()
        self.final_batch_norm = torch.nn.BatchNorm1d(self.D_INTERNAL_250)
        self.final_reflective_padding = torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2))
        self.final_conv_layer = torch.nn.Conv1d(self.D_INTERNAL_250, self.D_OUT, self.K64)

        # self.final_linear = torch.nn.Linear(self.D_OUT*temporal_len, self.D_OUT) # added this layer (not in paper) for comparisson purposes in generalizing to other tasks

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        orig_batch_num = x.size(0)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)

        # embed x using various convolutional modules
        x_embed_1 = self.conv_block_1(x)
        x_embed_2 = self.conv_block_2(x)
        x_embed_3 = self.conv_block_3(x)

        # combine embeddings
        x = torch.cat(tuple([x_embed_1, x_embed_2, x_embed_3]), dim=1)

        # first residual block pass
        x_hat = self.res_block_1(x)
        x = x + x_hat

        # second residual block pass
        x_hat = self.res_block_2(x)
        x = x + x_hat
        
        # third residual block pass
        x_hat = self.res_block_3(x)
        x = x + x_hat
        
        # fourth residual block pass
        x_hat = self.res_block_4(x)
        x = x + x_hat
        
        # final output generation
        x = self.final_relu(x)
        x = self.final_batch_norm(x)
        x = self.final_reflective_padding(x)
        x = self.final_conv_layer(x)

        # x = self.final_linear(x.view(orig_batch_num, -1))

        return x, torch.mean(x, dim=2) # x is used for upstream task and torch.mean(x, dim=1) for downstream

class SeqCLRDecoder(torch.nn.Module):
    """
    see Figure 4.C of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, num_upstream_decode_features=32):
        super(SeqCLRDecoder, self).__init__()
        self.TEMPORAL_DIM = 2
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1

        self.out_features_256 = 32 # 256
        self.out_features_128 = 16 # 128
        self.out_features_64 = 8 # 64
        self.bdlstm_1 = torch.nn.LSTM(num_channels, self.out_features_256, bidirectional=True)
        self.bdlstm_2 = torch.nn.LSTM(num_channels, self.out_features_128, bidirectional=True)
        self.bdlstm_3 = torch.nn.LSTM(num_channels, self.out_features_64, bidirectional=True)

        self.x_linear_1 = torch.nn.Linear(2*2*self.out_features_256+2*2*self.out_features_128+2*2*self.out_features_64, self.out_features_128)
        self.x_relu_1 = torch.nn.ReLU()
        # self.h_linear_1 = torch.nn.Linear(2*2*256+2*2*128+2*2*64, 128)
        # self.h_relu_1 = torch.nn.ReLU()
        # self.c_linear_1 = torch.nn.Linear(2*2*256+2*2*128+2*2*64, 128)
        # self.c_relu_1 = torch.nn.ReLU()

        self.final_linear = torch.nn.Linear(self.out_features_128, num_upstream_decode_features)

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.num_upstream_decode_features = num_upstream_decode_features
    
    def forward(self, x):
        # print("SeqCLRDecoder.forward: x shape == ", x.shape)
        orig_temporal_len = x.size(self.TEMPORAL_DIM)
        orig_batch_num = x.size(self.BATCH_DIM)
        
        # prepare x for processing
        # see https://discuss.pytorch.org/t/how-to-downsample-a-time-series/78485
        # and https://discuss.pytorch.org/t/f-interpolate-weird-behaviour/36088
        x_down_1 = torch.nn.functional.interpolate(x, size=(x.size(self.TEMPORAL_DIM)//2))
        # print("SeqCLRDecoder.forward: x_down_1 shape == ", x_down_1.shape)
        intermed_temporal_len = x_down_1.size(self.TEMPORAL_DIM)
        x_down_2 = torch.nn.functional.interpolate(x_down_1, size=(x_down_1.size(self.TEMPORAL_DIM)//2))
        # print("SeqCLRDecoder.forward: x_down_2 shape == ", x_down_2.shape)
        
        x = x.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x shape == ", x.shape)
        x_down_1 = x_down_1.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x_down_1 shape == ", x_down_1.shape)
        x_down_2 = x_down_2.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x_down_2 shape == ", x_down_2.shape)

        # embed x using various gru modules
        x_embed_1, (h_1, c_1) = self.bdlstm_1(x) # TO-DO: do we need to remember this hidden, or not?
        x_embed_2, (h_2, c_2) = self.bdlstm_2(x_down_1)# TO-DO: do we need to remember this hidden, or not?
        x_embed_3, (h_3, c_3) = self.bdlstm_3(x_down_2)# TO-DO: do we need to remember this hidden, or not?

        x_embed_1 = x_embed_1.permute(1, 0, 2)
        x_embed_2 = x_embed_2.permute(1, 0, 2)
        x_embed_3 = x_embed_3.permute(1, 0, 2)

        # combine embeddings
        x = torch.cat(tuple([x_embed_1[:,0,:], 
                             x_embed_1[:,-1,:], 
                             x_embed_2[:,0,:], 
                             x_embed_2[:,-1,:], 
                             x_embed_3[:,0,:], 
                             x_embed_3[:,-1,:]]), 
                      dim=1
        )
        # print("x shape == ", x.shape)
        # h = torch.cat(tuple([h_1, h_2, h_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?
        # c = torch.cat(tuple([c_1, c_2, c_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?

        x = self.x_linear_1(x)
        # h = self.h_linear_1(h)
        # c = self.c_linear_1(c)
        x = self.x_relu_1(x)
        # h = self.h_relu_1(h)
        # c = self.c_relu_1(c)
        
        # final output generation
        out = self.final_linear(x) # TO-DO: should I include the final h & c in this as well (via concatenation)??? if so, need to change final_linear and preceding code in forward pass
        # raise NotImplementedError()
        return out

class SQNet(torch.nn.Module):
    """
    see proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
    """
    def __init__(self, encoder_type, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100, num_upstream_decode_features=32):
        super(SQNet, self).__init__()

        if encoder_type == "recurrent":
            self.embed_model = SeqCLRRecurrentEncoder(num_channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = SeqCLRDecoder(embed_dim, 
                                     temporal_len, 
                                     dropout_rate=dropout_rate, 
                                     num_upstream_decode_features=num_upstream_decode_features
            )
        elif encoder_type == "convolutional":
            self.embed_model = SeqCLRConvolutionalEncoder(num_channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = SeqCLRDecoder(embed_dim, 
                                     temporal_len, 
                                     dropout_rate=dropout_rate, 
                                     num_upstream_decode_features=num_upstream_decode_features
            )
        elif encoder_type == "simplified":
            self.embed_model = PhaseSwapFCN(num_channels, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = PhaseSwapUpstreamDecoder(embed_dim, temporal_len, dropout_rate=dropout_rate, decode_dim=num_upstream_decode_features)
        else:
            raise ValueError("encoder_type "+str(encoder_type)+" not supported")

        self.encoder_type = encoder_type
        pass
    
    def forward(self, x): 
        x, _ = self.embed_model(x)
        x = self.decode_model(x)
        return x


class PhaseSwapFCN(torch.nn.Module):
    """
    See Section 3 of arxiv.org/pdf/2009.07664.pdf for description and
    see Figure 1.b in arxiv.org/pdf/1611.06455.pdf for most of diagram
    """
    def __init__(self, num_channels, dropout_rate=0.5, embed_dim=100):
        super(PhaseSwapFCN, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 2
        self.TEMPORAL_DIM = 1

        self.K8 = 8
        self.K5 = 5
        self.K3 = 3

        self.D_INTERNAL_128 = 128
        self.D_INTERNAL_256 = 256
        self.D_OUT = embed_dim

        # self.conv_block_1 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K128//2)-1, self.K128//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K128)
        # )
        # self.conv_block_2 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K64)
        # )
        # self.conv_block_3 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K16//2)-1, self.K16//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_50, self.K16)
        # )

        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K8//2)-1, self.K8//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_128, self.K8), 
            torch.nn.BatchNorm1d(self.D_INTERNAL_128, track_running_stats=False), 
            torch.nn.ReLU()
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K5//2), self.K5//2)), 
            torch.nn.Conv1d(self.D_INTERNAL_128, self.D_INTERNAL_256, self.K5), 
            torch.nn.BatchNorm1d(self.D_INTERNAL_256, track_running_stats=False), 
            torch.nn.ReLU()
        )
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K3//2), self.K3//2)), 
            torch.nn.Conv1d(self.D_INTERNAL_256, self.D_OUT, self.K3), # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
            torch.nn.BatchNorm1d(self.D_OUT, track_running_stats=False), # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
            torch.nn.ReLU()
        )

        self.avg_pool = torch.nn.AvgPool1d(self.D_OUT)#, padding=self.D_INTERNAL_128//2) # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
        pass
    
    def forward(self, x):
        # print("\nPhaseSwapFCN: x.shape == ", x.shape)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_1(x)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_2(x)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_3(x)
        # print("PhaseSwapFCN: x_resid.shape == ", x.shape)
        x = self.avg_pool(x)
        # print("PhaseSwapFCN: out.shape == ", x.shape)
        return x, torch.mean(x, dim=2)#self.TEMPORAL_DIM) # x is used for upstream task and torch.mean(x, dim=2) for downstream

class PhaseSwapUpstreamDecoder(torch.nn.Module):
    """
    See Section 3 of arxiv.org/pdf/2009.07664.pdf for description
    """
    def __init__(self, hidden_dim, temporal_len, dropout_rate=0.5, decode_dim=1):
        super(PhaseSwapUpstreamDecoder, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1
        self.TEMPORAL_DIM = 2
        
        self.linear = torch.nn.Linear(hidden_dim*(temporal_len//hidden_dim), decode_dim)
        pass
    
    def forward(self, x):
        # print("\nPhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        x = x.view(x.size(0), -1)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        x = self.linear(x)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        # x = torch.nn.functional.softmax(x)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        return x

class PSNet(torch.nn.Module):
    """
    see arxiv.org/pdf/2009.07664.pdf
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(PSNet, self).__init__()

        self.embed_model = PhaseSwapFCN(num_channels, 
                                         dropout_rate=0.5, 
                                         embed_dim=embed_dim
        )

        self.decode_model = PhaseSwapUpstreamDecoder(embed_dim, 
                                                     temporal_len, 
                                                     dropout_rate=0.5
        )
        pass
    
    def forward(self, x): 
        x, _ = self.embed_model(x) # The second output is ignored because it is meant for the downstream task
        x = self.decode_model(x)
        return x