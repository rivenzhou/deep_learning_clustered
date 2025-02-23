import torch
import torch.nn as nn
def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device
def th_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)

class Cox_cluster(nn.Module):
    def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes):
        super(Cox_cluster, self).__init__()
        #self.pathway_mask = Pathway_Mask
        ###gene layer --> pathway layer
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.sigmoid=nn.Sigmoid()
        self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
        ###pathway layer --> hidden layer 1
        self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
        ###hidden layer --> hidden layer 2
        self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
        ###hidden layer 2 + cluster identification --> Cox layer
        self.sc4 = nn.Linear(Out_Nodes + 200, 1, bias=False)

        self.sc4.weight.data.uniform_(-0.001, 0.001)
        #self.sc5=torch.nn.parameter(torch.FloatTensor(Out_Nodes + 1000))
        ##self.sc4_weights = torch.nn.Parameter(torch.tensor(torch.zeros(Out_Nodes + 1000),
         #                                       dtype=self.sc4.weight.dtype))
        ###randomly select a small sub-network
        #self.do_m1 = torch.ones(Pathway_Nodes)
        #self.do_m2 = torch.ones(Hidden_Nodes)
        ###if gpu is being used
        #if torch.cuda.is_available():
           # self.do_m1 = self.do_m1.cuda()
           # self.do_m2 = self.do_m2.cuda()

    ###

    def forward(self, x1, x2):
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        #self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
        x1=tuple_of_tensors_to_tensor(x1)
        x2=tuple_of_tensors_to_tensor(x2)
        #x1= torch.from_numpy(x1)
        #x2 = torch.from_numpy(x2)
        #x2= df_to_tensor(x2)
       # print(x1)
        #print(x1.dtype)
        #print(x1.shape)
       # print(x2)
       # print(x2.dtype)
       # print(x2.shape)
        x1=self.relu(self.sc1(x1.float()))
        #x_1 = self.tanh(self.sc1(x_1))
        #if self.training == True:  ###construct a small sub-network for training only
           # x_1 = x_1.mul(self.do_m1)
        #x_1 = self.tanh(self.sc2(x_1))
        x1  =self.relu(self.sc2(x1))
        #if self.training == True:  ###construct a small sub-network for training only
         #   x_1 = x_1.mul(self.do_m2)
        x1 =self.relu(self.sc3(x1))
        #print(x1.shape)
        ###combine age with hidden layer 2
        x_cat = torch.cat((x1, x2), 2)
        #print(x_cat.shape)
        lin_pred = self.sc4(x_cat.float())
        #lin_predcompute_baseline_hazards().weight=nn.parameter(weight)
        #print(lin_pred.weight)
        #weights=th_delete(self.sc4.weight, [0])
        weights = th_delete(self.sc4.weight.view(-1), [0])
        #print(weights.shape)
        x2_mat=x2[-1, :, :]
       # print(x2_mat.numpy().dtype)
        weights=torch.mm(x2_mat.double(),weights.unsqueeze(1).double(), out=None)

       # print(weights.shape)
       # weights=self.sc4(x_cat)
        #print( self.sc4(x_cat))
        #weights= self.sc4_weights


        return lin_pred,weights