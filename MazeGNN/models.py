import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric import nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim)) 
        
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return F.softmax(x, dim=1)

    def loss(self, cluster_assignment, coords):
        loss=0
        cluster_size = torch.sum(cluster_assignment,axis=0).reshape(cluster_assignment.shape[1],1)
        centroids = torch.div(torch.matmul(cluster_assignment.t(),coords), cluster_size)
#        print()
#         print('centro',centroids.shape, centroids)
        num_sum=0
        for k in range(cluster_assignment.shape[1]):
            xi = torch.masked_select(coords[:,0], cluster_assignment.max(axis=1).indices==k) #torch.masked_select(coords[:,0], cluster_assignment.ge(0.5)[:,k]) #none of the element is bigger than 0.5 initially
            yi = torch.masked_select(coords[:,1], cluster_assignment.max(axis=1).indices==k)
            coords_in_cluster = torch.concat((xi[:, None],yi[:, None]),axis=-1)
#             print(k,coords_in_cluster.shape)
#             print(k, centroids[k].shape)
#             print(k, (coords_in_cluster-centroids[k])**2)
            dist = torch.sum((coords_in_cluster-centroids[k])**2,axis=-1)
#            print('dist',dist.shape)
            loss_k=torch.mean(dist) # mean squared distane within each cluster
#            print('loss cal',k,loss_k, 'num_nodes in cluster', len(coords_in_cluster))
            num_sum+=len(coords_in_cluster)
            loss+=loss_k
#        print('ca', cluster_assignment.shape)
#        print(num_sum)    
       
        return loss   