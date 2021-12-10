# Codes borrowed and modified from Stanford CS224W colabs

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

    
class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = Linear(self.in_channels,self.out_channels, bias=bias)
        self.lin_r = Linear(self.in_channels,self.out_channels, bias=bias)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        """"""
        x_agg = self.propagate(edge_index, x=(x,x), size=size)
        out = self.lin_l(x)+self.lin_r(x_agg)
        if self.normalize:
            out = torch.nn.functional.normalize(out) 

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size = None):
        # The axis along which to index number of nodes.
        node_dim = self.node_dim #I'm not sure where this is coming from
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean') 
        return out
      
      
      
# class DMoN(torch.nn.Module):
#   """Implementation of Deep Modularity Network (DMoN) layer.
#   Deep Modularity Network (DMoN) layer implementation as presented in
#   "Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer (we
#   modify it her to PyTorch).
#   DMoN optimizes modularity clustering objective in a fully unsupervised mode,
#   however, this implementation can also be used as a regularizer in a supervised
#   graph neural network. Optionally, it does graph unpooling.
#   Attributes:
#     num_clusters: Number of clusters in the model.
#     collapse_regularization: Collapse regularization weight.
#     dropout: Dropout rate. Note that the dropout in applied to the
#       intermediate representations before the softmax.
#     do_unpooling: Parameter controlling whether to perform unpooling of the
#       features with respect to their soft clusters. If true, shape of the input
#       is preserved.
#   """

#   def __init__(self,
#                in_channels,
#                collapse_regularization = 0.1,
#                args,
#                do_unpooling = False):
#     """Initializes the layer with specified parameters."""
#     super(DMoN, self).__init__()
#     self.in_channels = in_channels
#     self.num_clusters = args.num_clusters
#     self.collapse_regularization = collapse_regularization
#     self.dropout = args.dropout
#     self.do_unpooling = do_unpooling

#   def build(self, in_channels):
#     """Builds the Keras model according to the input shape."""
    
#     #self.transform = tf.keras.models.Sequential([
#     #    tf.keras.layers.Dense(
#     #        self.num_clusters,
#     #        kernel_initializer='orthogonal',
#     #        bias_initializer='zeros'),
#     #    tf.keras.layers.Dropout(self.dropout)
#     #])
   
#     self.transfrom = nn.Sequential([
#       nn.Linear(in_channels, self.num_clusters, bias=True),
#       torch.nn.Dropout(p=self.dropout)
#       ])

#     super(DMoN, self).build(in_channels)

#   def call(
#       self, inputs):
#     """Performs DMoN clustering according to input features and input graph.
#     Args:
#       inputs: A tuple of Torch tensors. First element is (n*d) node feature
#         matrix and the second one is (n*n) sparse graph adjacency matrix.
#     Returns:
#       A tuple (features, clusters) with (k*d) cluster representations and
#       (n*k) cluster assignment matrix, where k is the number of cluster,
#       d is the dimensionality of the input, and n is the number of nodes in the
#       input graph. If do_unpooling is True, returns (n*d) node representations
#       instead of cluster representations.
#     """
#     features, adjacency = inputs

    
#     assert isinstance(features, Tensor)
#     assert len(features.shape) == 2
#     assert len(adjacency.shape) == 2
#     assert features.shape[0] == adjacency.shape[0]

#     assignments = torch.nn.Softmax(self.transform(features), dim=1)
#     cluster_sizes = torch.sum(assignments, dim=0).item()  # Size [k]. Need to get float from tensor, item() should work...
#     assignments_pooling = assignments / cluster_sizes  # Size [n, k].

    
#     features_pooled = torch.mm(torch.transpose(assignments_pooling), features)
#     features_pooled = torch.nn.SELU(features_pooled)
#     if self.do_unpooling:
#       features_pooled = torch.mm(assignments_pooling, features_pooled)
#     return features_pooled, assignments
