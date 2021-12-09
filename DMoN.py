
"""Deep Modularity Network (DMoN) Keras layer.
Deep Modularity Network (DMoN) layer implementation as presented in
"Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer.
DMoN optimizes modularity clustering objective in a fully unsupervised regime.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


import html
import time
import torch


class DMoN(torch.nn.Module):
  """Implementation of Deep Modularity Network (DMoN) layer.
  Deep Modularity Network (DMoN) layer implementation as presented in
  "Graph Clustering with Graph Neural Networks" in a form of TF 2.0 Keras layer.
  DMoN optimizes modularity clustering objective in a fully unsupervised mode,
  however, this implementation can also be used as a regularizer in a supervised
  graph neural network. Optionally, it does graph unpooling.
  Attributes:
    n_clusters: Number of clusters in the model.
    collapse_regularization: Collapse regularization weight.
    dropout_rate: Dropout rate. Note that the dropout in applied to the
      intermediate representations before the softmax.
    do_unpooling: Parameter controlling whether to perform unpooling of the
      features with respect to their soft clusters. If true, shape of the input
      is preserved.
  """

  def __init__(self,
               n_clusters,
               collapse_regularization = 0.1,
               dropout_rate = 0,
               do_unpooling = False):
    """Initializes the layer with specified parameters."""
    super(DMoN, self).__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.dropout_rate = dropout_rate
    self.do_unpooling = do_unpooling

  def build(self, input_shape):
    """Builds the Keras model according to the input shape."""
    
    #self.transform = tf.keras.models.Sequential([
    #    tf.keras.layers.Dense(
    #        self.n_clusters,
    #        kernel_initializer='orthogonal',
    #        bias_initializer='zeros'),
    #    tf.keras.layers.Dropout(self.dropout_rate)
    #])
   
    self.transfrom = nn.Sequential([
      nn.Linear(input_shape.shape[0], self.n_clusters, bias=True),
      torch.nn.Dropout(p=self.dropout_rate)
      ])

    super(DMoN, self).build(input_shape)

  def call(
      self, inputs):
    """Performs DMoN clustering according to input features and input graph.
    Args:
      inputs: A tuple of Torch tensors. First element is (n*d) node feature
        matrix and the second one is (n*n) sparse graph adjacency matrix.
    Returns:
      A tuple (features, clusters) with (k*d) cluster representations and
      (n*k) cluster assignment matrix, where k is the number of cluster,
      d is the dimensionality of the input, and n is the number of nodes in the
      input graph. If do_unpooling is True, returns (n*d) node representations
      instead of cluster representations.
    """
    features, adjacency = inputs

    
    assert isinstance(features, Tensor)
    assert isinstance(adjacency, SparseTensor) #not sure SparseTensor is correct here
    assert len(features.shape) == 2
    assert len(adjacency.shape) == 2
    assert features.shape[0] == adjacency.shape[0]

    assignments = torch.nn.Softmax(self.transform(features), dim=1)
    cluster_sizes = torch.sum(assignments, dim=0).item()  # Size [k]. Need to get float from tensor, item() should work...
    assignments_pooling = assignments / cluster_sizes  # Size [n, k].

    degrees = torch.sparse.sum(adjacency, dim=0)  # Size [n].
    degrees = torch.reshape(degrees, (-1, 1))

    number_of_nodes = adjacency.shape[1]
    number_of_edges = torch.sum(degrees).item() # check you get float here, not tensor

    # Computes the size [k, k] pooled graph as S^T*A*S in two multiplications.
    graph_pooled = torch.transpose(
        torch.sparse.mm(adjacency, assignments))
    graph_pooled = torch.mm(graph_pooled, assignments)

    # We compute the rank-1 normalizer matrix S^T*d*d^T*S efficiently
    # in three matrix multiplications by first processing the left part S^T*d
    # and then multyplying it by the right part d^T*S.
    # Left part is [k, 1] tensor.
    normalizer_left = torch.mm(torch.transpose(assignments, 0, 1), degrees)
    # Right part is [1, k] tensor.
    normalizer_right = torch.mm(torch.transpose(degrees, 0, 1), assignments)

    # Normalizer is rank-1 correction for degree distribution for degrees of the
    # nodes in the original graph, casted to the pooled graph.
    normalizer = torch.mm(normalizer_left,
                           normalizer_right) / 2 / number_of_edges
    spectral_loss = -torch.trace(graph_pooled -
                                     normalizer) / 2 / number_of_edges
    #self.add_loss(spectral_loss) #how to change?

    collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(
        float(self.n_clusters)) - 1
    spectral_loss += self.collapse_regularization * collapse_loss

    features_pooled = torch.mm(torch.transpose(assignments_pooling), features)
    features_pooled = torch.nn.SELU(features_pooled)
    if self.do_unpooling:
      features_pooled = torch.mm(assignments_pooling, features_pooled)
    return features_pooled, assignments
