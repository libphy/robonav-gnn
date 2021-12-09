import torch
import torch_geometric.transforms as T


def kmeansloss(cluster_assignment, coords):
    loss=0
    cluster_size = torch.sum(cluster_assignment,axis=0).reshape(cluster_assignment.shape[1],1)
    centroids = torch.div(torch.matmul(cluster_assignment.t(),coords), cluster_size)
    for k in range(cluster_assignment.shape[1]):
        xi = torch.masked_select(coords[:,0], cluster_assignment.max(axis=1).indices==k) 
        yi = torch.masked_select(coords[:,1], cluster_assignment.max(axis=1).indices==k)
        coords_in_cluster = torch.concat((xi[:, None],yi[:, None]),axis=-1)
        dist = torch.sum((coords_in_cluster-centroids[k])**2,axis=-1)
        loss_k=torch.mean(dist) # mean squared distance within each cluster
        loss+=loss_k
    return loss

def dmonloss(assignments, edge_index, collapse_regularization)
    
    number_of_edges = edge_index.shape[1]//2
    number_of_nodes = assignments.shape[0]
    adjacency = torch.zeros(number_of_nodes, number_of_nodes)
    for i in range(number_of_edges):
        adjacency[edge_index[0][i], edge_index[1][i] += 1
    adjacency = adjacency//2
    adj = adjacency.to_sparse()
    
    
    degrees = torch.sparse.sum(adjacency, axis=0)  # Size [n].
    degrees = torch.reshape(degrees, (-1, 1))
    
    number_of_edges = torch.sum(degrees).item()
    cluster_sizes = torch.sum(assignments, dim=0).item()  # Size [k].
    graph_pooled = torch.transpose(torch.sparse.mm(adjacency, assignments))
    graph_pooled = torch.mm(graph_pooled, assignments)
                  
    # We compute the rank-1 normaizer matrix S^T*d*d^T*S efficiently
    # in three matrix multiplications by first processing the left part S^T*d
    # and then multyplying it by the right part d^T*S.
    # Left part is [k, 1] tensor.
    normalizer_left = torch.mm(torch.transpose(assignments), degrees)
    # Right part is [1, k] tensor.
    normalizer_right = torch.mm(torch.transpose(degrees), assignments)

    # Normalizer is rank-1 correction for degree distribution for degrees of the
    # nodes in the original graph, casted to the pooled graph.
    normalizer = torch.mm(normalizer_left,
                           normalizer_right) / 2 / number_of_edges
 
    
    
    spectral_loss = (-torch.trace(graph_pooled -
                                     normalizer) / 2 / number_of_edges).item()
    collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(
        float(self.num_clusters)).item() - 1
    loss = spectral_loss + collapse_regularization * collapse_loss
    return loss
