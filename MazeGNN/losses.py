import torch

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

def dmonloss(assignments, adjacency, cluster_sizes, collapse_regularization)
    number_of_nodes = adjacency.shape[1]
    number_of_edges = torch.sum(degrees).item()
    cluster_sizes = torch.sum(assignments, dim=0).item()  # Size [k].
    graph_pooled = torch.transpose(torch.sparse.mm(adjacency, assignments))
    graph_pooled = torch.mm(graph_pooled, assignments)
 
    degrees =torch.sparse.sum(adjacency, axis=0)  # Size [n].
    degrees = torch.reshape(degrees, (-1, 1))
    
    spectral_loss = (-torch.trace(graph_pooled -
                                     normalizer) / 2 / number_of_edges).item()
    collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * torch.sqrt(
        float(self.num_clusters)).item() - 1
    loss = spectral_loss + collapse_regularization * collapse_loss
    return loss
