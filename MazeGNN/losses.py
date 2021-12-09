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
        loss_k=torch.mean(dist) # mean squared distane within each cluster
        loss+=loss_k
    return loss   