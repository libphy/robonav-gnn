import torch
import math
import numpy as np
import torch_geometric.transforms as T

def kmeansloss(cluster_assignment, coords):
    loss_d=0
    loss_d_neg=0
    loss_invd_neg=0
    loss_inter=0
    cluster_size = torch.sum(cluster_assignment,axis=0).reshape(cluster_assignment.shape[1],1)
    centroids = torch.div(torch.matmul(cluster_assignment.t(),coords), cluster_size)
    num_nodes = cluster_assignment.shape[0]
    num_clusters = cluster_assignment.shape[1]

    for k in range(num_clusters):
        xi = torch.masked_select(coords[:,0], cluster_assignment.max(axis=1).indices==k) 
        yi = torch.masked_select(coords[:,1], cluster_assignment.max(axis=1).indices==k)
        xi_neg = torch.masked_select(coords[:,0], cluster_assignment.max(axis=1).indices!=k) 
        yi_neg = torch.masked_select(coords[:,1], cluster_assignment.max(axis=1).indices!=k)        
        coords_in_cluster = torch.concat((xi[:, None],yi[:, None]),axis=-1)
        coords_out_cluster = torch.concat((xi_neg[:, None],yi_neg[:, None]),axis=-1)

        dist_within = torch.sum((coords_in_cluster-centroids[k])**2,axis=-1) #size: number of nodes in the cluster
        dist_neg = torch.sum((coords_out_cluster-centroids[k])**2,axis=-1)
        loss_mean_dist2 = torch.mean(dist_within) # mean squared distane within each cluster
        loss_mean_dist2_neg = torch.mean(dist_neg) # mean squared distance for negative samples
        loss_mean_inv_d_neg = torch.mean(1/(dist_neg+0.01))
        if math.isnan(loss_mean_dist2):
            loss_mean_dist2=0
        loss_d+=loss_mean_dist2
        loss_d_neg+=loss_mean_dist2_neg
        loss_invd_neg+=loss_mean_inv_d_neg
        
        if len(xi)>1:
            if len(xi)<1000:
                inds = torch.randint(len(xi), (1000,))
            else:
                inds = torch.randperm(len(xi))[:1000]
            inter_x = torch.cartesian_prod(xi[inds], xi[inds]) #too expensive to calculate when there are many nodes, so we sample 1000.
            inter_y = torch.cartesian_prod(yi[inds], yi[inds])
            loss_inter += torch.mean((inter_x[:,0]-inter_x[:,1])**2+(inter_y[:,0]-inter_y[:,1])**2)
        
    reg_assn = -torch.mean(torch.sum(torch.abs(num_clusters*cluster_assignment-1),axis=-1)) # penalize when assign probs are uniform
    reg_num = num_nodes*torch.sum(1/torch.sum(cluster_assignment, axis=0)) # penalize when one cluster has all nodes

    loss = loss_d-loss_d_neg+0.1*loss_invd_neg+0.1*reg_assn+reg_num +5*loss_inter
    print('loss', loss.item(), 'd_pos', loss_d.item(), 'd_neg', loss_d_neg.item(), 'loss_invd_neg', loss_invd_neg.item(), 'reg_assn', reg_assn.item(), 'reg_num', reg_num.item(), 'loss_inter', loss_inter.item())
    return loss   
