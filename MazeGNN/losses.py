import torch
import math
import numpy as np

def kmeansloss(cluster_assignment, coords):
    loss_d=0
    cluster_size = torch.sum(cluster_assignment,axis=0).reshape(cluster_assignment.shape[1],1)
    centroids = torch.div(torch.matmul(cluster_assignment.t(),coords), cluster_size)
    num_nodes = cluster_assignment.shape[0]
    num_clusters = cluster_assignment.shape[1]
#    print('cluster size', cluster_size)
    #nums_k=[]
    mx, _ =torch.max(cluster_assignment,dim=-1)
    mn, _ =torch.min(cluster_assignment,dim=-1)
    print('max confi', torch.mean(mx), 'min conf',  torch.mean(mn))
#     centroids=[]
    for k in range(num_clusters):
        xi = torch.masked_select(coords[:,0], cluster_assignment.max(axis=1).indices==k) 
        yi = torch.masked_select(coords[:,1], cluster_assignment.max(axis=1).indices==k)
        coords_in_cluster = torch.concat((xi[:, None],yi[:, None]),axis=-1)
#         centroids.append(torch.mean(coords_in_cluster,axis=0))
#     print(centroids)    
#     for k in range(num_clusters):
   #     print(coords.shape, cluster_assignment.shape)
#        coords_in_cluster = torch.mul(coords, cluster_assignment[:,k].reshape(cluster_assignment.shape[0],1))
        dist_within = torch.sum((coords_in_cluster-centroids[k])**2,axis=-1) #size: number of nodes in the cluster
        loss_mean_dist2 = torch.mean(dist_within) # mean squared distane within each cluster
        if math.isnan(loss_mean_dist2):
            loss_mean_dist2=0
        loss_d+=loss_mean_dist2
        #hnums_k.append(len(xi))
    reg_assn = -torch.mean(torch.sum(torch.abs(num_clusters*cluster_assignment-1),axis=-1)) # penalize when Pc are uniform
    reg_num = num_nodes*torch.sum(1/torch.sum(cluster_assignment, axis=0))
    print('ass', reg_assn, 'reg_num', reg_num)
    #reg_num = -torch.sum(torch.log10(torch.sum(cluster_assignment.max(axis=1).indices==k)+1)) #regularization on number of nodes in a cluster
    #reg_assn = -torch.mean(torch.sum(torch.log10(cluster_assignment),axis=-1))
    #loss = loss+0.1*reg_assn
    #print('loss_d', loss_d, 'reg_num', reg_num)
    loss = loss_d+reg_assn+reg_num
    return loss   