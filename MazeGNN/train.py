# Codes borrowed and modified from Stanford CS224W colabs
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import matplotlib.pyplot as plt
import time

from utils import *
from datagen import *
from losses import *
from models import *

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def train(dataset, args):
    test_loader = loader = DataLoader(dataset, batch_size=32, shuffle=True) #somehow it does not batch correctly

    # build model
    model = GNNStack(args.node_feat_dim, args.hidden_dim, args.num_clusters, args) 
    model.loss = args.lossft
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_loss = []
    best_loss = 1. #0 choose carefully to something possible
    best_model =  copy.deepcopy(model) 
    t0=time.perf_counter()
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            loss = model.loss(pred[batch.train_mask],batch.x[batch.train_mask])
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)
        
        if epoch % 10 == 0:
            if args.visualize:
                visualize(pred, batch, batch.train_mask)
            if args.test:    
                loss_test = test(test_loader, model)
                test_loss.append(loss_test)
                t1=time.perf_counter()
                print('ep:',epoch, ', train loss', loss.item(), ', test loss:', loss_test, t1-t0, 's.')
                if loss_test < best_loss:
                    best_loss = loss_test
                    best_model = copy.deepcopy(model)
                else:
                    best_model = copy.deepcopy(model)    #update anyway for now
                if args.visualize:
                    pred_test = best_model(batch)
                    visualize(pred_test, batch, batch.test_mask)
        else:
            if args.test:
                test_loss.append(test_loss[-1])
        
    if not args.test:
        best_model = copy.deepcopy(model) #just return the final trained model if there is no test
            
    return test_loss, losses, best_model, best_loss, test_loader

def test(loader, test_model, is_validation=False, save_model_preds=False, model_type=None):
    test_model.eval()
    total_loss=0
    for batch in loader:
        with torch.no_grad():
            pred = test_model(batch)
        mask = batch.val_mask if is_validation else batch.test_mask
        loss = test_model.loss(pred[mask],batch.x[mask])
        total_loss += loss.item() * batch.num_graphs
    total_loss /= len(loader.dataset)
    return total_loss


def visualize(pred,data,mask, overlay=False):
    """
    pred: before masked
    data: data object
    mask: train or test mask
    """
    if type(mask)!=list:
        mask = list(mask)
    original_x = data.x*data.scaler+data.shift
    ## soft centroid
    cluster_size = torch.sum(pred[mask],axis=0).reshape(pred[mask].shape[1],1)
    centroids = torch.div(torch.matmul(pred[mask].t(),original_x[mask]), cluster_size).detach().numpy()
    colors=['r','orange','yellow','g','b','magenta','purple','turquoise']
    num_clusters = pred.shape[1]
    if num_clusters>=4:
        rows = num_clusters//4+1 if num_clusters%4!=0 else num_clusters//4
        cols = 4
        aspectratio = rows/cols
    else:
        rows=1
        cols=num_clusters

    if overlay:
        fig2 =plt.figure(figsize=(6, 6))
        
        for k in range(pred.shape[1]):
            xi = torch.masked_select(original_x[mask][:,0], pred[mask].max(axis=1).indices==k)
            yi = torch.masked_select(original_x[mask][:,1], pred[mask].max(axis=1).indices==k)
            ## hard centroid
            xm=xi.mean()
            ym=yi.mean()
            plt.plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
            plt.scatter(xm,ym, marker='*',c='k',s=30) 
            plt.scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20) #c=colors[k%len(colors)])
        plt.show()
        
    else:    
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(15, int(15*aspectratio)))

        for k in range(pred.shape[1]):
            xi = torch.masked_select(original_x[mask][:,0], pred[mask].max(axis=1).indices==k)
            yi = torch.masked_select(original_x[mask][:,1], pred[mask].max(axis=1).indices==k)
            ## hard centroid
            xm=xi.mean()
            ym=yi.mean()
            r = k//4
            c = k%4
            if num_clusters>4:
                axs[r,c].plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
                axs[r,c].scatter(xm,ym, marker='*',c='k',s=30) 
                axs[r,c].scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20) #c=colors[k%len(colors)])
                axs[r,c].set_title(str(k)+' ('+str(len(xi))+')')
            else:    
                axs[c].plot(xi,yi, marker='.', color=colors[k%len(colors)],linestyle="None", alpha=0.1)
                axs[c].scatter(xm,ym, marker='*',c='k',s=30) 
                axs[c].scatter(centroids[k,0],centroids[k,1], marker='o',c='k',s=20) #c=colors[k%len(colors)])
                axs[c].set_title(str(k)+' ('+str(len(xi))+')')

        plt.show()
    
