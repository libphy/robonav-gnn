from utils import *
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
import pandas as pd
import copy

class MazeDataset(InMemoryDataset):
    def __init__(self, data_obj, test_size=None, transform=None):
        super(MazeDataset, self).__init__(data_obj, transform, None, None)
        self.data_obj = copy.deepcopy(data_obj)
        
        self.data_obj.num_nodes = len(self.data_obj.x)
        #normalize x
        original_x = copy.deepcopy(data_obj.x)
        self.data_obj['scaler'] = original_x.max()-original_x.min()
        self.data_obj['shift'] = original_x.mean()
        self.data_obj.x = (original_x-self.data_obj['shift'])/self.data_obj['scaler']
        if test_size is not None:
            assert (test_size>0)&(test_size<1), 'Invalid test size. It needs to be 0 < test_size < 1.'
            # splitting the data into train, validation and test
            X_train, X_test = train_test_split(pd.Series(range(self.data_obj.num_nodes)), 
                                                                test_size=test_size, 
                                                                random_state=42)


            # create train and test masks for data
            train_mask = torch.zeros(self.data_obj.num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(self.data_obj.num_nodes, dtype=torch.bool)
            train_mask[X_train.index] = True
            test_mask[X_test.index] = True
            self.data_obj['train_mask'] = train_mask
            self.data_obj['test_mask'] = test_mask
        else:
            train_mask = torch.ones(self.data_obj.num_nodes, dtype=torch.bool)
            self.data_obj['train_mask'] = train_mask

        self.data, self.slices = self.collate([self.data_obj])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)    
    
def create_data(args, verbose=True):
    try:
        startlayer = args.startlayer
    except AttributeError:
        startlayer = 1
    try:
        goallayer = args.goallayer
    except AttributeError:
        goallayer = 2        
 
    maze_orig = image2maze(args.imgpath, start=args.start, goal=args.goal, startlayer=startlayer, goallayer=goallayer)
    mz = coarse_map(maze_orig,args.poolsize)
    node_coords = np.argwhere(1-mz[:,:,2])
    t0 = time.perf_counter()
    edges, edge_features = extract_edges_and_features_from_image(node_coords, verbose)
    t1 = time.perf_counter()
    if verbose: 
        print(t1-t0, 'Edges and edge features extracted from func:extract_edges_and_features_from_image.')
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.long)
    node_features = node_features = torch.tensor(node_coords, dtype=torch.float)  #here we use coordinate as node feature
    #node_degree = degree(edge_index.reshape(edge_index.shape[0]*2)) #node degree as an alternative node feature
    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_features = edge_features)

    return data
    
def create_dataset(args, verbose=False):
    data = create_data(args, verbose)
    try:
        test_size = args.test_size
    except AttributeError:
        test_size = None
        
    dataset = MazeDataset(data,test_size=test_size)
    return dataset

