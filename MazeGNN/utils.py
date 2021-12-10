import numpy as np
import matplotlib.pyplot as plt
import time
import copy

def maxpool(img, window_size=50):
    newrows = img.shape[0]//window_size+1
    newcols = img.shape[1]//window_size+1
    newimg = np.zeros((newrows,newcols))
    for row in range(newrows):
        for col in range(newcols):
            newimg[row,col] = np.max(img[row*window_size:(row+1)*window_size,col*window_size:(col+1)*window_size])
    return newimg      

def coarse_map(map_orig, poolsize=5):
    return np.concatenate([np.expand_dims(maxpool(map_orig[:,:,0],poolsize),-1),np.expand_dims(maxpool(map_orig[:,:,1],poolsize),-1),np.expand_dims(maxpool(map_orig[:,:,2],poolsize),-1)],axis=-1)

def image2maze(imgpath, start=None, goal=None, startlayer=1, goallayer=2):
    """
    imagepath:string
    start:tuple (start_row_min, start_row_max, start_col_min, start_col_max)
    goal:tuple (goal_row_min, goal_row_max, goal_col_min, goal_col_max)
    Returns augmented image [row x col x 3] with depth layers of [start region, goal region, map].
    """
    maze = plt.imread(imgpath).copy()
       
    if maze.shape[-1]==4: # For ppt-saved B/W images, the walllayer is 4th. 
        if (start != None)&(goal != None):
            maze[start[0]:start[1],start[2]:start[3],startlayer]=1 #start layer
            maze[goal[0]:goal[1],goal[2]:goal[3],goallayer]=1 #goal layer
            return maze[:,:,1:]
        else:
            print('Either the start or goal regions not specified, returning the image as is')
            return maze      
    else:
        print("Returing image read as is")
        return maze

def extract_edges_and_features_from_image(node_coords, verbose=True):
    """
    Input node_coords:np.array (coordinate of each nodes in the maze)
    Returns edges_out (edge index), edge_features_out (action tuple)
    """
    edges=dict({})
    edge_features=dict({})
    t0=time.perf_counter()
    for i in range(1,node_coords[:,0].max()+1): #row sweep
        indexes=np.argwhere(node_coords[:,0]==i).T[0] 
        for k in range(1,len(indexes)):
            if abs(node_coords[indexes[k],1]-node_coords[indexes[k-1],1])==1:
                if indexes[k-1] in edges.keys():
                    e = edges[indexes[k-1]]
                    e.append((indexes[k-1],indexes[k]))
                    f = edge_features[indexes[k-1]]
                    f.append((0,1)) # move to right
                    edges.update({indexes[k-1]:e})
                    edge_features.update({indexes[k-1]:f}) 
                else:
                    edges.update({indexes[k-1]:[(indexes[k-1],indexes[k])]})
                    edge_features.update({indexes[k-1]:[(0,1)]})
                if indexes[k] in edges.keys():
                    e = edges[indexes[k]]
                    e.append((indexes[k],indexes[k-1]))
                    f = edge_features[indexes[k]]
                    f.append((0,-1)) # move to left
                    edges.update({indexes[k]:e})
                    edge_features.update({indexes[k]:f}) 
                else:
                    edges.update({indexes[k]:[(indexes[k],indexes[k-1])]})   
                    edge_features.update({indexes[k]:[(0,-1)]}) 
                     
    for j in range(1,node_coords[:,1].max()+1): #col sweep
        indexes=np.argwhere(node_coords[:,1]==j).T[0]       
        for k in range(1,len(indexes)):
            if abs(node_coords[indexes[k],0]-node_coords[indexes[k-1],0])==1:
                if indexes[k-1] in edges.keys():
                    e = edges[indexes[k-1]]
                    e.append((indexes[k-1],indexes[k]))
                    f = edge_features[indexes[k-1]]
                    f.append((1,0)) # move down
                    edges.update({indexes[k-1]:e})
                    edge_features.update({indexes[k-1]:f}) 
                else:
                    edges.update({indexes[k-1]:[(indexes[k-1],indexes[k])]})
                    edge_features.update({indexes[k-1]:[(1,0)]})
                if indexes[k] in edges.keys():
                    e = edges[indexes[k]]
                    e.append((indexes[k],indexes[k-1]))
                    f = edge_features[indexes[k]]
                    f.append((-1,0)) # move up
                    edges.update({indexes[k]:e})
                    edge_features.update({indexes[k]:f}) 
                else:
                    edges.update({indexes[k]:[(indexes[k],indexes[k-1])]})    
                    edge_features.update({indexes[k]:[(-1,0)]}) 
                
    t1=time.perf_counter()
    if verbose:
        print('extract_edges_and_features_from_image: edges and edge_features dicts created',t1-t0)
        print('sanity checks...')
    #sanity checks
    assert len(edges)==len(edge_features), 'numbers of nodes in edge index and edge features do not match'
    num_nodes=len(edges)
    edges = dictsort(edges)
    edge_features = dictsort(edge_features)
    edges_out=[]
    edge_features_out=[]
    #print(len(edges))
    t0=time.perf_counter()
    for n in range(len(edges)):
        if verbose & (n%1000==0):
            t1=time.perf_counter()
            print(n,'/',num_nodes,t1-t0)
#         assert n==edges[n][0],'edge node index not match'
#         assert n==edge_features[n][0],'edge feature node index not match'
#         assert len(edges[n][1])==len(edge_features[n][1]),'number of edges and number of edge features do not match'
        edges_out.extend(edges[n][1])
        edge_features_out.extend(edge_features[n][1])
    t1=time.perf_counter()
    if verbose:
        print(t1-t0)
    return edges_out, edge_features_out

def dictsort(d):
    items = d.items()
    return sorted(items)    
    
    
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d