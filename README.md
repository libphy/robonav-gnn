# aMazeGNN

In this project, we apply a GNN model to cluster maze grids using k-means clustering.     
[**Try in colab**](https://colab.research.google.com/drive/1v-NJB2RWRwZTiBZEAsaRBPqPbEuFfOZF?usp=sharing)     
[**Read our blog post about this project**](https://medium.com/@paulwfalcone/f89529be3c2d)        


`utils.py` has various utils preprocessing images and other functions. The function `image2maze` assumes 4-channel image as input. Feel free to modify if you're using 1-channel or 3-channel images.    
`datagen.py` creates torch_geometric dataset and data objects from image.     
`models.py` include GNN models. (Currently it has GraphSAGE only. Feel free to add your own GNN layers)    
`losses.py` implements kmeans clustering using Euclidean distance. It includes additional regularization terms. For more details, please read our blog.    
`train.py` includes training, testing and visualization codes.    
Training examples can be found in the `\notebooks` and our [colab](https://colab.research.google.com/drive/1v-NJB2RWRwZTiBZEAsaRBPqPbEuFfOZF?usp=sharing) example.    
