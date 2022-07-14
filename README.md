# NBLAST-KNN
Playground for tinkering with improving KNN for NBLAST. 

## `lkdtree` 

The `lkdtree` module contains a pure-Python/numpy implementation of a "labelled" `LKDTree` where the nearest-neighbour
search will return the closest point for each unique labels in the tree. The idea is that we have to traverse less
branches if we do this in one go instead of having a single KDTree for each label.

It works in principle but it turns out to be much slower than using multiple separate vanilla KDTrees. It's particularly 
bad when the data is not dense (which is the case for neurons). See `lkdtree_benchmark.py` for examples and the benchmark.

## Other ideas to explore

1. Use the neuron's topology to save time searching the tree.

2. Check out "all-nearest-neighbours" 
