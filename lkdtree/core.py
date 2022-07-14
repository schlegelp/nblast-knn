import numpy as np

try:
    from fastremap import unique
except ImportError:
    from numpy import unique

class KDTree:
    """A vanilla KDTree for comparison.

    Since this is for demonstration purpose, it only supports 3 dimensions and
    k=1, and has a fixed leaf size of 1.

    Parameters
    ----------
    points :    (N, 3) array

    """
    def __init__(self, points):
        self.points = points

        assert points.ndim == 2
        assert points.shape[1] == 3

        # Build tree
        self.build()

    def __len__(self):
        """Number of points in the tree."""
        return len(self.points)

    def build(self):
        """Build the tree."""
        self.n_nodes = 0
        self.root = self._divide_tree(np.arange(len(self)))

    def _divide_tree(self, ind, i=0):
        # Track max depth
        self.depth = i
        self.n_nodes += 1

        # If only one node left, this is a leaf node
        if len(ind) <= 1:
            node = LeafNode()
            node.ind = ind[0]
            node.co = self.points[node.ind]
        else:
            node = Node()
            # Get the median for this set of points
            co = self.points[ind, i % 3]
            node.cutax = i % 3
            node.cutval = np.median(co)  # this is the bottleneck during build

            # We have to make sure that the indices split into two approx
            # evenly sized batches. Under certain circumstances the median
            # doesn't work though. For example: if co=[3,4,4], the median will be 4
            # and the split would be left=[3,4,4] and right=[].
            left_ind = ind[co <= node.cutval]
            right_ind = ind[co > node.cutval]

            if not len(left_ind) or not len(right_ind):
                node.cutval = min(co)
                if min(co) != max(co):
                    left_ind = ind[co <= node.cutval]
                    right_ind = ind[co > node.cutval]
                else:
                    left_ind = ind[:1]
                    right_ind = ind[1:]

            # Split into left and right
            node.left = self._divide_tree(left_ind, i + 1)
            node.right = self._divide_tree(right_ind, i + 1)

            node.left.parent = node.right.parent = node

        # Track the level (for debugging only)
        node.level = i

        return node

    def query(self, point):
        """Find nearest-neighbour for given point.

        Parameters
        ----------
        point :     (x, y, z) tuple

        Returns
        -------
        dist :      int
                    Squared distance.
        ind :       int
                    Index of the node.

        """
        # Make sure point is an array
        point = np.asarray(point)

        # Keep track of how many nodes we had to search
        # (mostly for benchmarking)
        self._nodes_visited = 0

        # Depth-first search for the closest leaf
        leaf, dist = self._forward_search(self.root, point)

        # Now move back and check previous branches to make sure that
        # (A) we have the actual nearest-neighbour
        # (B) collect neighbours for the other labels
        node, dist = self._reverse_search(leaf, point, dist, leaf)

        return dist, node.ind

    def _forward_search(self, node, point):
        self._nodes_visited += 1
        # If this is a leaf node, return the node and the distance
        if isinstance(node, LeafNode):
            dist = ((point - node.co) ** 2).sum()
            return node, dist

        # If this is a branch point figure out which child to go for
        if point[node.cutax] <= node.cutval:
            return self._forward_search(node.left, point)
        else:
            return self._forward_search(node.right, point)

    def _reverse_search(self, node, point, dist_min, node_min):
        self._nodes_visited += 1

        # Get this node's parent
        p = node.parent

        # Check if we need to walk down the other child
        if (point[p.cutax] - p.cutval)**2 < dist_min:
            other = p.left if p.left != node else p.right
            new_node_min, new_dist_min = self._forward_search(other, point)

            if new_dist_min < dist_min:
                dist_min = new_dist_min
                node_min = new_node_min

        # If we're at the root our search should be complete
        if p == self.root:
            return node_min, dist_min

        # Else keep tracking back
        return self._reverse_search(p, point, dist_min, node_min)


class LKDTree(KDTree):
    """Labelled version of the KDTree.

    Parameters
    ----------
    points :    (N, 3) array
    labels :    (N, ) array
                This is assumed to be contiguous integers ranging from 0 to
                max(labels).

    """
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels

        assert points.ndim == 2
        assert points.shape[1] == 3
        assert len(points) == len(labels)

        # Get unique labels
        self._unique_labels = unique(labels)

        # Build tree
        self.build()

    def query(self, point):
        """Find nearest-neighbours for given point.

        Parameters
        ----------
        point :     (x, y, z) tuple

        Returns
        -------
        dist :      np.array
                    (N, ) array of squared distance where N is `max(labels)`.
        ind :       np.array
                    (N, ) array of node indices where N is `max(labels)`.

        """
        # Make sure point is an array
        point = np.asarray(point)

        # Prepare arrays of distances and nodes for each label
        dists = np.full(self._unique_labels.max() + 1, np.inf)
        nodes = np.full(self._unique_labels.max() + 1, None)

        # Keep track of how many nodes we had to search
        # (mostly for benchmarking)
        self._nodes_visited = 0

        # Depth-first search for the closest leaf - doesn't matter which
        # label that leaf has
        leaf, dist = self._forward_search_single(self.root, point)

        # Track distance and node for this first leaf
        dists[self.labels[leaf.ind]] = dist
        nodes[self.labels[leaf.ind]] = leaf

        # Now move back and check previous branches to make sure that
        # (A) we have the actual nearest-neighbour
        # (B) collect neighbours for the other labels
        self._reverse_search(leaf, point, dists, nodes)

        return dists, np.array([n.ind for n in nodes])

    def _divide_tree(self, ind, i=0):
        # If only one node left, this is a leaf node
        node = super()._divide_tree(ind, i=i)

        # Track the labels in this node
        node.labels = np.unique(self.labels[ind])

        return node

    def _forward_search_single(self, node, point):
        """This function simply finds the closest leaf node."""
        # If this is a leaf node, return the node and the distance
        self._nodes_visited += 1
        if isinstance(node, LeafNode):
            dist = ((point - node.co) ** 2).sum()
            return node, dist

        # If this is a branch point figure out which child to go down
        if point[node.cutax] <= node.cutval:
            return self._forward_search_single(node.left, point)
        else:
            return self._forward_search_single(node.right, point)

    def _forward_search_multi(self, node, point, dists_min, nodes_min):
        self._nodes_visited += 1
        # If this is a leaf node and the new distance is closer than the old
        if isinstance(node, LeafNode):
            dist = ((point - node.co) ** 2).sum()
            if dist < dists_min[node.labels[0]]:
                dists_min[node.labels[0]] = dist
                nodes_min[node.labels[0]] = node
            return

        # If this is a branchpoint figure out which childs we need to traverse
        for ch in (node.left, node.right):
            search = False
            # For each label in this branch check if it warrants going down this branch
            for i in ch.labels:
                # If we haven't found a node for this label yet
                if not nodes_min[i]:
                    search = True
                    break

                # If the child is a leaf node with a label we're looking for
                # just go for it
                if isinstance(ch, LeafNode):
                    search = True
                    break

            # If we have a node for this label check if there is the chance
            # for a closer match
            if not search:
                d = (point[ch.cutax] - ch.cutval)**2
                for i in ch.labels:
                    if d < dists_min[i]:
                        search = True
                        break

            if search:
                self._forward_search_multi(ch, point, dists_min, nodes_min)

    def _reverse_search(self, node, point, dists_min, nodes_min):
        self._nodes_visited += 1
        # Get this node's parent
        p = node.parent

        # Get the branch we hadn't traversed yet
        other = p.left if p.left != node else p.right

        search = False
        # For each label in thise node check if it warrants going down the branch
        for i in node.labels:
            # If we haven't found a node for this label yet
            if not nodes_min[i]:
                search = True
                break

        # If we have a node for this label check if there is the chance for
        # a closer match
        if not search:
            d = (point[p.cutax] - p.cutval)**2
            for i in node.labels:
                if d < dists_min[i]:
                    search = True
                    break

        if search:
            self._forward_search_multi(other, point, dists_min, nodes_min)

        # If we're at the root our search should be complete
        if p == self.root:
            return

        # Else keep tracking back
        return self._reverse_search(p, point, dists_min, nodes_min)


class Node:
    def __str__(self):
        return f'Node <labels={self.labels.tolist()}, cutval={self.cutval}, cutax={self.cutax}>'


class LeafNode(Node):
    def __str__(self):
        return f'LeafNode <label={self.labels[0]}, co={self.co}>'
