import numpy as np
import pandas as pd
from typing import List
import sys

def get_eu_dist(arr1: List, arr2: List) -> float:
    """Calculate the Euclidean distance of two vectors.
    Arguments:
        arr1 {list} -- 1d list object with int or float
        arr2 {list} -- 1d list object with int or float
    Returns:
        float -- Euclidean distance
    """

    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5

class Node(object):
    def __init__(self, d, val, point, left, right):
        self.d = d
        self.val = val
        self.point = point
        self.left = left
        self.right = right

    def brother(self):
        """Find the node's brother.
        Returns:
            node -- Brother node.
        """
        if not self.parent:
            ret = None
        else:
            if self.parent.left is self:
                ret = self.parent.right
            else:
                ret = self.parent.left
        return ret

    def __str__(self):
        return f"point.x={self.point[0]}, y={self.point[1]}"

    def get_label(self):
        return self.point[1]


"""
Require: A set of points P of M dimensions and current depth D.
if P is empty then
    return null
else if P only has one data point then
    Create new node node
        node.d ← d
        node.val ← val
        node.point ← current point
        return node
else
    d ← D mod M
    val ← Median value along dimension among points in P.
    Create new node node.
        node.d ← d
        node.val ← val
        node.point ← point at the median along dimension d
        node.lef t ← BuildKdTree( points in P for which value at dimension d is less than or equal to val, D+1)
        node.right ← BuildKdTree( points in P for which value at dimension d is greater than val, D+1)
        return node
end if

"""
class KDTree(object):
    def __init__(self, _start_dim):
        self.root = None
        self.start_dim = _start_dim

    def build_tree(self, X, y):
        # indices = np.ones_like(y)
        # indices = np.where(indices != np.nan)
        idx = range(len(X))
        # print("indexes = ", idx)
        # print(len(indices[0]))
        # print("values=", X[idx])
        depth = self.start_dim
        self.root = self.build_tree_recursive(X, y, idx, depth)
        pass

    def build_tree_recursive(self, X, y, idx, depth):
        M = len(X[0])
        dim = depth % M
        if len(idx) == 0:
            return None
        elif len(idx) == 1:
            x_val = X[idx][0]
            y_val = y[idx][0]

            val = x_val[dim]
            p = (x_val, y_val)
            # print("val, p = ", val, p)
            node = Node(dim, val, p, None, None)
            return node
        else:

            med_idx = self.get_median_index(X, idx, dim)
            # print("med_idx = , shape", med_idx, X[med_idx].shape)
            val = X[med_idx][dim]
            point = (X[med_idx], y[med_idx])
            left_idx = [i for i in idx if (i != med_idx and X[i][dim] <= val)]
            right_idx = [i for i in idx if (i != med_idx and X[i][dim] > val)]
            left_node = self.build_tree_recursive(X, y, left_idx, depth+1)
            right_node = self.build_tree_recursive(X, y, right_idx, depth+1)

            n = Node(dim, val, point, left_node, right_node)
            if left_node:
                left_node.parent = n
            if right_node:
                right_node.parent = n

            return n




    def get_median_index(self, x_list, idx, dim):
        # print("get_median")
        # print(x_list)
        # print(idx)
        # print(x_list[idx])

        # print(one_d_x)
        # print("get_median")
        # print("get_median2222")
        one_d_x = x_list[idx][:, dim]
        # print(one_d_x)
        # print(idx)
        pair_list = list(map(lambda i: (x_list[i][dim], i), idx))

        pair_list = sorted(pair_list, key=lambda x:x[0])  # sort according values
        med = (len(pair_list)-1) // 2
        return pair_list[med][1]  # return only the index


    def _get_hyper_plane_dist(self, Xi, nd):
        """Calculate euclidean distance between Xi and hyper plane.
        Arguments:
            Xi {list} -- 1d list with int or float.
            nd {node}
        Returns:
            float -- Euclidean distance.
        """

        dim = nd.d
        X0 = nd.point[0]
        return abs(Xi[dim] - X0[dim])

    def _search_tree(self, Xi, startNode):

        cur = startNode
        while cur.left or cur.right:
            # if get_eu_dist(Xi.tolist(), cur.point[0].tolist()) < 0.001:
            #     return cur.point

            if cur.left is None:
                cur = cur.right
            elif cur.right is None:
                cur = cur.left
            else:
                dim = cur.d
                my_val = Xi[dim]
                divide_val = cur.val
                if my_val <= divide_val:
                    cur = cur.left
                else:
                    cur = cur.right

        return cur


    def nearest_neighbour_search(self, Xi):
        """Nearest neighbour search and backtracking.
        Arguments:
            Xi {list} -- 1d list with int or float.
        Returns:
            node -- The nearest node to Xi.
        """

        # The leaf node after searching Xi.
        dist_best = float("inf")
        best_node = self._search_tree(Xi, self.root)
        que = [(self.root, best_node)]
        while que:
            sub_root_node, cur_node = que.pop(0)

            dist = get_eu_dist(Xi, sub_root_node.point[0])

            if dist < dist_best:
                dist_best, best_node = dist, sub_root_node

            while cur_node is not sub_root_node:

                dist = get_eu_dist(Xi, cur_node.point[0])

                if dist < dist_best:
                    dist_best, best_node = dist, cur_node

                # If it's necessary to visit brother node.
                if cur_node.brother() and dist_best > self._get_hyper_plane_dist(Xi, cur_node.parent):
                    _nd_best = self._search_tree(Xi, cur_node.brother())
                    que.append((cur_node.brother(), _nd_best))
                # Back track.
                cur_node = cur_node.parent

        return best_node





if __name__ == "__main__":
    # start_dim = 1
    # tree = KDTree(start_dim)
    # X = np.array([[6, 2], [6, 3], [3, 5], [5, 0]]) #, [1, 2], [4, 9], [8, 1]])
    # y = np.array([1,      2,       1,       1]) # ,      2,      1,      1])
    #
    # tree.build_tree(X, y)
    # for i in range(len(y)):
    #     print("find ,", X[i])
    #     print(tree.nearest_neighbour_search(X[i]))

    param_train = sys.argv[1]
    param_test = sys.argv[2]
    param_start_dim = sys.argv[3]
    param_start_dim = int(param_start_dim)

    df = pd.read_csv(param_train, delim_whitespace=True)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    tree = KDTree(param_start_dim)
    tree.build_tree(X, y)

    # for i in range(len(y)):
    #     ret_node = tree.nearest_neighbour_search(X[i])
    #     label = ret_node.get_label()
    #     assert label == y[i], "bad match"
    #     print(label)


    df = pd.read_csv(param_test, delim_whitespace=True)
    X_test = df.iloc[:, :].values
    for i in range(len(X_test)):
        ret_node = tree.nearest_neighbour_search(X_test[i])
        label = ret_node.get_label()
        print(label)


