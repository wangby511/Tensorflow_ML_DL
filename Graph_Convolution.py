import numpy as np
from networkx import to_numpy_matrix
import networkx as nx
import matplotlib.pyplot as plt

# https://blog.csdn.net/qq_36793545/article/details/84844867?utm_medium=distribute.pc_relevant.
# none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.
# none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase
def small_demo():
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]],dtype=float
    )
    W = np.matrix([[1, -1], [-1, 1]])

    I = np.matrix(np.eye(A.shape[0]))

    X = np.matrix([[i, -i] for i in range(A.shape[0])], dtype=float)

    A_hat  = A + I

    D_hat = np.array(np.sum(A, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))

    B = D_hat**-1 * A_hat * X * W

    print(B)
    # [[  2.  -2.]
    #  [  6.  -6.]
    #  [  3.  -3.]
    #  [ 10. -10.]]

zkc = nx.karate_club_graph()
order = sorted(list(zkc.nodes()))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

A = to_numpy_matrix(zkc, nodelist=order)
# [[0. 1. 1. ... 1. 0. 0.]
#  [1. 0. 1. ... 0. 0. 0.]
#  [1. 1. 0. ... 0. 1. 0.]
#  ...
#  [1. 0. 0. ... 0. 1. 1.]
#  [0. 0. 1. ... 1. 0. 1.]
#  [0. 0. 0. ... 1. 1. 0.]]
# (34, 34)

I = np.eye(zkc.number_of_nodes())
# 34

A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))


def relu(x):
    return (abs(x) + x) / 2

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat ** -1 * A_hat * X * W)

# print(zkc.number_of_nodes())
# 34
W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))

# print(W_1.shape)
# (34, 4)
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2
feature_representations = {node: np.array(output)[node] for node in zkc.nodes()}

for k,v in feature_representations.items():
    print(k,v)