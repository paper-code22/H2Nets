import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
path = os.getcwd()
import torch
import torch.utils.data


def load_taxi_dataset(type):
    # taxi volume data
    train_data_path = os.path.join(path + '/data/volume_train.npz')
    test_data_path = os.path.join(path + '/data/volume_test.npz')
    train_data_ = np.load(open(train_data_path, "rb"))["volume"]#[:,:,:,0]  # pickup or dropoff volume; (1920, 10, 20, 1) -> (1920, 10, 20, 2)
    test_data_ = np.load(open(test_data_path, "rb"))["volume"]#[:,:,:,0]   # pickup dropoff volume; (960, 10, 20, 1) -> (960, 10, 20, 2)

    train_data_max = 1. # train_data_.max()
    test_data_max = 1. #test_data_.max()

    train_data = train_data_.reshape(train_data_.shape[0], -1, 2)/train_data_max # (1920, 200, 1) -> (1920, 200, 2)
    test_data = test_data_.reshape(test_data_.shape[0], -1, 2)/test_data_max # (960, 200, 1) -> (960, 200, 2)
    return train_data, test_data

# hyperedge index
def hypergraph_load(hyperedge):
    edge_index_node_id = []
    edge_index_hyperedge_id = []
    for i in range(hyperedge.shape[0]): # 26
        node_id = np.where(hyperedge[i,:]!=0)[0]
        hyperedge_id = np.repeat(int(i), node_id.shape[0])
        edge_index_node_id.extend(node_id)
        edge_index_hyperedge_id.extend(hyperedge_id)

    edge_index_node_id = np.array(edge_index_node_id).reshape(-1,1)
    edge_index_hyperedge_id = np.array(edge_index_hyperedge_id).reshape(-1,1)
    edge_index = np.concatenate([edge_index_node_id, edge_index_hyperedge_id], axis=-1)

    return edge_index.transpose()

def load_hyperedge(hyper_data_type):
    # shape of hyperedge is (32, 200)
    hyperedge_path = os.path.join(path + '/data/halfhourlyHyperedge.npz')
    hyperedge_ = np.load(hyperedge_path)['hyperedge']
    hyper_edge_index_ = hypergraph_load(hyperedge_)
    hyper_edge_index = np.expand_dims(hyper_edge_index_, axis= 0)
    return hyper_edge_index

def load_hypernode(hyper_data_type):
    # shape of load_hypernode is (26, 100)
    hypernode_path = os.path.join(path + '/data/hypernode_edgeindex_cos.npz')
    hypernode_ = np.load(hypernode_path, allow_pickle=True)['hypernode']
    hyper_node_index_ = hypergraph_load(hypernode_)
    hyper_node_index = np.expand_dims(hyper_node_index_, axis= 0)
    return hyper_node_index

def load_Hodge_Laplacian():
    # L1: (M, M)
    hodge_laplacian_path = os.path.join(path + '/data/NYC_taxi_Hodge_Laplacian.npz')
    hodge_laplacian_ = np.load(hodge_laplacian_path, allow_pickle=True)['arr_0']
    hodge_laplacian = np.expand_dims(hodge_laplacian_, axis= 0)
    return hodge_laplacian

def load_incidence_matrix():
    # B1: (M, M)
    incidence_matrix_path = os.path.join(path + '/data/NYC_taxi_B1.npz')
    incidence_matrix_ = np.load(incidence_matrix_path, allow_pickle=True)['arr_0']
    incidence_matrix = np.expand_dims(incidence_matrix_, axis= 0)
    return incidence_matrix

def load_time_dataset(type):
    # taxi volume data
    train_data_path = os.path.join(path + '/data/volume_train.npz')
    test_data_path = os.path.join(path + '/data/volume_test.npz')
    volume_train = np.load(open(train_data_path, "rb"))["volume"] # pickup and dropoff volume; (1920, 10, 20, 2)
    volume_test = np.load(open(test_data_path, "rb"))["volume"] # pickup and dropoff volume; (960, 10, 20, 2)

    day_train = int(volume_train.shape[0] / 48)
    day_test = int(volume_test.shape[0] / 48)

    # train
    time_of_the_day_train = np.vstack([np.identity(48)] * day_train)
    day_of_the_week_train = np.tile(np.identity(7)[0, :], (48, 1))

    for i in range(1, day_train):
        day = i % 7
        day_of_the_week_train = np.vstack([day_of_the_week_train, np.tile(np.identity(7)[day, :], (48, 1))])

    time_train_ = np.hstack([time_of_the_day_train, day_of_the_week_train]) # (1920, 55)
    time_train = np.expand_dims(time_train_, axis= 1)


    # test
    time_of_the_day_test = np.vstack([np.identity(48)] * day_test)
    day_of_the_week_test = np.tile(np.identity(7)[day_train % 7, :], (48, 1))

    for i in range(day_train + 1, day_train + day_test):
        day = i % 7

        day_of_the_week_test = np.vstack([day_of_the_week_test, np.tile(np.identity(7)[day, :], (48, 1))])

    time_test_ = np.hstack([time_of_the_day_test, day_of_the_week_test]) # (960, 55)
    time_test = np.expand_dims(time_test_, axis=1)

    return time_train, time_test

def load_edge_dataset():
    # taxi edge data
    train_edge_data_path = os.path.join(path + '/data/NYC_taxi_train_x_edge.npz')
    test_edge_data_path = os.path.join(path + '/data/NYC_taxi_test_x_edge.npz')
    train_edge_data_ = np.load(train_edge_data_path, allow_pickle=True)['arr_0']
    test_edge_data_ = np.load(test_edge_data_path, allow_pickle=True)['arr_0']

    train_edge_data = train_edge_data_.reshape(train_edge_data_.shape[0], -1, 1) # (1920, 100, 1)
    test_edge_data = test_edge_data_.reshape(test_edge_data_.shape[0], -1, 1) # (960, 100, 1)
    return train_edge_data, test_edge_data

