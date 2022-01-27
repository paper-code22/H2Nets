import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon, Add_Single_Window_Horizon
from load_dataset import load_taxi_dataset, load_hyperedge, load_hypernode, load_time_dataset, load_edge_dataset, load_Hodge_Laplacian, load_incidence_matrix
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def data_time_edge_loader(X, X_time, X_e, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, X_time, X_e, Y = TensorFloat(X), TensorFloat(X_time), TensorFloat(X_e), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, X_time, X_e, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def hyper_data_loader(hyper_data, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorLong = torch.cuda.LongTensor if cuda else torch.LongTensor
    hyper_data = TensorLong(hyper_data)
    hyper_data = torch.utils.data.TensorDataset(hyper_data)
    dataloader = torch.utils.data.DataLoader(hyper_data, batch_size=1,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def hyper_edge_node_data_loader(hyper_edge_data, hyper_node_data, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorLong = torch.cuda.LongTensor if cuda else torch.LongTensor
    hyper_edge_data, hyper_node_data = TensorLong(hyper_edge_data), TensorLong(hyper_node_data)
    hyper_data = torch.utils.data.TensorDataset(hyper_edge_data, hyper_node_data)
    dataloader = torch.utils.data.DataLoader(hyper_data, batch_size=1,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def hodge_data_loader(hodge_laplacian, incidence_matrix, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    hodge_laplacian, incidence_matrix = TensorFloat(hodge_laplacian), TensorFloat(incidence_matrix)
    hodge_data = torch.utils.data.TensorDataset(hodge_laplacian, incidence_matrix)
    dataloader = torch.utils.data.DataLoader(hodge_data, batch_size=1,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_hyper_dataloader(args, type, hyper_data_type):
    #load raw taxi (volume) dataset
    data_train, data_test = load_taxi_dataset(type)

    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: x, y ->', x_tra.shape, y_tra.shape)
    print('Test: x, y ->', x_test.shape, y_test.shape)
    ##############get triple dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = None
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    #load hyper dataset
    hyperdata = load_hyperedge(hyper_data_type)
    hyper_dataloader = hyper_data_loader(hyperdata)

    return train_dataloader, val_dataloader, test_dataloader, hyper_dataloader

def get_hyper_time_dataloader(args, type, hyper_data_type):
    #load raw taxi (volume) dataset
    data_train, data_test = load_taxi_dataset(type)

    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: x, y ->', x_tra.shape, y_tra.shape) # # (num_obs, lag, num_nodes, 2)
    print('Test: x, y ->', x_test.shape, y_test.shape)

    #load time variables
    time_train, time_test = load_time_dataset(type)
    x_time_tra = Add_Time_Window_Horizon(time_train, args.lag, args.horizon, single)
    x_time_test = Add_Time_Window_Horizon(time_test, args.lag, args.horizon, single)
    print('Train: time_x ->', x_time_tra.shape) # (num_obs, lag, 55)
    print('Test: time_x ->', x_time_test.shape)

    ##############get triple dataloader######################
    train_dataloader = data_time_loader(x_tra, x_time_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = None
    test_dataloader = data_time_loader(x_test, x_time_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    #load hyper dataset
    hyperdata = load_hyperedge(hyper_data_type)
    hyper_dataloader = hyper_data_loader(hyperdata)

    return train_dataloader, val_dataloader, test_dataloader, hyper_dataloader


def get_hyper_hodge_time_edge_dataloader(args, type, hyper_type_0, hyper_type_1):
    #load raw taxi (volume) dataset
    data_train, data_test = load_taxi_dataset(type)

    single = False
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: x, y ->', x_tra.shape, y_tra.shape) # # (num_obs, lag, num_nodes, 2)
    print('Test: x, y ->', x_test.shape, y_test.shape)

    #load time variables
    time_train, time_test = load_time_dataset(type)
    x_time_tra = Add_Single_Window_Horizon(time_train, args.lag, args.horizon, single)
    x_time_test = Add_Single_Window_Horizon(time_test, args.lag, args.horizon, single)
    print('Train: time_x ->', x_time_tra.shape) # (num_obs, lag, 55)
    print('Test: time_x ->', x_time_test.shape)

    # load edge variables
    edge_train, edge_test = load_edge_dataset()
    x_edge_tra = Add_Single_Window_Horizon(edge_train, args.lag, args.horizon, single)
    x_edge_test = Add_Single_Window_Horizon(edge_test, args.lag, args.horizon, single)
    print('Train: edge_x ->', x_edge_tra.shape)  # (num_obs, lag, 1)
    print('Test: edge_x ->', x_edge_test.shape)

    ##############get triple dataloader######################
    train_dataloader = data_time_edge_loader(x_tra, x_time_tra, x_edge_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = None
    test_dataloader = data_time_edge_loader(x_test, x_time_test, x_edge_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    #load hyper dataset
    hyper_edge_data = load_hyperedge(hyper_type_0)
    hyper_node_data = load_hypernode(hyper_type_1)
    hyper_dataloader = hyper_edge_node_data_loader(hyper_edge_data, hyper_node_data)

    # load hodge dataset
    Hodge_Laplacian_data = load_Hodge_Laplacian()
    incidence_matrix_data = load_incidence_matrix()
    hodge_dataloader = hodge_data_loader(Hodge_Laplacian_data, incidence_matrix_data)

    return train_dataloader, val_dataloader, test_dataloader, hyper_dataloader, hodge_dataloader

