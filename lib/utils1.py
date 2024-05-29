import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np

def re_normalization(x, mean, std):
    x = x * std + mean
    return x

def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    这个函数转为mstgcn，astgcn设计，返回的数据x都是通过减均值除方差进行归一化的，y都是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) +'_dstagnn'

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x = file_data['train_x']  # (10181, 307, 3, 12)
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)#.to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_x_tensor, train_loader, train_target_tensor, val_x_tensor, val_loader, val_target_tensor, test_x_tensor, test_loader, test_target_tensor, mean, std


def compute_val_loss_mstgcn(DEVICE, net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            encoder_inputs = encoder_inputs.to('cpu')
            labels = labels.to('cpu')
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def predict_and_save_results_mstgcn(DEVICE, net, train_target_tensor, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()
        train_target_tensor = train_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            encoder_inputs = encoder_inputs.to('cpu')
            labels = labels.to('cpu')

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        print('train_target_tensor:', train_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s-th point' % (global_step, i+1))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)

            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))

            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)

        metrics_filename = os.path.join(params_path, 'metrics_epoch_%s_%s' % (global_step, type))
        np.savez(metrics_filename, mae=mae, rmse=rmse, mape=mape)

        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))

        excel_list.extend([mae, rmse, mape])
        print(excel_list)





