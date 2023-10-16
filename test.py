# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
import numpy as np
import pandas as pd
#
# #
# a=np.load('./data/PJM/train.npz')
# # b=np.load('./data/PEMS-BAY/val.npz')
# print('a',a.files)
# # print('b',b.files)
# xa = a['x']
# ya = a['y']
# # xb = b['x']
# # yb = b['y']
# x_offsets_a = a['x_offsets']
# y_offsets_a = a['y_offsets']
# x_offsets_b = b['x_offsets']
# y_offsets_b = b['y_offsets']
# print('xa', np.shape(xa))
# # print('xa', xa[0,:,0,:])
# print('ya', np.shape(ya))
# # print('xb', np.shape(xb))
# # print('yb', np.shape(yb))
# print('x_offsets', x_offsets_a)
# print('y_offsets', y_offsets_b)
# #
# df = pd.read_hdf('./data/metr-la.h5')
# # print(df.index.tolist()[:5])
# print(df[:12])
#
df = pd.read_csv('./data/PJM-aep5.csv')
df = df.set_index(['Datetime'])
df.index = pd.to_datetime(df.index)

df = df[:200]
print(df)
#
# period=7
# test_ratio=0.2
# null_val=0.
#
# n_sample, n_sensor = df.shape
# n_test = int(round(n_sample * test_ratio))
# n_train = n_sample - n_test
# y_test = df[-n_test:]
# y_predict = pd.DataFrame.copy(y_test)
#
# for i in range(n_train, min(n_sample, n_train + period)):
#     # print(i)
#     inds = [j for j in range(i % period, n_train, period)]
#     print(inds)
#     historical = df.iloc[inds, :]
#     # print(historical)
#     # print(historical[historical != null_val].mean())
#     y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
#     # print(y_predict)

# # Copy each period.
# for i in range(n_train + period, n_sample, period):
#     # print(i)
#     size = min(period, n_sample - i)
#     start = i - n_train
#     print(start)
#     y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values

# test_num = int(round(df.shape[0] * 0.2))
# # print(test_num)
# y_test = df[-test_num:]
# y_predict = df.shift(1).iloc[-test_num:]
# print(y_test, y_predict)

#
# # df = pd.read_hdf('data/metr-la.h5')
# #
# # datetime_series = pd.to_datetime(df.index)
# #
# # # create datetime index passing the datetime series
# # datetime_index = pd.DatetimeIndex(datetime_series.values)
# #
# # print(datetime_index)
#
# # python -m scripts.gen_pjm_data --output_dir=data/PJM-aep5 --traffic_df_filename=data/PJM-aep5.csv
# # python train.py --config_filename=data/model/para_pjm.yaml --temperature=0.5
# # df = pd.read_csv('./data/pjm.csv')
# # df = df.set_index(['Datetime'])
# # df.index = pd.to_datetime(df.index)
#
#
#
#
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from lib import utils
# from model.pytorch.model import GTSModel
# from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
# import pandas as pd
# import os
# import time
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # # df = pd.read_csv('./data/pjm.csv')
# # # df = df.set_index(['Datetime'])
# # # df.index = pd.to_datetime(df.index)
# # df = pd.read_hdf('./data/pems-bay.h5')
# #
# # #else:
# # #    df = pd.read_csv('./data/pmu_normalized.csv', header=None)
# # #    df = df.transpose()
# # num_samples = df.shape[0]
# #
# # num_train = round(num_samples * 0.7)
# # df = df[:num_train].values
# # scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
# # train_feas = scaler.transform(df)
# # print(train_feas.shape)
# # train_feas_a = torch.Tensor(train_feas).to(device)
# # #print(self._train_feas.shape)
# #
# # k = 5
# # knn_metric = 'cosine'
# # from sklearn.neighbors import kneighbors_graph
# # g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
# # g = np.array(g.todense(), dtype=np.float32)
#
#
# df = pd.read_csv('./data/PJM.csv')
# df = df.set_index(['Datetime'])
# df.index = pd.to_datetime(df.index)
#
# #else:
# #    df = pd.read_csv('./data/pmu_normalized.csv', header=None)
# #    df = df.transpose()
# num_samples = df.shape[0]
#
# num_train = round(num_samples * 0.7)
# df = df[:num_train].values
# scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
#
# train_feas = scaler.transform(df)
#
# train_feas = torch.Tensor(train_feas)
#
# #print(self._train_feas.shape)
#
# k = 5
# knn_metric = 'cosine'
# from sklearn.neighbors import kneighbors_graph
# g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
#
# g = np.array(g.todense(), dtype=np.float32)
# print(g)
#

import numpy as np

a = np.arange(10*10*12).reshape((10, 12, 10))
b = np.arange(10*64).reshape((10, 64))
c = np.arange(12*64*10).reshape((64, 12, 10))
# a.reshape(1, 12,10,64)
# print(a)
print(np.shape(np.matmul(a,b)))
