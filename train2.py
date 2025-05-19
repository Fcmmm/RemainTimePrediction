#coding: utf-8
from csv import writer
from GRU import GRU
from GRUAtt import GRUAtt
from BiGRU import BiGRU
from BiGRUAtt import BiGRUAtt
from LSTM import LSTM
from LSTMAtt import LSTMAtt
from BiLSTM import BiLSTM
from BiLSTMAtt import BiLSTMAtt
from TCN import TCN
from QRNN import QRNN
from BiQRNN import BiQRNN
from Transformer import Transformer
from BiQRNNAtt import BiQRNNAtt
from input_data import InputData
from collections import deque
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
from torch.utils.tensorboard import SummaryWriter
import time
#time_unit = second minute hour day month


# time_unit 时间单位 batch_size 一次学习几条曲线 start_pos,stop_pos 截取的区间 train_splitThreshold 训练集与测试集的划分
# embd_dimension 表示事件被压缩成几维向量

# 5 10
def train(data_address,data_name,vector_address=None, vocab_address=None, embd_dimension =3, train_splitThreshold=0.7,
          time_unit='minute', batch_size=64, start_pos=3, stop_pos=5, length_size=3, prefix_minLength=0, prefix_maxLength=None,
          loss_type= 'L1Loss', optim_type= 'Adam', model_type='LSTMAtt', hidden_dim=5,
          train_type='mix', n_layer=1, dropout=0.3, max_epoch_num=500, learn_rate_min = 0.008,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/' ):
    #初始化数据
    out_size = 1
    epoch = 0
    learn_rate = 0.01
    learn_rate_down = 0.001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    loss_chage = 0

    data = InputData(data_address, embd_dimension = embd_dimension, time_unit = 'minute')

    if vector_address == None:
        data.encodeEvent(None)
    elif vector_address != None and vocab_address != None:
        data.encodeEventByVocab(vocab_address, vector_address)
    else:
        data.encodeEvent(vector_address)
    data.encodeTrace()
    data.splitData(train_splitThreshold)
    data.initBatchData(time_unit, start_pos)
    # 选择使用混合长度或者固定长度
    if train_type == 'mix':
        data.generateMixLengthBatch(batch_size)
    else:
        data.generateSingleLengthBatch(batch_size,start_pos)

    #初始化模型
    if model_type == 'LSTM':

        model=LSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                   batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'LSTMAtt':
        model=LSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                        batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiLSTM':
        model=BiLSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiLSTMAtt':
        model=BiLSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                          batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'GRU':
        model=GRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer ,dropout=dropout, embedding=data.embedding)
    elif model_type == 'GRUAtt':
        model=GRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiGRU':
        model=BiGRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                      batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)
    elif model_type == 'BiGRUAtt':
        model=BiGRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)
    elif model_type == 'TCN':
        model=TCN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,dropout=dropout, embedding=data.embedding)
    elif model_type == 'QRNN':
        model=QRNN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,dropout=dropout, embedding=data.embedding)
    elif model_type =='BiQRNN':
        model=BiQRNN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,dropout=dropout, embedding=data.embedding)
    elif model_type =='BiQRNNAtt':
        model=BiQRNNAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,dropout=dropout, embedding=data.embedding)
    elif model_type =='Transformer':
        model=Transformer(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                         batch_size=batch_size,dropout=dropout, embedding=data.embedding)
    if loss_type == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss()

    # 假设 device 已经被正确设置，例如：
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 确保模型在正确的设备上
    model = model.to(device)
    model.train()  # 确保模型处于训练模式
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    #初始化存储文件
    start_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
    model_detal = 'embdDim' + str(embd_dimension) + '_loss' + loss_type + '_optim' + optim_type + '_hiddenDim' \
                  + str(hidden_dim) + '_startPos' + str(start_pos) + '_trainType' + train_type + '_nLayer' + str(n_layer) \
                  + '_dropout' + str(dropout)
    save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_record_all = train_record_folder + data_name +'_sum.csv'
    save_record_single = train_record_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_result_folder = result_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    for folder in [save_model_folder,save_record_single]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_record_single = save_record_single + start_time + '.csv'
    if not os.path.exists(save_record_all):
        save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
        save_record_all_write = 'modelType,embdDim,lossType,optimType,hiddenDim,startPos,trainType,layerNum,' \
                                'dropout,epoch,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss,modelFile,recordFile,resultFile\n'
        save_record_all_open.writelines(save_record_all_write)
        save_record_all_open.close()
    save_record_single_open = open(save_record_single,'w',encoding='utf-8')
    save_record_single_write = 'epoch,startPos,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
    save_record_single_open.writelines(save_record_single_write)

    # 初始化TensorBoard
    # tb_logs_dir = 'tb_logs'
    # shutil.rmtree(tb_logs_dir)
    # writer = SummaryWriter(tb_logs_dir)
    #开始训练
    if train_type != 'iteratioin':
        while epoch < max_epoch_num and learn_rate >= learn_rate_min:


            total_loss = torch.FloatTensor([0])
            for (input, target) in data.train_batch:
                optimizer.zero_grad()
                input = np.array(input)
                target = np.array([[t] for t in target])
                input = torch.LongTensor(input)
                target = torch.LongTensor(target)
                target = target.float()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input = input.to(device)
                output = model(input)
                target = target.to(device)
                loss = criterion(output, target)
                loss.backward(retain_graph=True)
                total_loss = total_loss.to(device)
                optimizer.step()
                total_loss += loss.data
                print(output)
            loss_deque.append(total_loss.item())
            loss_change_deque.append(total_loss.item())
            loss_change = total_loss.item() - sum(loss_deque) / len(loss_deque)
            loss_change = abs(loss_change)
            MSE, MAE, RMSE, TOTAL, MEAN = evaluate(model,data.test_batch)
            # 当loss变化稳定时降低学习率   自适应学习率
            if loss_change < 5 and len(loss_deque) == 20:
                now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                model_save = save_model_folder + now_time + '.pth'
                torch.save(model, model_save)
                # result 文件命名格式
                result_save_file = result_save_folder + str(model_type) + '_epoch' + str(epoch)+ '_' + now_time + '.csv'
                result_save_open = open(result_save_file,'w',encoding='utf-8')
                result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                result_save_open.writelines(result_save_write)
                result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                    total_loss.item()) + '\n'
                result_save_open.writelines(result_save_write)
                for prefix_length in range(start_pos,stop_pos + 1):
                    # if prefix_length % 2 != 0 and prefix_length != 5:
                        # continue
                    data.generateSingleLengthBatch(batch_size,prefix_length)
                    if len(data.test_batch) != 0:
                        MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                        result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                            + ',' + str(RMSE1) + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    else:
                        result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + '无测试数据' + ',' + '无测试数据' \
                                            + ',' + '无测试数据' + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write)
                result_save_open.close()
                if train_type == 'mix':
                    data.generateMixLengthBatch(batch_size)
                else:
                    data.generateSingleLengthBatch(batch_size, start_pos)
                save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                        +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos) \
                                        +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                        +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                        +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                        +','+ model_save +',' + save_record_single +',' + result_save_file + '\n'

                save_record_all_open.writelines(save_record_all_write)
                save_record_all_open.close()
                if learn_rate > learn_rate_down:
                    learn_rate = learn_rate - learn_rate_down
                else:
                    learn_rate_down = learn_rate_down * 0.1
                    learn_rate = learn_rate - learn_rate_down
                optimizer = optim.Adam(model.parameters(), lr=learn_rate)
                loss_deque = deque(maxlen=20)
                loss_deque.append(total_loss.item())
            if len(loss_change_deque) == 30 and (max(loss_change_deque) - min(loss_change_deque) < 20):
                now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                model_save = save_model_folder + now_time + '.pth'
                torch.save(model, model_save)
                result_save_file = result_save_folder + now_time + '.pth'
                result_save_open = open(result_save_file,'w',encoding='utf-8')
                result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                result_save_open.writelines(result_save_write)
                result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                    total_loss.item()) + '\n'
                result_save_open.writelines(result_save_write)
                for prefix_length in range(start_pos,stop_pos + 1):
                    # if prefix_length % 2 != 0 and prefix_length != 5:
                        # continue
                    data.generateSingleLengthBatch(batch_size,prefix_length)
                    if len(data.test_batch) != 0:
                        MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                        result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                            + ',' + str(RMSE1) + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    else:
                        result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + '无测试数据' + ',' + '无测试数据' \
                                            + ',' + '无测试数据' + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write)
                result_save_open.close()
                if train_type == 'mix':
                    data.generateMixLengthBatch(batch_size)
                else:
                    data.generateSingleLengthBatch(batch_size, start_pos)
                save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                        +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos) \
                                        +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                        +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                        +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                        +','+ model_save +',' + save_record_single + ',' + result_save_file + '\n'
                save_record_all_open.writelines(save_record_all_write)
                save_record_all_open.close()
                if learn_rate > learn_rate_down:
                    learn_rate = learn_rate - learn_rate_down
                else:
                    learn_rate_down = learn_rate_down * 0.1
                    learn_rate = learn_rate - learn_rate_down
                optimizer = optim.Adam(model.parameters(), lr=learn_rate)
                loss_change_deque = deque(maxlen=30)
                loss_change_deque.append(total_loss.item())
            print(MSE, MAE, RMSE, TOTAL, total_loss.item(),epoch,learn_rate,loss_change)

            save_record_single_write = str(epoch) + ','+ str(start_pos) +','+ str(learn_rate) + ','+ str(MSE) +','+ str(MAE)\
                                    +','+ str(RMSE) + ','+ str(total_loss.item()/len(data.train_batch)) + ','+ str(total_loss.item()) + '\n'
            save_record_single_open.writelines(save_record_single_write)
            #print(loss_change)
            epoch = epoch + 1

    save_record_single_open.close()
def evaluate(model, test_batchs):
    target_list = list()
    predict_list = list()
    for (input, target) in test_batchs:

        input = np.array(input)
        #print(input)
        input = Variable(torch.LongTensor(input).cuda())

        prediction = model(input)
        predict_list += [pdic.item() for pdic in prediction]
        target_list += target
    MSE = computeMSE(target_list,predict_list)
    MAE = computeMAE(target_list,predict_list)
    RMSE = sqrt(MSE)
    TOTAL = computeTOTAL(target_list,predict_list)
    MEAN = computeMEAN(target_list,predict_list)
    return MSE,MAE,RMSE,TOTAL,MEAN
def computeMAE(list_a,list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num]-list_b[num]))
    MAE = sum(MAE_temp)/len(list_a)
    return MAE
def computeMSE(list_a,list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE
def computeTOTAL(list_a,list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL
def computeMEAN(list_a,list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp)/len(list_a)
    return MEAN



# train('./data/helpdesk_extend.csv',vector_address = './vector/vector.txt', vocab_address = None, train_splitThreshold = 0.7,
#       time_unit = 'second', batch_size=20, )
# train('E:/学习/毕业论文/RemainTimePrediction-master-caorui/data/Road_Traffic_Fine_Management_P.csv', data_name='Road_Traffic_Fine_Management_P',
      # vector_address='./vector/helpdesk_extend_vectors_2CBoW_noTime_noEnd_Vector_vLoss_v1.txt',
      # vocab_address='./vector/helpdesk_extend_2CBoW_noTime_noEnd_vocabulary.txt',
      # embd_dimension=2, train_splitThreshold=0.7, time_unit='month', batch_size=20)



start_time = time.time()  # 记录开始时间
train('F:\python_project\数据集\RemainTimePrediction-master-liubocheng\data\codeforces 936_15.csv', data_name='codeforces 936_15',
       vector_address='./vector/Codeforces 936_15_vectors_GloVe.txt',
       vocab_address='./vector/Codeforces 936_15_2CBoW_noTime_noEnd_vocabulary.txt',
      embd_dimension=3, train_splitThreshold=0.8, time_unit='minute', batch_size=64)
end_time = time.time()    # 记录结束时间
print(f"运行时间：{end_time - start_time} 秒")
# 创建一个空白的 txt 文件
now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
file_name =  'result\\'+'---------------------------分割------------------------' + now_time
with open(file_name, 'w', encoding='utf-8') as file:
    pass  # 这里使用 pass 语句，不进行任何写入操作