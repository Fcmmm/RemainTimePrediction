# coding: utf-8
import random
from my_model.Mamba import Mamba
from my_model.GRU import GRU
from my_model.GRUAtt import GRUAtt
from my_model.BiGRU import BiGRU
from my_model.BiGRUAtt import BiGRUAtt
from my_model.LSTM import LSTM
from my_model.LSTMAtt import LSTMAtt
from my_model.BiLSTM import BiLSTM
from my_model.BiLSTMAtt import BiLSTMAtt
from my_model.TCN import TCN
from my_model.QRNN import QRNN
from my_model.BiQRNN import BiQRNN
from my_model.Transformer import Transformer
from my_model.BiQRNNAtt import BiQRNNAtt
from input_data_1 import InputData
from collections import deque
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from datetime import datetime
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置全局字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
import time

seed = 41

# 设置PyTorch、NumPy、Python随机种子
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 针对CUDA（GPU）
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def generate_prediction_samples(model, data, num_samples=10, sample_file_path=None):
    """
    生成预测样本并保存到文件

    Args:
        model: 训练好的模型
        data: 数据对象
        num_samples: 生成样本数量
        sample_file_path: 保存文件路径
    """
    model.eval()  # 设置为评估模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_count = 0
    samples_data = []

    print("\n" + "=" * 60)
    print(f"生成 {num_samples} 条预测样本:")
    print("=" * 60)

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data.test_batch):
            if sample_count >= num_samples:
                break

            input_array = np.array(input_batch)
            input_tensor = torch.LongTensor(input_array).to(device)
            predictions = model(input_tensor)

            # 处理当前batch中的每个样本
            for i in range(len(input_batch)):
                if sample_count >= num_samples:
                    break

                sample_count += 1
                original_input = input_batch[i]
                true_value = target_batch[i]
                predicted_value = predictions[i].item()

                # 转换输入序列为可读格式（如果有词汇表映射的话）
                input_str = str(original_input)

                sample_info = {
                    'sample_id': sample_count,
                    'original_input': original_input,
                    'input_str': input_str,
                    'true_value': true_value,
                    'predicted_value': predicted_value,
                    'absolute_error': abs(true_value - predicted_value)
                }

                samples_data.append(sample_info)

                print(f"样本 {sample_count}:")
                print(f"  原始输入: {input_str}")
                print(f"  真实值: {true_value:.4f}")
                print(f"  预测值: {predicted_value:.4f}")
                print(f"  绝对误差: {abs(true_value - predicted_value):.4f}")
                print("-" * 40)

    # 如果提供了文件路径，保存到CSV文件
    if sample_file_path:
        try:
            os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
            with open(sample_file_path, 'w', encoding='utf-8') as f:
                f.write('样本ID,原始输入,真实值,预测值,绝对误差\n')
                for sample in samples_data:
                    f.write(f"{sample['sample_id']},\"{sample['input_str']}\","
                            f"{sample['true_value']:.4f},{sample['predicted_value']:.4f},"
                            f"{sample['absolute_error']:.4f}\n")
            print(f"\n预测样本已保存到: {sample_file_path}")
        except Exception as e:
            print(f"保存样本文件时出错: {e}")

    model.train()  # 恢复训练模式
    return samples_data


def train(data_address, data_name, vector_address=None, vocab_address=None, embd_dimension=3, train_splitThreshold=0.8,
          time_unit='minute', batch_size=32, start_pos=1, stop_pos=4, length_size=3, prefix_minLength=0,
          prefix_maxLength=None,
          loss_type='L1Loss', optim_type='Adam', model_type='Mamba', hidden_dim=64,
          train_type='unmix', n_layer=1, dropout=0, max_epoch_num=30, learn_rate_min=0.0001, numeric_input_dim=7,
          numeric_output_dim=4,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/',
          random_seed=42):
    # 初始化数据

    out_size = 1
    epoch = 0
    learn_rate = 0.0001
    learn_rate_down = 0.00001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    loss_chage = 0

    train_loss_list = []
    val_loss_list = []
    epochs_record = []

    data = InputData(data_address, embd_dimension=embd_dimension, time_unit='minute')

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
        data.generateSingleLengthBatch(batch_size, start_pos)

    # 初始化模型
    if model_type == 'LSTM':

        model = LSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                     batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                     numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'Mamba':
        model = Mamba(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                      out_size=out_size,
                      batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                      numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'LSTMAtt':
        model = LSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                        out_size=out_size,
                        batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                        numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiLSTM':
        model = BiLSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                       out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                       numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiLSTMAtt':
        model = BiLSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                          out_size=out_size,
                          batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                          numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'GRU':
        model = GRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                    numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'GRUAtt':
        model = GRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                       out_size=out_size,
                       batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                       numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiGRU':
        model = BiGRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                      out_size=out_size,
                      batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                      numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiGRUAtt':
        model = BiGRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                         out_size=out_size,
                         batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding,
                         numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'TCN':
        model = TCN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, dropout=dropout, embedding=data.embedding,
                    numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'QRNN':
        model = QRNN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                     batch_size=batch_size, dropout=dropout, embedding=data.embedding,
                     numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiQRNN':
        model = BiQRNN(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                       out_size=out_size,
                       batch_size=batch_size, dropout=dropout, embedding=data.embedding,
                       numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'BiQRNNAtt':
        model = BiQRNNAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                          out_size=out_size,
                          batch_size=batch_size, dropout=dropout, embedding=data.embedding,
                          numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
    elif model_type == 'Transformer':
        model = Transformer(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                            out_size=out_size,
                            batch_size=batch_size, dropout=dropout, embedding=data.embedding,
                            numeric_input_dim=numeric_input_dim, numeric_output_dim=numeric_output_dim)
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
    # 初始化存储文件
    start_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
    model_detal = 'embdDim' + str(embd_dimension) + '_loss' + loss_type + '_optim' + optim_type + '_hiddenDim' \
                  + str(hidden_dim) + '_startPos' + str(start_pos) + '_trainType' + train_type + '_nLayer' + str(
        n_layer) \
                  + '_dropout' + str(dropout)
    save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_record_all = train_record_folder + data_name + '_sum.csv'
    save_record_single = train_record_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    save_result_folder = result_save_folder + data_name + '/' + model_type + '/' + model_detal + '/'
    for folder in [save_model_folder, save_record_single]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_record_single = save_record_single + start_time + '.csv'
    if not os.path.exists(save_record_all):
        save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
        save_record_all_write = 'modelType,embdDim,lossType,optimType,hiddenDim,startPos,trainType,layerNum,' \
                                'dropout,epoch,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss,modelFile,recordFile,resultFile\n'
        save_record_all_open.writelines(save_record_all_write)
        save_record_all_open.close()
    save_record_single_open = open(save_record_single, 'w', encoding='utf-8')
    save_record_single_write = 'epoch,startPos,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
    save_record_single_open.writelines(save_record_single_write)

    # 初始化TensorBoard
    # tb_logs_dir = 'tb_logs'
    # shutil.rmtree(tb_logs_dir)
    # writer = SummaryWriter(tb_logs_dir)
    # 开始训练
    if train_type != 'iteration':
        while epoch < max_epoch_num and learn_rate >= learn_rate_min:
            total_loss = torch.FloatTensor([0])
            for (input, target) in data.train_batch:
                optimizer.zero_grad()
                # input = np.array(input)
                # target = np.array([[t] for t in target])
                # input = torch.LongTensor(input)
                # target = torch.LongTensor(target)
                input = np.array(input)
                target = np.array([[t] for t in target])
                # 复制数组以确保其可写
                input = np.copy(input)
                target = np.copy(target)
                input = torch.LongTensor(input)
                target = torch.LongTensor(target)
                target = target.float()

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input = input.to(device)
                output = model(input)
                target = target.to(device)
                loss = criterion(output, target)
                loss.backward(retain_graph=True)
                # 监控梯度
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f'{name} gradient norm: {param.grad.norm()}')

                total_loss = total_loss.to(device)
                optimizer.step()
                # print(f"Epoch {epoch}, lr={optimizer.param_groups[0]['lr']}")
                total_loss += loss.data
                # print(output)
            loss_deque.append(total_loss.item())
            loss_change_deque.append(total_loss.item())
            loss_change = total_loss.item() - sum(loss_deque) / len(loss_deque)
            loss_change = abs(loss_change)

            # ------【每轮训练结束后记录loss和评估指标】------
            train_mean_loss = total_loss.item() / len(data.train_batch)
            train_loss_list.append(train_mean_loss)

            MSE, MAE, RMSE, TOTAL, MEAN, _, _ = evaluate(model, data.test_batch)
            # 可选用MSE、MAE、RMSE作为验证指标
            val_loss_list.append(MAE)  # 或者MSE/其他
            epochs_record.append(epoch)
            # ------【其余保存模型/结果/日志逻辑照常不变】------

            # 当loss变化稳定时降低学习率   自适应学习率
            if loss_change < 5 and len(loss_deque) == 20:
                now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                model_save = save_model_folder + now_time + '.pth'
                torch.save(model, model_save)
                # result 文件命名格式
                result_save_file = result_save_folder + str(model_type) + '_epoch' + str(
                    epoch) + '_' + now_time + '.csv'
                result_save_open = open(result_save_file, 'w', encoding='utf-8')
                result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                result_save_open.writelines(result_save_write)
                result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(
                    MSE) + ',' + str(MAE) \
                                    + ',' + str(RMSE) + ',' + str(
                    total_loss.item() / len(data.train_batch)) + ',' + str(
                    total_loss.item()) + '\n'
                result_save_open.writelines(result_save_write)
                for prefix_length in range(start_pos, stop_pos + 1):
                    # if prefix_length % 2 != 0 and prefix_length != 5:
                    # continue
                    data.generateSingleLengthBatch(batch_size, prefix_length)
                    if len(data.test_batch) != 0:
                        MSE1, MAE1, RMSE1, TOTAL1, MEAN1, _, _ = evaluate(model, data.test_batch)
                        result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                            + ',' + str(RMSE1) + ',' + str(train_loss_list[-1]) + ',' + str(
                            TOTAL1) + '\n'
            else:
                result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                    learn_rate) + ',' + '无测试数据' + ',' + '无测试数据' \
                                    + ',' + '无测试数据' + ',' + str(train_loss_list[-1]) + ',' + '无测试数据' + '\n'
            result_save_open.writelines(result_save_write)
    print(f"最终模型和结果csv已保存: {result_save_file}")

    # ------【训练结束后画图】------
    plt.figure()
    plt.plot(epochs_record, train_loss_list, label='Train Loss')
    plt.plot(epochs_record, val_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(
        f"{model_type}, lr={learn_rate}, batch={batch_size}"
        # 如果加 hidden_dim、n_layer等，也可以加到标题里
        # f"{model_type}, lr={learn_rate}, batch={batch_size}, hidden_dim={hidden_dim}, n_layer={n_layer}"
    )
    plt.grid(True)
    # 你也可以在角落用plt.text()加注释（如有更多参数需要展示）
    loss_fig_path = os.path.join(
        result_save_folder,
        f"loss_curve_{model_type}_lr{learn_rate}_batch{batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()
    print(f"训练与验证集损失曲线已保存至: {loss_fig_path}")

    data.generateMixLengthBatch(batch_size)
    if len(data.test_batch) > 0:
        _, _, _, _, _, predictions, targets = evaluate(model, data.test_batch)

        # 创建对比图
        plt.figure(figsize=(12, 5))

        # 左图：散点图
        plt.subplot(1, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 右图：时序对比（前50个点）
        plt.subplot(1, 2, 2)
        show_points = min(50, len(targets))
        x = range(show_points)
        plt.plot(x, targets[:show_points], 'b-', label='真实值', linewidth=2)
        plt.plot(x, predictions[:show_points], 'r--', label='预测值', linewidth=2)
        plt.xlabel('样本序号')
        plt.ylabel('数值')
        plt.title(f'时序对比 (前{show_points}个样本)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        pred_fig_path = os.path.join(
            result_save_folder,
            f"prediction_vs_actual_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(pred_fig_path, dpi=150)
        plt.close()
        print(f"预测对比图已保存至: {pred_fig_path}")

    # ========== 新增：生成10条预测样本 ==========
    print("\n" + "=" * 80)
    print("训练完成！正在生成预测样本...")
    print("=" * 80)

    # 确保使用测试数据生成样本
    data.generateMixLengthBatch(batch_size)

    # 生成样本文件路径
    sample_file_path = os.path.join(
        result_save_folder,
        f"prediction_samples_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # 生成10条预测样本
    samples = generate_prediction_samples(
        model=model,
        data=data,
        num_samples=10,
        sample_file_path=sample_file_path
    )

    print(f"\n预测样本生成完成！")
    print(f"共生成 {len(samples)} 条样本")
    print(f"样本文件保存路径: {sample_file_path}")

    # 计算样本统计信息
    if samples:
        errors = [sample['absolute_error'] for sample in samples]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)

        print(f"\n样本统计信息:")
        print(f"平均绝对误差: {avg_error:.4f}")
        print(f"最大绝对误差: {max_error:.4f}")
        print(f"最小绝对误差: {min_error:.4f}")

    print("=" * 80)


def evaluate(model, test_batchs):
    target_list = list()
    predict_list = list()
    for (input, target) in test_batchs:
        input = np.array(input)
        input = Variable(torch.LongTensor(input).cuda())
        prediction = model(input)
        predict_list += [pdic.item() for pdic in prediction]
        target_list += target

    MSE = computeMSE(target_list, predict_list)
    MAE = computeMAE(target_list, predict_list)
    RMSE = sqrt(MSE)
    TOTAL = computeTOTAL(target_list, predict_list)
    MEAN = computeMEAN(target_list, predict_list)

    # 添加这一行，返回预测值和真实值
    return MSE, MAE, RMSE, TOTAL, MEAN, predict_list, target_list


def computeMAE(list_a, list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num] - list_b[num]))
    MAE = sum(MAE_temp) / len(list_a)
    return MAE


def computeMSE(list_a, list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE


def computeTOTAL(list_a, list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL


def computeMEAN(list_a, list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp) / len(list_a)
    return MEAN





start_time = time.time()  # 记录开始时间
train('F:\python_project\数据集\RemainTimePrediction_25_948\data\codeforces948_4.csv', data_name='codeforces948_4',
      vector_address='./vector/Codeforces948_4_vectors_GloVe_concepts.txt',
      vocab_address='./vector/Codeforces948_4glove_vocabulary_a.txt',
      embd_dimension=3, train_splitThreshold=0.8, time_unit='minute', batch_size=32)
end_time = time.time()  # 记录结束时间
print(f"运行时间：{end_time - start_time} 秒")

# 创建一个空白的 txt 文件
now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
file_name = 'result\\' + '---------------------------分割------------------------' + now_time
with open(file_name, 'w', encoding='utf-8') as file:
    pass  # 这里使用 pass 语句，不进行任何写入操作(
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
    save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
    save_record_all_write = model_type + ',' + str(embd_dimension) + ',' + loss_type \
                            + ',' + optim_type + ',' + str(hidden_dim) + ',' + str(start_pos) \
                            + ',' + train_type + ',' + str(n_layer) + ',' + str(dropout) \
                            + ',' + str(epoch) + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                            + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
        total_loss.item()) \
                            + ',' + model_save + ',' + save_record_single + ',' + result_save_file + '\n'

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
    result_save_open = open(result_save_file, 'w', encoding='utf-8')
    result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
    result_save_open.writelines(result_save_write)
    result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(
        MSE) + ',' + str(MAE) \
                        + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
total_loss.item()) + '\n'
result_save_open.writelines(result_save_write)
for prefix_length in range(start_pos, stop_pos + 1):
# if prefix_length % 2 != 0 and prefix_length != 5:
# continue
    data.generateSingleLengthBatch(batch_size, prefix_length)
if len(data.test_batch) != 0:
    MSE1, MAE1, RMSE1, TOTAL1, MEAN1, _, _ = evaluate(model, data.test_batch)
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
save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
save_record_all_write = model_type + ',' + str(embd_dimension) + ',' + loss_type \
                        + ',' + optim_type + ',' + str(hidden_dim) + ',' + str(start_pos) \
                        + ',' + train_type + ',' + str(n_layer) + ',' + str(dropout) \
                        + ',' + str(epoch) + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                        + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
    total_loss.item()) \
                        + ',' + model_save + ',' + save_record_single + ',' + result_save_file + '\n'
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
print(MSE, MAE, RMSE, TOTAL, total_loss.item(), epoch, learn_rate, loss_change)

save_record_single_write = str(epoch) + ',' + str(start_pos) + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                           + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
    total_loss.item()) + '\n'
save_record_single_open.writelines(save_record_single_write)
# print(loss_change)
epoch = epoch + 1

save_record_single_open.close()

# ========== 没有触发早停机制：保存最终模型和csv ================
now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
model_save = save_model_folder + now_time + '_final.pth'
torch.save(model, model_save)
result_save_file = result_save_folder + str(model_type) + '_final_' + now_time + '.csv'
if not os.path.exists(result_save_folder):
    os.makedirs(result_save_folder)
with open(result_save_file, 'w', encoding='utf-8') as result_save_open:
    result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
result_save_open.writelines(result_save_write)
# 先生成混合长度的batch和评估
data.generateMixLengthBatch(batch_size)
MSE, MAE, RMSE, TOTAL, MEAN, _, _ = evaluate(model, data.test_batch)
result_save_write = str(epoch) + ',' + str(start_pos) + ',mix,' + str(learn_rate) + ',' + str(MSE) + ',' + str(
MAE) \
    + ',' + str(RMSE) + ',' + str(train_loss_list[-1]) + ',' + str(TOTAL) + '\n'
result_save_open.writelines(result_save_write)
# 再分别对每个前缀长度评估
for prefix_length in range(start_pos, stop_pos + 1):
    data.generateSingleLengthBatch(batch_size, prefix_length)
if len(data.test_batch) != 0:
    MSE1, MAE1, RMSE1, TOTAL1, MEAN1, _, _ = evaluate(model, data.test_batch)
result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str