import time
import warnings
import gensim
from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Event log
eventlog = 'Codeforces948_4'
dim = 3
window_size = 1
problem2concept = {
    "A": "math",
    "B": "bitmasks constructive algorithms greedymath",
    "C": "brute force data structures dp greedy math number theory sortings",
    "D": "bitmasks brute force greedy hashing",
    "E": "constructive algorithms graphs interactive"
}


def readcsv(eventlog, problem2concept):
    csvfile = open('data/%s' % eventlog, 'r', encoding='utf-8')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    sequence = []
    next(spamreader, None)  # skip the headers
    for line in spamreader:
        # 替换题目标识符为知识点概念
        original_problem = line[1]
        if original_problem in problem2concept:
            line[1] = problem2concept[original_problem]
        sequence.append(line)
    return sequence


data = readcsv(eventlog + '.csv', problem2concept)


# Vocabulary creation - 修改为处理知识点概念
def makeVocabulary(data, eventlog):
    temp = list()
    for line in data:
        # 将完整的知识点概念作为一个词汇项
        temp.append(line[1])

    temp_temp = set(temp)
    vocabulary = {sorted(list(temp_temp))[i]: i + 1 for i in range(len(temp_temp))}
    vocabulary['0'] = 0
    vocabulary['end'] = len(vocabulary)

    f = open('vector/%s' % eventlog + 'glove_vocabulary' + '.txt', 'w', encoding='utf-8')
    for k in vocabulary:
        f.write(str(k) + '\t' + str(vocabulary[k]) + '\n')
    return vocabulary


vocabulary = makeVocabulary(data, eventlog)


# Data preprocessing - 修改为处理知识点概念
def processData(data, vocabulary, eventlog):
    front = data[0]
    data_new = []
    time_code_temp = {}
    time_code = {}

    for line in data[1:]:
        temp = 0
        if line[0] == front[0]:
            temp1 = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
            temp2 = time.strptime(front[2], "%Y-%m-%d %H:%M:%S")
            temp = datetime.fromtimestamp(time.mktime(temp1)) - datetime.fromtimestamp(time.mktime(temp2))
        else:
            temp = 0
        t = time.strptime(line[2], "%Y-%m-%d %H:%M:%S")
        week = datetime.fromtimestamp(time.mktime(t)).weekday()
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight

        # 将知识点概念转换为对应的ID
        concept_id = vocabulary.get(line[1], 0)  # 如果概念不在词汇表中，使用0

        data_new.append([line[0], concept_id, line[2], timesincemidnight, week])
        front = line

    # 构建时间编码
    front = data_new[0]
    for row in range(1, len(data_new)):
        line = data_new[row]
        if line[0] == front[0]:
            key = str(line[1]) + '-' + str(front[1])
            if key not in time_code_temp:
                time_code_temp[key] = []
                time_code_temp[key].append(line[3].seconds)
            else:
                time_code_temp[key].append(line[3].seconds)
        front = data_new[row]

    for key in time_code_temp:
        time_code_temp[key] = sorted(time_code_temp[key])

    # 构建数据合并
    data_merge = []
    data_temp = [data_new[0]]
    for line in data_new[1:]:
        if line[0] != data_temp[-1][0]:
            data_merge.append(data_temp)
            data_temp = [line]
        else:
            data_temp.append(line)
    data_merge.append(data_temp)

    vocabulary_num = len(vocabulary)
    vocabulary_temp = vocabulary
    return data_merge, data_new, time_code_temp, vocabulary_num, vocabulary_temp


data_merge, data_new, time_code_temp, vocabulary_num, vocabulary = processData(data, vocabulary, eventlog)


# Build co-occurrence matrix - 处理概念序列
def build_cooccurrence_matrix(corpus, vocab_size, window_size):
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for sequence in corpus:
        # 每个序列中的条目现在包含单个概念ID
        for i, center_entry in enumerate(sequence):
            center_id = center_entry[1]
            start = max(0, i - window_size)
            end = min(len(sequence), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_id = sequence[j][1]
                    distance = abs(j - i)
                    cooccurrence_matrix[center_id, context_id] += 1.0 / distance
    return cooccurrence_matrix


# Prepare training data
def prepare_training_data(cooccurrence_matrix):
    word_i_data = []
    word_j_data = []
    cooccurrence_data = []
    for i in range(len(cooccurrence_matrix)):
        for j in range(len(cooccurrence_matrix[i])):
            if cooccurrence_matrix[i, j] > 0:
                word_i_data.append(i)
                word_j_data.append(j)
                cooccurrence_data.append(cooccurrence_matrix[i, j])
    word_i_data = np.array(word_i_data)
    word_j_data = np.array(word_j_data)
    cooccurrence_data = np.array(cooccurrence_data)
    return word_i_data, word_j_data, cooccurrence_data


class GloveModel(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.wi = nn.Embedding(vocab_size, emb_dim)
        self.wj = nn.Embedding(vocab_size, emb_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

    def forward(self, word_i, word_j):
        wi = self.wi(word_i)
        wj = self.wj(word_j)
        bi = self.bi(word_i).squeeze(-1)
        bj = self.bj(word_j).squeeze(-1)
        dot = (wi * wj).sum(dim=1)
        pred = dot + bi + bj
        return pred


def glove_loss(pred, cooccurrence, x_max=10.0, alpha=0.75):
    weight = torch.pow(torch.clamp(cooccurrence / x_max, max=1.0), alpha)
    loss = weight * (pred - torch.log(cooccurrence + 1)) ** 2
    return loss.mean()


# ========== PyTorch训练主流程 ==========
def train_glove(word_i_data, word_j_data, cooccurrence_data, vocabulary_num, dim, eventlog):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 转为Tensor并放到GPU/CPU
    word_i_data = torch.LongTensor(word_i_data)
    word_j_data = torch.LongTensor(word_j_data)
    cooccurrence_data = torch.FloatTensor(cooccurrence_data)
    dataset_size = word_i_data.shape[0]

    data = TensorDataset(word_i_data, word_j_data, cooccurrence_data)
    val_size = int(0.3 * dataset_size)
    train_size = dataset_size - val_size
    train_data, val_data = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    model = GloveModel(vocabulary_num, dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 2000
    patience = 80
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for wi, wj, cooc in train_loader:
            wi, wj, cooc = wi.to(device), wj.to(device), cooc.to(device)
            optimizer.zero_grad()
            pred = model(wi, wj)
            loss = glove_loss(pred, cooc)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * wi.size(0)
        train_loss = total_loss / train_size
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for wi, wj, cooc in val_loader:
                wi, wj, cooc = wi.to(device), wj.to(device), cooc.to(device)
                pred = model(wi, wj)
                loss = glove_loss(pred, cooc)
                total_val_loss += loss.item() * wi.size(0)
        val_loss = total_val_loss / val_size
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'vector/{eventlog}_GloVe_concepts_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # 损失曲线绘制
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train & Validation Loss Curve (Concept-based GloVe)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'vector/{eventlog}_glove_concepts_loss_curve.png', dpi=150)
    plt.close()
    print(f"训练与验证损失曲线已保存为 vector/{eventlog}_glove_concepts_loss_curve.png")

    # 恢复最好模型并导出词向量
    model.load_state_dict(torch.load(f'vector/{eventlog}_GloVe_concepts_best.pt'))
    word_vectors = model.wi.weight.data.cpu().numpy() + model.wj.weight.data.cpu().numpy()

    # 保存概念向量，同时保存概念到ID的映射
    with open(f'vector/{eventlog}_vectors_GloVe_concepts.txt', 'w') as f:
        f.write(f"{vocabulary_num} {dim}\n")
        for word_id in range(vocabulary_num):
            vec = word_vectors[word_id]
            f.write(f"{word_id} {' '.join(map(str, vec))}\n")

    # 保存概念名称映射
    reverse_vocab = {v: k for k, v in vocabulary.items()}
    with open(f'vector/{eventlog}_concept_mapping.txt', 'w') as f:
        for word_id in range(vocabulary_num):
            concept_name = reverse_vocab.get(word_id, f"unknown_{word_id}")
            f.write(f"{word_id}\t{concept_name}\n")

    print(f"概念向量已保存为 vector/{eventlog}_vectors_GloVe_concepts.txt")
    print(f"概念映射已保存为 vector/{eventlog}_concept_mapping.txt")


if __name__ == '__main__':
    cooccurrence_matrix = build_cooccurrence_matrix(data_merge, vocabulary_num, window_size)
    word_i_data, word_j_data, cooccurrence_data = prepare_training_data(cooccurrence_matrix)
    train_glove(word_i_data, word_j_data, cooccurrence_data, vocabulary_num, dim, eventlog)