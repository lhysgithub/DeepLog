import torch
import torch.nn as nn
import time
import argparse
import numpy as np

# Device configuration
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_by_mask(line: str):
    mask = "0. 1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1"
    mask = [int(i) for i in mask.split(".")]
    for i in range(len(mask)):
        if mask[i] == 0:
            target = str(int(i+1))
            line = line.replace(target, "")
    return line


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    # hdfs = set()
    hdfs = []
    with open('data/' + name, 'r') as f:
        # temp_lines = f.readlines()
        # print(f"File length： {len(temp_lines)}")
        for ln in f.readlines():
            ln = filter_by_mask(ln)
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            # hdfs.add(tuple(ln))
            hdfs.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def record_abnormal_one_hot(seq, max_id=28):
    one_hot_line = np.zeros(max_id)
    for o in seq:
        if o != -1:
            one_hot_line[o] = 1
    one_hot_str = ",".join([str(int(i)) for i in one_hot_line])
    with open("data/one_hot_abnormal_predict.csv", "a") as a:
        a.write(one_hot_str + "\n")


if __name__ == '__main__':

    # Hyperparameters
    num_classes = 28
    input_size = 1
    model_path = 'model/Adam_batch_size=2048_epoch=300.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_normal_loader = generate('hdfs_test_normal')
    # test_normal_loader = generate('hdfs_train')
    test_abnormal_loader = generate('hdfs_test_abnormal')
    TP = 0
    FP = 0
    # Test the model
    start_time = time.time()
    with torch.no_grad():
        for line in test_normal_loader:
            flag = False
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    flag = True
                    FP += 1
                    break
            if flag:
                record_abnormal_one_hot(line, num_classes)
    with torch.no_grad():
        for line in test_abnormal_loader:
            flag = False
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    flag = True
                    TP += 1
                    break
            if flag:
                record_abnormal_one_hot(line, num_classes)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))
    # Compute precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('Finished Predicting')


