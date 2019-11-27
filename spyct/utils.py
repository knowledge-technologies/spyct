import numpy as np


def ranking_loss(labels, predictions):
    n, l = labels.shape
    loss = 0
    for i in range(n):
        oopsies = 0
        pos = 0
        for j in range(l):
            if labels[i, j] == 1:
                pos += 1
                for k in range(l):
                    if labels[i, k] == 0 and predictions[i, k] >= predictions[i, j]:
                        oopsies += 1
        loss += oopsies / (pos * (l - pos))
    return loss / n


def precision(labels, predictions, k, sparse=False):
    n, l = labels.shape
    prec = 0
    for i in range(n):
        if sparse:
            idx_sorted = np.argsort(predictions[i].A.squeeze())
        else:
            idx_sorted = np.argsort(predictions[i])
        prec += np.sum(labels[i, idx_sorted[-k:]]) / k
    return prec / n


def parse_arff(path):
    rows = []
    with open(path) as f:
        for line in f:
            if '@data' in line.lower():
                break
        for line in f:
            rows.append([float(x) for x in line.strip().split(',')])
    return np.array(rows)


def parse_sparse_arff(path, attributes):
    rows = []
    with open(path) as f:
        for line in f:
            if '@data' in line.lower():
                break
        for line in f:
            row = np.zeros(attributes)
            pairs = line.strip()[1:-1].split(',')
            for p in pairs:
                idx, val = p.split()
                row[int(idx)-1] = float(val)
            rows.append(row)
    return np.array(rows)


def parse_clus_predictions(path, models):
    num_targets = 0
    with open(path) as f:
        for line in f:
            if '@attribute' in line.lower():
                num_targets += 1
            if '@data' in line.lower():
                break
        num_targets = (num_targets-models) // (3*models+1)
        labels = [[] for _ in range(models+1)]
        for line in f:
            values = line.strip().split(',')
            labels[0].append([float(x) for x in values[:num_targets]])
            for m in range(models):
                labels[m+1].append([float(x) for x in values[(2+3*m)*num_targets+m:(4+3*m)*num_targets+m:2]])

    labels = [np.array(lst) for lst in labels]
    return tuple(labels)
