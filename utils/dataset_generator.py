import torch

def generate(device):
    splitter_coefficient = .8
    X = torch.arange(start=-1, end=1, step=.01, device=device).unsqueeze(1)
    Y = 2.3 * X * X - 5.1 * X + 0.65
    split = int(len(X) * splitter_coefficient)
    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]
    return X_train, Y_train, X_test, Y_test