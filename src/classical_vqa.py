import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

# ============================================================
# 0. LOAD XLSX + remove first column
# ============================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "Track2_QML", "xlsx", "train.xlsx")

raw = pd.read_excel(TRAIN_PATH).values
dataset_train = raw[:, 1:]



# ============================================================
# 1. PCA 2D
# ============================================================
def compute_pca(data):
    pca = PCA(n_components=2)
    data2 = pca.fit_transform(data)
    return data2, pca


# ============================================================
# 2. Windows
# ============================================================
def make_windows(data2):
    X, Y = [], []
    T = len(data2)
    for t in range(T - 6):
        X.append(data2[t:t+3])
        Y.append(data2[t+3:t+6])
    return np.array(X), np.array(Y)


# ============================================================
# 3. Dataset
# ============================================================
class TimeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ============================================================
# 4. Tiny classical “feature” layer (24 params total)
# ============================================================
class ClassicalObsLayer(nn.Module):
    """
    Classical analogue of QiskitObsLayer.

    - n = 6 features (flattened 3x2 window)
    - reps = 4 “layers”
    - trainable parameters: theta of shape (reps, n) = 24 scalars
    """
    def __init__(self, n=6, reps=4):
        super().__init__()
        self.n = n
        self.reps = reps

        # 24 trainable parameters, like the quantum thetas
        self.theta = nn.Parameter(0.01 * torch.randn(reps, n))

    def forward(self, x):
        # x: (B, 3, 2) -> (B, 6)
        x_flat = x.view(-1, self.n)        # (B, 6)

        # Broadcast: (B, reps, n)
        # Use a simple nonlinear mixing that vaguely plays the role of
        # "encoding + entangling + measurement"
        phase = x_flat.unsqueeze(1) * self.theta.unsqueeze(0)  # (B, reps, n)

        # Nonlinearity + aggregation over reps
        # (you can swap tanh for sin/cos if you want)
        out = torch.mean(torch.tanh(phase), dim=1)  # (B, n)

        return out                                # (B, 6)


# ============================================================
# 5. Full model (no extra trainable params beyond 24)
# ============================================================
class ClassicalModel(nn.Module):
    def __init__(self, n=6, reps=4):
        super().__init__()
        self.q = ClassicalObsLayer(n=n, reps=reps)   # 24 params

    def forward(self, x):
        q_out = self.q(x)           # (B, 6)
        return q_out.view(-1, 3, 2) # match target shape


# ============================================================
# 6. Training (same as before, just swap Model -> ClassicalModel)
# ============================================================
def train_model(dataset_train, epochs=20, bs=16):
    data2, pca = compute_pca(dataset_train)
    X, Y = make_windows(data2)

    loader = DataLoader(TimeDataset(X, Y), batch_size=bs, shuffle=True)

    model = ClassicalModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # sanity check: should print 24
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)

    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print("epoch", epoch, "loss", total / len(loader))

    return model, pca


# ============================================================
# 7. Autoregressive generation (same logic)
# ============================================================
def generate_next(model, pca, dataset_train, steps=2):
    last3 = dataset_train[-3:]
    seq = pca.transform(last3)

    gen = []

    for _ in range(steps):
        inp = torch.tensor(seq[-3:], dtype=torch.float32).unsqueeze(0)
        pred = model(inp)[0].detach().numpy()
        gen.append(pred)
        seq = np.vstack([seq, pred])

    gen = np.vstack(gen)
    return pca.inverse_transform(gen)


# ============================================================
# TRAIN + GENERATE
# ============================================================
if __name__ == "__main__":
    model, pca = train_model(dataset_train)
    future_points = generate_next(model, pca, dataset_train, steps=2)