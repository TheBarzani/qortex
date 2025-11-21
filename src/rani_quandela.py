import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

# --- Quandela / MerLin ---
import merlin as ML  # pip install merlinquantum

# ============================================================
# 0. LOAD XLSX + remove first column
# ============================================================
raw = pd.read_excel("train.xlsx").values
dataset_train = raw[:, 1:]


# ============================================================
# 1. PCA 2D
# ============================================================
def compute_pca(data):
    pca = PCA(n_components=2)
    data2 = pca.fit_transform(data)
    return data2, pca


# ============================================================
# 2. Windows t1–t3 → t4–t6
# ============================================================
def make_windows(data2):
    X, Y = [], []
    T = len(data2)
    for t in range(T - 6):
        X.append(data2[t:t+3])      # shape (3, 2)
        Y.append(data2[t+3:t+6])    # shape (3, 2)
    return np.array(X), np.array(Y)


# ============================================================
# 3. Dataset
# ============================================================
class TimeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ============================================================
# 4. MerLin quantum layer (simple interface)
#    Input: (B, 3, 2) -> flatten to (B, 6)
# ============================================================
class MerlinTimeModel(nn.Module):
    def __init__(self, input_size=6, n_params=50):
        """
        Simple hybrid model:
         - MerLin QuantumLayer.simple for the quantum block
         - Linear(quantum_output_size -> 6) to map back to 3x2 window
        """
        super().__init__()

        # Photonic quantum layer of size=6, output size=variable
        self.q_layer = ML.QuantumLayer.simple(
            input_size=input_size,
            n_params=n_params
        )

        # of size quantum_output_size, output size=6 (3x2 window)
        self.fc = nn.Linear(self.q_layer.output_size, 6)

    def forward(self, x):
        """
        x: (B, 3, 2)
        returns: (B, 3, 2)
        """
        B = x.shape[0]
        # Flatten 3x2 window -> 6 features
        x_flat = x.view(B, -1)              # (B, 6)

        # Quantum forward pass
        q_out = self.q_layer(x_flat)        # (B, q_dim)

        # Classical linear head
        y = self.fc(q_out)                  # (B, 6)

        # Reshape back to (3, 2) window
        return y.view(B, 3, 2)


# ============================================================
# 5. Training
# ============================================================
def train_model(dataset_train, epochs=20, bs=8, n_params=50):
    # 1) PCA to 2D
    data2, pca = compute_pca(dataset_train)

    # 2) Build windows t1–t3 → t4–t6
    X, Y = make_windows(data2)

    # 3) Dataloader
    loader = DataLoader(TimeDataset(X, Y), batch_size=bs, shuffle=True)

    # 4) Hybrid MerLin model
    model = MerlinTimeModel(input_size=6, n_params=n_params)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 5) Training loop
    for epoch in range(epochs):
        tot = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
        print("epoch", epoch, "loss", tot / len(loader))

    return model, pca


# ============================================================
# 6. Autoregressive prediction
# ============================================================
def generate_next(model, pca, dataset_train, steps=2):
    # Take last 3 original points (high-dim)
    last3 = dataset_train[-3:]

    # Project them to PCA space (3, 2)
    seq = pca.transform(last3)

    gen = []

    for _ in range(steps):
        # Current context: last 3 PCA points
        inp = torch.tensor(seq[-3:], dtype=torch.float32).unsqueeze(0)  # (1,3,2)
        out = model(inp)[0].detach().numpy()                            # (3,2)

        gen.append(out)
        # Append predicted 3 steps to the sequence
        seq = np.vstack([seq, out])

    gen = np.vstack(gen)  # shape (3*steps, 2)

    # Inverse PCA back to original feature space
    return pca.inverse_transform(gen)


# ============================================================
# RUN TRAINING + GENERATION
# ============================================================
if __name__ == "__main__":
    model, pca = train_model(dataset_train)

    future_points = generate_next(model, pca, dataset_train, steps=2)

    # ========================================================
    # 7. Build final dataset (Tenor/Maturity grid)
    # ========================================================
    tenors = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
    maturities = [
        0.0833333333333333, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10,
        15, 20, 25, 30
    ]

    columns = []
    for mat in maturities:
        for ten in tenors:
            columns.append(f"Tenor : {ten}; Maturity : {mat}")

    dates = [
        "24/12/2051",
        "26/12/2051",
        "27/12/2051",
        "29/12/2051",
        "30/12/2051",
        "01/01/2052"
    ]

    N = len(dates)
    final_data = future_points[:N, :len(columns)]

    df_final = pd.DataFrame(final_data, columns=columns)
    df_final.insert(0, "Date", dates)

    print(df_final)
    df_final.to_excel("final_dataset.xlsx", index=False)

import perceval as pcvl

# Draw circuit
q_layer = model.q_layer        
circ = q_layer.circuit

pcvl.pdisplay(circ)

from perceval import RemoteConfig, RemoteProcessor
PLATFORM_SIM = "sim:slos"
PLATFORM_QPU = "qpu:ascella"   # or "qpu:belenos" etc.
# Save your token and proxy configuration into Perceval persistent data, you only need to do it once per machine.
remote_config = RemoteConfig()
remote_config.set_token("mytoken")  # Replace with your Token from the Cloud
remote_config.save()
remote_processor = RemoteProcessor(PLATFORM_QPU)
remote_processor.min_detected_photons_filter(1)