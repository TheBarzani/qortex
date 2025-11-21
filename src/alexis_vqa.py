import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp

# ============================================================
# 0. LOAD XLSX + remove first column
# ============================================================

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
# 2. Windows t1–t3 → t4–t6
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
# 4. NEW PARAMETRIC CIRCUIT — exactly as you want
# ============================================================
def make_feature_circuit(n=6, reps=4):
    qc = QuantumCircuit(n)

    # ---- 1) ENCODAGE DATA (UNE SEULE FOIS) ----
    data_params = []
    for q in range(n):
        p = Parameter(f"data_{q}")
        qc.ry(p, q)
        data_params.append(p)

    # ---- 2) BLOCS THETA + CASCADE CZ ----
    theta_params = []

    for r in range(reps):
        row = []
        # couche RY(theta)
        for q in range(n):
            t = Parameter(f"theta_{r}_{q}")
            qc.ry(t, q)
            row.append(t)
        theta_params.append(row)

        # cascade CZ
        for q in range(n - 1):
            qc.cz(q, q + 1)

    return qc, data_params, theta_params


# ============================================================
# 5. Quantum layer <Z_i>
# ============================================================
class QiskitObsLayer(nn.Module):
    def __init__(self, n=6, reps=4):
        super().__init__()
        self.n = n
        self.reps = reps

        self.template, self.data_params, self.theta_params = make_feature_circuit(n, reps)
        self.theta = nn.Parameter(0.01 * torch.randn(reps, n))

        self.obs = [
            SparsePauliOp.from_list([("I"*i + "Z" + "I"*(n-i-1), 1.0)])
            for i in range(n)
        ]

    def forward(self, x):
        B = x.shape[0]
        out = []

        for b in range(B):
            # x[b] = (3,2) → flatten to 6 angles, encoded ONCE
            angles = x[b].reshape(6).tolist()

            assign = {}

            # data encoding
            for q in range(self.n):
                assign[self.data_params[q]] = float(angles[q])

            # theta blocks
            for r in range(self.reps):
                for q in range(self.n):
                    assign[self.theta_params[r][q]] = float(self.theta[r, q].detach())

            circ = self.template.assign_parameters(assign)
            state = Statevector.from_instruction(circ)

            vals = []
            for obs in self.obs:
                vals.append(float(state.expectation_value(obs).real))

            out.append(vals)

        return torch.tensor(out, dtype=torch.float32)


# ============================================================
# 6. Full model
# ============================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = QiskitObsLayer()
        self.fc = nn.Linear(6, 6)
    def forward(self, x):
        q_out = self.q(x)
        y = self.fc(q_out)
        return y.view(-1, 3, 2)


# ============================================================
# 7. Training
# ============================================================
def train_model(dataset_train, epochs=20, bs=8):
    data2, pca = compute_pca(dataset_train)
    X, Y = make_windows(data2)

    loader = DataLoader(TimeDataset(X, Y), batch_size=bs, shuffle=True)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        tot = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item()
        print("epoch", epoch, "loss", tot/len(loader))

    return model, pca


# ============================================================
# 8. Autoregressive prediction
# ============================================================
def generate_next(model, pca, dataset_train, steps=2):
    last3 = dataset_train[-3:]
    seq = pca.transform(last3)

    gen = []

    for _ in range(steps):
        inp = torch.tensor(seq[-3:], dtype=torch.float32).unsqueeze(0)
        out = model(inp)[0].detach().numpy()
        gen.append(out)
        seq = np.vstack([seq, out])

    gen = np.vstack(gen)
    return pca.inverse_transform(gen)


# ============================================================
# RUN TRAINING + GENERATION
# ============================================================
model, pca = train_model(dataset_train)
future_points = generate_next(model, pca, dataset_train, steps=2)


# ============================================================
# 9. Build final dataset (Tenor/Maturity grid)
# ============================================================
tenors = [1,2,3,4,5,6,7,8,9,10,15,20,25,30]
maturities = [0.0833333333333333, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]

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
df_final.to_csv("final_dataset.csv", index=False)
