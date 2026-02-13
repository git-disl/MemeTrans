import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import pickle as pkl
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from ml_model_train import data_process

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 输出 shape: (batch,)
        return self.net(x).squeeze(1)


if __name__ == '__main__':

    X_train, y_train, X_test, y_test, mint_list_test, mint2label_info, feature_cols = data_process()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 转成 torch.Tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = MLP(input_dim).to(device)

    pos_weight_value = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
    print("pos_weight:", pos_weight.item())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    auprc_list = []
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        # ========= Train =========
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)  # (batch,)
            loss = criterion(logits, batch_y)  # BCEWithLogitsLoss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        # ========= Eval on test set =========
        model.eval()
        y_pred_proba = []

        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                y_pred_proba.append(probs.cpu())

        y_pred_proba = torch.cat(y_pred_proba).numpy().reshape(-1)
        y_pred = (y_pred_proba >= 0.65).astype(int)  # [0, 1]

        auprc = average_precision_score(y_test, y_pred_proba)
        print(f"Epoch:{epoch}, AUPRC:", auprc)
        # print(f"\nEpoch [{epoch}/{num_epochs}] - Train Loss: {epoch_loss:.4f}")
        print(classification_report(y_test, y_pred, digits=4))

        df = pd.DataFrame({
            "mint": mint_list_test,
            "label": y_test,
            "prob": y_pred_proba
        })
        df.to_csv(f"../results/mlp_pred_{epoch}_{auprc}.csv", index=False)



