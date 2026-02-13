import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report
)
import numpy as np
from mlp_train import data_process

def topk_selection(k, y_pred_proba, y_test, mint_list_test, mint2label_info):
    topk_idx = np.argsort(-y_pred_proba)[:k]
    topk_y = [y_test[i] for i in topk_idx]
    topk_mints = [mint_list_test[i] for i in topk_idx]
    topk_probs = y_pred_proba[topk_idx]
    topk_pred = (topk_probs >= 0.5).astype(int)
    print(classification_report(topk_y, topk_pred, digits=4))
    return_ratio_list = []
    for idx in topk_idx:
        mint = mint_list_test[idx]
        label_info = mint2label_info[mint]
        return_ratio = label_info["return_ratio"]
        if return_ratio > 0 and label_info["pred_proba"] > 0.6:
            continue
        return_ratio_list.append(return_ratio)

    return_ratio_list = np.array(return_ratio_list)
    avg_return = return_ratio_list.mean()
    median_return = np.median(return_ratio_list)
    wins = return_ratio_list[return_ratio_list >= 0]
    losses = return_ratio_list[return_ratio_list < 0]
    num_trades = len(return_ratio_list)
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = num_wins / num_trades if num_trades > 0 else np.nan
    if num_losses > 0:
        win_loss_ratio = wins.sum() / abs(losses.sum())
    else:
        win_loss_ratio = np.inf

    print("avg_return:", avg_return)
    print("median_return:", median_return)
    print("win rate:", win_rate)
    print("win/loss:", win_loss_ratio)

    return avg_return, median_return, win_rate, win_loss_ratio


if __name__ == '__main__':

    X_train, y_train, X_test, y_test, mint_list_test, mint2label_info, feature_cols = data_process()
    # model_df = pd.read_csv("../results/mlp_pred_6_0.572861123678359.csv")
    # model_df = pd.read_csv("../results/lgbm_pred_0.5674679411984582.csv")
    model_df = pd.read_csv("../results/rf_pred_0.5684831293402106.csv")
    # model_df = pd.read_csv("../results/lr_pred_0.5337647849222107.csv")
    # model_df = pd.read_csv("../results/xgb_pred_0.5628290207732173.csv")

    y_pred_proba = model_df["prob"].values
    y_test = model_df["label"].values

    # y_pred_proba = np.random.rand(len(X_test))
    auprc = average_precision_score(y_test, y_pred_proba)
    print("ensemble AUPRC:", auprc)

    for k in [100, 200]:

        print("==============================")
        print("k:", k)
        avg_return, median_return, win_rate, win_loss_ratio = topk_selection(k, y_pred_proba, y_test, mint_list_test,
                                                                             mint2label_info)


