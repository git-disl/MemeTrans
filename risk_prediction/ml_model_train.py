import json
import pandas as pd
from collections import defaultdict
import pytz
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import pickle as pkl
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
bundled_meme_cnt = 0
no_mint_cnt = 0
import argparse

def data_process():

    Xy_df = pd.read_csv("../data/feat_label.csv")
    Xy_df = Xy_df[(Xy_df["group3_time_span_valid"] >= 60) & (Xy_df["group3_holder_num"] >= 100)]

    Xy_df = Xy_df.sample(frac=1, random_state=42)
    y_df = Xy_df[['token', 'pred_proba', 'min_ratio', 'return_ratio', 'label']]
    mint2label_info = y_df.set_index("token").to_dict(orient="index")

    feature_cols = ['mint_address', 'mint_ts', 'label'] + [c for c in Xy_df.columns if "group" in c]
    Xy_df = Xy_df[feature_cols]

    unnormed_df = Xy_df[["mint_address", "mint_ts", "label"]]
    feature_cols = [c for c in Xy_df.columns if "group" in c]
    mint_list = list(unnormed_df["mint_address"].values)
    X = Xy_df.drop(columns=["mint_address", "mint_ts", "label"])
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    y = np.where(unnormed_df["label"] == "high", 0, 1)

    split_idx = int(len(Xy_df) * 0.7)
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    X_train = X[:split_idx]
    X_test = X[split_idx:]

    mint_list_test = mint_list[split_idx:]

    return X_train, y_train, X_test, y_test, mint_list_test, mint2label_info, feature_cols


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb", "lgb", "mlp"],
        help="Model type to use (default: rf)"
    )
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, mint_list_test, mint2label_info, feature_cols = data_process()

    metric_list = []
    random_seed = random.randint(0, 2 ** 32 - 1)
    metric = []

    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=800,
            max_depth=16,
            # max_depth=None,
            min_samples_leaf=3,
            min_samples_split=6,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_seed,
            max_features="sqrt"
        )
    elif args.model == "lgbm":
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        )
    elif args.model == "lr":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=random_seed,
            multi_class="auto",
        )
    elif args.model == "xgb":
        pos_ratio = y_train.mean()
        neg_ratio = 1 - pos_ratio
        scale_pos_weight = neg_ratio / pos_ratio
        print("scale_pos_weight:", scale_pos_weight)

        model = XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1.0,
            reg_alpha=0.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_seed,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        )
    else:
        raise ValueError("Incorrect Model")
    #

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auprc = average_precision_score(y_test, y_proba)
    print("AUPRC:", auprc)

    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred_proba >= 0.53).astype(int)  # [0, 1] 预测
    print(classification_report(y_test, y_pred, digits=4))
    print("pause")

    df = pd.DataFrame({
        "mint": mint_list_test,
        "label": y_test,
        "prob": y_pred_proba
    })

    df.to_csv(f"../results/{args.model}_pred_{auprc}.csv", index=False)

    if args.model == "rf":
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [feature_cols[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]

        color_map = {
            "group1_": "tab:blue",
            "group2_": "tab:orange",
            "group3_": "tab:green",
            "group4_": "tab:red",
        }

        colors = []
        for feat in sorted_features:
            for prefix, color in color_map.items():
                if feat.startswith(prefix):
                    colors.append(color)
                    break
            else:
                colors.append("gray")

        plt.figure(figsize=(16, 10))
        plt.barh(sorted_features[:80], sorted_importances[:80], color=colors[:80])
        plt.xlabel("Importance Score")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

