import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from clasifier import MLPClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from tabnet import train_tabnet_model, predict_tabnet_submission
from pytorch_tabnet.tab_model import TabNetClassifier
from train import load_data, concat_monthly_data, reduce_data_fraction, merge_all_data, preprocess

# ============================
# 1. 설정값
# ============================
DATA_SPLITS = ["train", "test"]
MONTHS = ['07', '08', '09', '10', '11', '12']
DATA_CATEGORIES = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}
INFO_CATEGORIES = [v["var_prefix"] for v in DATA_CATEGORIES.values()]

# (생략: 데이터 관련 함수들 그대로 유지)

# ============================
# 8. 전체 실행 함수
# ============================
def main():
    print("=== Load ===")
    load_data()

    print("=== Concat ===")
    train_dfs = concat_monthly_data("train")
    train_dfs = reduce_data_fraction(train_dfs, split="train", frac=0.7)

    test_dfs = concat_monthly_data("test")

    print("=== Merge ===")
    train_df = merge_all_data(train_dfs, "train")
    test_df = merge_all_data(test_dfs, "test")

    print("=== Preprocess ===")
    feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment"]]
    X = train_df[feature_cols].copy()
    y = train_df["Segment"].copy()
    X_test = test_df.copy()
    X, X_test, y_encoded, le_target = preprocess(X, X_test, y)

    print("=== Load & Predict with TabNet ===")
    clf = TabNetClassifier()
    clf.load_model("tabnet_best.zip")

    print("=== Predict & Submit ===")
    predict_tabnet_submission(clf, X_test, test_df, le_target)


# 실행
if __name__ == "__main__":
    main()
