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


# ============================
# 2. 데이터 로드 함수   
# ============================
def load_data():
    for split in DATA_SPLITS:
        for category, info in DATA_CATEGORIES.items():
            for month in MONTHS:
                file_path = f"../{split}/{info['folder']}/2018{month}_{split}_{info['suffix']}.parquet"
                variable_name = f"{info['var_prefix']}_{split}_{month}"
                globals()[variable_name] = pd.read_parquet(file_path)
                print(f"[LOAD] {variable_name} from {file_path}")
    gc.collect()


# ============================
# 3. 월별 데이터 통합 함수
# ============================
def concat_monthly_data(split):
    result = {}
    for prefix in INFO_CATEGORIES:
        df_list = [globals()[f"{prefix}_{split}_{month}"] for month in MONTHS]
        result[f"{prefix}_{split}_df"] = pd.concat(df_list, axis=0)
        print(f"[CONCAT] {prefix}_{split}_df: shape {result[f'{prefix}_{split}_df'].shape}")
        gc.collect()
    return result

# 현재 컴퓨팅 환경에서 전체 데이터를 메모리에 로드하기엔 메모리 부족 에러 발생 
def reduce_data_fraction(dfs_dict, split="train", frac=0.1, seed=42):
    """
    각 데이터프레임에서 ID 기준으로 frac 만큼만 샘플링
    """
    # 우선 하나의 기준 df (보통 customer_df)에서 unique ID 추출
    base_df = dfs_dict[f"customer_{split}_df"]
    unique_ids = base_df["ID"].dropna().unique()
    
    np.random.seed(seed)
    sampled_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * frac), replace=False)
    sampled_ids_set = set(sampled_ids)

    print(f"[INFO] 샘플링된 ID 수: {len(sampled_ids)} / 전체 ID 수: {len(unique_ids)}")

    for key in list(dfs_dict.keys()):
        before = dfs_dict[key].shape[0]
        dfs_dict[key] = dfs_dict[key][dfs_dict[key]["ID"].isin(sampled_ids_set)].copy()
        after = dfs_dict[key].shape[0]
        print(f"[REDUCE] {key}: {before} → {after}")
        gc.collect()

    return dfs_dict


# ============================
# 4. 데이터 병합 함수
# ============================
def merge_all_data(dfs_dict, split="train"):
    df = dfs_dict[f"customer_{split}_df"].merge(dfs_dict[f"credit_{split}_df"], on=["기준년월", "ID"], how="left")
    del dfs_dict[f"customer_{split}_df"], dfs_dict[f"credit_{split}_df"]
    gc.collect()

    merge_order = ["channel", "sales", "billing", "balance", "marketing", "performance"]
    
    for key in merge_order:
        df_to_merge = dfs_dict[f"{key}_{split}_df"]
        df = df.merge(df_to_merge, on=["기준년월", "ID"], how="left")
        print(f"[MERGE] {key}_{split}_df, shape: {df.shape}")
        del dfs_dict[f"{key}_{split}_df"]
        gc.collect()

    return df


# ============================
# 5. 전처리 및 인코딩 함수
# ============================
def preprocess(X, X_test, y=None):
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}

    # Label encoding for categorical features
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

        unseen_labels = set(X_test[col]) - set(le.classes_)
        if unseen_labels:
            le.classes_ = np.append(le.classes_, list(unseen_labels))
        X_test[col] = le.transform(X_test[col].astype(str))

    # Fill NA / Inf
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)

    # ⚠️ ID 제거
    X_id = X_test["ID"] if "ID" in X_test.columns else None
    if "ID" in X.columns:
        X = X.drop(columns=["ID"])
    if "ID" in X_test.columns:
        X_test = X_test.drop(columns=["ID"])

    # StandardScaler 적용
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # ID 컬럼 복구 (나중에 제출용으로 필요)
    if X_id is not None:
        X_test_scaled["ID"] = X_id.values

    if y is not None:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        return X_scaled, X_test_scaled, y, le_target
    return X_scaled, X_test_scaled, None, None


# ============================
# 6. 모델 학습 함수
# ============================

# 파이토치 이용 
def train_model_pytorch(X, y, epochs=20, batch_size=512, lr=1e-3, device='cuda', save_path='best_model.pth'):
    # 텐서 변환
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 데이터로더 생성
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 구성
    model = MLPClassifier(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        all_preds = []
        all_labels = []

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)  # 전체 loss 누적
            all_preds.append(torch.argmax(outputs, dim=1).detach().cpu())
            all_labels.append(yb.detach().cpu())

        # 평균 loss 계산
        avg_loss = total_loss / len(dataset)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc = accuracy_score(labels.numpy(), preds.numpy())

        print(f"[Epoch {epoch:2d}] Loss: {avg_loss:.4f} | Accuracy: {acc * 100:.2f}%")

        # 모델 저장 (가장 낮은 Loss일 때)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ [Epoch {epoch}] New best model saved! (Loss: {best_loss:.4f})")

    return model


# ============================
# 7. 예측 및 제출 생성 함수
# ============================
# 파이토치 이용
def generate_submission_pytorch(model, X_test, test_df, le_target, device='cuda'):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test.drop(columns=['ID'], errors='ignore').values, dtype=torch.float32).to(device)
        preds = model(X_tensor)
        pred_labels = preds.argmax(dim=1).cpu().numpy()
        pred_segments = le_target.inverse_transform(pred_labels)

    test_df = test_df.copy()
    test_df["pred_label"] = pred_segments
    submission = test_df.groupby("ID")["pred_label"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    submission.columns = ["ID", "Segment"]
    submission.to_csv("pytorch_submit.csv", index=False)
    print("[SAVE] Submission saved to pytorch_submit.csv")


# ============================
# 8. 전체 실행 함수
# ============================
def main():
    print("=== Load ===")
    load_data()

    print("=== Concat ===")
    train_dfs = concat_monthly_data("train")
    train_dfs = reduce_data_fraction(train_dfs, split="train", frac=0.1)

    test_dfs = concat_monthly_data("test")
    test_dfs = reduce_data_fraction(test_dfs, split="test", frac=0.1)

    print("=== Merge ===")
    train_df = merge_all_data(train_dfs, "train")
    test_df = merge_all_data(test_dfs, "test")

    print("=== Preprocess ===")
    feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment"]]
    X = train_df[feature_cols].copy()
    y = train_df["Segment"].copy()
    X_test = test_df.copy()
    
    X, X_test, y_encoded, le_target = preprocess(X, X_test, y)
    print("[Check] NaN in X:", np.isnan(X.values).sum())
    print("[Check] Inf in X:", np.isinf(X.values).sum())
    print("[Check] y min:", y_encoded.min(), "max:", y_encoded.max())
    print("[Check] y unique:", np.unique(y_encoded))

    print("=== Train ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model_pytorch(X, y_encoded, device=device)

    print("=== Predict & Submit ===")
    generate_submission_pytorch(model, X_test, test_df, le_target, device=device)


# 실행
if __name__ == "__main__":
    main()
