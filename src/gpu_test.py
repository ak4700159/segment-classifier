import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification

# 간단한 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# DMatrix로 변환
dtrain = xgb.DMatrix(X, label=y)

# GPU 사용을 명시
params = {
    "tree_method": "hist",     # XGBoost 2.0+에서 권장
    "device": "cuda",          # 👈 GPU 사용
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

try:
    booster = xgb.train(params, dtrain, num_boost_round=10)
    print("✅ GPU 사용 가능 (device='cuda')")
except xgb.core.XGBoostError as e:
    print("❌ GPU 사용 불가:", e)
