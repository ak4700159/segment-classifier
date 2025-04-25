from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd
import gc


def train_tabnet_model(X, y, model_path="tabnet_best"):
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = TabNetClassifier(
        n_d=64, n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=10,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )

    clf.fit(
        X_train=X_train.values,
        y_train=y_train,
        eval_set=[(X_val.values, y_val)],
        eval_name=["valid"],
        eval_metric=["accuracy"],
        max_epochs=30,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    val_preds = clf.predict(X_val.values)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\n✅ TabNet Validation Accuracy: {val_acc * 100:.2f}%")

    clf.save_model(model_path)
    return clf


def predict_tabnet_submission(clf, X_test, test_df, le_target, output_path="tabnet_submit.csv"):
    try:
        preds = clf.predict(X_test.drop(columns=['ID'], errors='ignore').values)
        pred_labels = le_target.inverse_transform(preds)
    except Exception as e:
        print("❌ 예측 오류 발생:", e)
        print("preds:", np.unique(preds))
        print("le_target.classes_:", le_target.classes_)
        return

    try:
        test_df = test_df.copy()
        test_df["pred_label"] = pred_labels
        submission = test_df.groupby("ID")["pred_label"].agg(lambda x: x.value_counts().idxmax()).reset_index()
        submission.columns = ["ID", "Segment"]
        submission.to_csv(output_path, index=False)
        print(f"[SAVE] Submission saved to {output_path}")
    except Exception as e:
        print("❌ 제출 생성 중 오류:", e)
        print("test_df preview:", test_df.head())
