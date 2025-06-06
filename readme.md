# 신용카드 고객 세그먼트 분류 AI 경진대회


## 개발 환경
Python 3.9.0


## 데이터셋 구조 
    .
    ├── train/
    │   ├── 1.회원정보/
    │   │   ├── 201807_train_회원정보.parquet
    │   │   ├── 201808_train_회원정보.parquet
    │   │   └── ...
    │   ├── 2.신용정보/
    │   ├── ...
    │   └── 8.성과정보/
    ├── test/
    │   ├── 1.회원정보/
    │   └── ...
    ├── sample_submission.csv
    ├── 데이터 명세.xlsx


    [회원정보]
    ┌──────┬──────────┬─────┐
    │  ID  │ 기준년월   │ 성별 ... │
    └──────┴──────────┴─────┘

    [신용정보]
    ┌──────┬──────────┬─────┐
    │  ID  │ 기준년월   │ 신용등급 ... │
    └──────┴──────────┴─────┘

    👇 Merge on ['ID', '기준년월']

    [통합 train_df]
    ┌──────┬──────────┬────┬───────┬─────┬───────┬────┬────┐
    │  ID  │ 기준년월   │ 성별 │ 신용등급 │ 매출  │ 청구금액 │ ... │ Segment │
    └──────┴──────────┴────┴───────┴─────┴───────┴────┴────┘

    [CONCAT] customer_train_df: shape (2400000, 78)
    [CONCAT] credit_train_df: shape (2400000, 42)
    [CONCAT] sales_train_df: shape (2400000, 406)
    [CONCAT] billing_train_df: shape (2400000, 46)
    [CONCAT] balance_train_df: shape (2400000, 82)
    [CONCAT] channel_train_df: shape (2400000, 105)
    [CONCAT] marketing_train_df: shape (2400000, 64)
    [CONCAT] performance_train_df: shape (2400000, 49)