import pandas as pd

dataset_1_call_related = pd.read_parquet(
    "./data/raw/dataset_1_call_related.parquet"
)
dataset_2_expert_metadata = pd.read_parquet(
    "./data/raw/dataset_2_expert_metadata.parquet"
)
dataset_3_historical_outcomes = pd.read_parquet(
    "./data/raw/dataset_3_historical_outcomes.parquet"
)
dataset_4_expert_state_interval = pd.read_parquet(
    "./data/raw/dataset_4_expert_state_interval.parquet"
)
