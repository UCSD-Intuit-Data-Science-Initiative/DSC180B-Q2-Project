"""
Verify resolution_rate values from agent analytics.
Run: PYTHONPATH=src python scripts/verify_resolution_rate.py
"""

from pathlib import Path

from main_module.workforce.agent_analytics import AgentAnalytics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for data_dir in ["data/parquet", "data/raw"]:
    path = PROJECT_ROOT / data_dir / "dataset_2_expert_metadata.parquet"
    if path.exists():
        print(f"Using data from: {path.parent}")
        aa = AgentAnalytics(data_dir=str(path.parent))
        aa.load(tax_year=None)
        df = aa.agents_df
        rr = df["resolution_rate"]
        print(f"  Agents: {len(df)}")
        print(
            f"  resolution_rate: min={rr.min():.1f}, max={rr.max():.1f}, "
            f"mean={rr.mean():.1f}"
        )
        print(f"  Unique values (sample): {sorted(rr.unique())[:15]}")
        print(f"  Value counts (top 5):\n{rr.round(1).value_counts().head()}")
        break
else:
    print("No parquet data found in data/parquet or data/raw")
