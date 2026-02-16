import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TURBOTAX_PEAK_DATES = {
    (1, 15): 2.0,
    (1, 31): 2.5,
    (2, 15): 2.0,
    (3, 15): 2.5,
    (4, 1): 3.0,
    (4, 10): 4.0,
    (4, 11): 4.5,
    (4, 12): 5.0,
    (4, 13): 5.5,
    (4, 14): 6.0,
    (4, 15): 7.0,
    (4, 16): 3.0,
    (4, 17): 2.0,
    (4, 18): 2.5,
    (10, 15): 2.5,
}

QUICKBOOKS_PEAK_DATES = {
    (1, 15): 3.0,
    (1, 31): 4.0,
    (3, 15): 3.5,
    (4, 15): 2.5,
    (4, 18): 2.0,
    (6, 15): 3.0,
    (9, 15): 3.0,
    (10, 15): 2.5,
    (12, 31): 3.5,
    (1, 1): 2.5,
}


def get_turbotax_multiplier(month, day):
    key = (month, day)
    if key in TURBOTAX_PEAK_DATES:
        return TURBOTAX_PEAK_DATES[key]

    if month == 1:
        return 2.5
    elif month == 2:
        return 2.8
    elif month == 3:
        return 3.2
    elif month == 4 and day <= 15:
        return 4.0
    elif month == 4 and day > 15:
        return 1.2
    elif month == 10 and 1 <= day <= 15:
        return 1.8
    elif month in [5, 6, 7, 8, 9, 11, 12]:
        return 0.3
    return 0.5


def get_quickbooks_multiplier(month, day):
    key = (month, day)
    if key in QUICKBOOKS_PEAK_DATES:
        return QUICKBOOKS_PEAK_DATES[key]

    if month == 1 and day <= 15:
        return 2.5
    elif month == 1 and day > 15 and day <= 31:
        return 3.0
    elif month == 2:
        return 1.5
    elif month == 3 and day <= 15:
        return 2.5
    elif month == 3 and day > 15:
        return 1.8
    elif month == 4 and day <= 18:
        return 2.0
    elif month == 4 and day > 18:
        return 1.0
    elif month == 5:
        return 0.8
    elif month == 6 and day <= 15:
        return 2.2
    elif month == 6 and day > 15:
        return 1.0
    elif month == 7:
        return 0.7
    elif month == 8:
        return 0.8
    elif month == 9 and day <= 15:
        return 2.2
    elif month == 9 and day > 15:
        return 1.0
    elif month == 10 and day <= 15:
        return 1.5
    elif month == 10 and day > 15:
        return 1.0
    elif month == 11:
        return 1.0
    elif month == 12:
        if day >= 20:
            return 2.5
        return 1.2
    return 1.0


def generate_intuit_scale_data(start_date_str="2023-01-01", days=1095, target_calls=300000):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=days)

    num_experts = 150
    expert_ids = [str(uuid.uuid4()) for _ in range(num_experts)]

    tt_experts = expert_ids[:90]
    qb_experts = expert_ids[60:]

    expert_data = {
        "Expert ID": expert_ids,
        "Skill certifications": [
            "TurboTax_L1"
            if i < 60
            else "QuickBooks_Pro"
            if i < 120
            else "Bilingual_Both"
            for i in range(num_experts)
        ],
        "Avg handle time by expert": np.random.normal(
            600, 100, num_experts
        ).astype(int),
        "Resolution Rate By expert": np.round(
            np.random.uniform(0.70, 0.99, num_experts), 2
        ),
        "Transfer Rate by expert": np.round(
            np.random.uniform(0.05, 0.20, num_experts), 2
        ),
        "Satisfaction score given from customers": np.round(
            np.random.uniform(4.0, 5.0, num_experts), 2
        ),
    }
    df_experts = pd.DataFrame(expert_data)

    num_weekdays = sum(
        1 for d in range(days)
        if (start_date + timedelta(days=d)).weekday() < 5
    )
    hours_per_day = 12
    seconds_per_day = hours_per_day * 3600
    total_seconds = num_weekdays * seconds_per_day
    avg_interval = total_seconds / target_calls

    scale_factor = avg_interval / 215

    tt_timestamps = []
    qb_timestamps = []
    current = start_date

    while current < end_date:
        month = current.month
        day = current.day
        hour = current.hour
        weekday = current.weekday()

        if weekday >= 5:
            days_until_monday = 7 - weekday
            current = current.replace(hour=5, minute=0, second=0) + timedelta(
                days=days_until_monday
            )
            continue
        elif hour < 5:
            current = current.replace(hour=5, minute=0, second=0)
            continue
        elif hour >= 17:
            current = (current + timedelta(days=1)).replace(
                hour=5, minute=0, second=0
            )
            continue

        tt_multiplier = get_turbotax_multiplier(month, day)
        qb_multiplier = get_quickbooks_multiplier(month, day)

        tt_base_step = 400 * scale_factor
        qb_base_step = 900 * scale_factor

        tt_step = np.random.exponential(scale=tt_base_step / tt_multiplier)
        qb_step = np.random.exponential(scale=qb_base_step / qb_multiplier)

        min_step = min(tt_step, qb_step)
        min_step = max(1, min_step)

        next_time = current + timedelta(seconds=min_step)

        if next_time.hour >= 17:
            current = (current + timedelta(days=1)).replace(
                hour=5, minute=0, second=0
            )
            continue

        if next_time < end_date:
            if tt_step <= qb_step:
                tt_timestamps.append(next_time)
            else:
                qb_timestamps.append(next_time)

        current = next_time

    tt_count = len(tt_timestamps)
    qb_count = len(qb_timestamps)
    total_calls = tt_count + qb_count

    print(f"Date range: {start_date_str} to {end_date.strftime('%Y-%m-%d')}")
    print(
        f"Generated {tt_count:,} TurboTax calls and {qb_count:,} QuickBooks calls"
    )
    print(f"Total: {total_calls:,} calls (target: {target_calls:,})")

    tt_data = pd.DataFrame(
        {
            "Session ID / Contact ID": [
                str(uuid.uuid4()) for _ in range(tt_count)
            ],
            "Customer ID": [str(uuid.uuid4()) for _ in range(tt_count)],
            "Communication channel": ["Inbound Calls"] * tt_count,
            "Product group": ["TurboTax"] * tt_count,
            "Arrival time": tt_timestamps,
        }
    )

    qb_data = pd.DataFrame(
        {
            "Session ID / Contact ID": [
                str(uuid.uuid4()) for _ in range(qb_count)
            ],
            "Customer ID": [str(uuid.uuid4()) for _ in range(qb_count)],
            "Communication channel": ["Inbound Calls"] * qb_count,
            "Product group": ["QuickBooks"] * qb_count,
            "Arrival time": qb_timestamps,
        }
    )

    df = pd.concat([tt_data, qb_data], ignore_index=True)
    df = df.sort_values("Arrival time").reset_index(drop=True)
    actual_num_calls = len(df)

    assigned_experts = df_experts.sample(
        n=actual_num_calls, replace=True
    ).reset_index(drop=True)
    df = pd.concat([df, assigned_experts], axis=1)
    df["Expert assigned"] = df["Expert ID"]

    df["Date"] = df["Arrival time"].dt.strftime("%m/%d/%Y")

    df["Month"] = df["Arrival time"].dt.month
    df["Day"] = df["Arrival time"].dt.day

    def is_peak_season(row):
        if row["Product group"] == "TurboTax":
            return row["Month"] in [1, 2, 3, 4] or (
                row["Month"] == 10 and row["Day"] <= 15
            )
        else:
            return (
                (row["Month"] == 1)
                or (row["Month"] == 3 and row["Day"] <= 15)
                or (row["Month"] == 6 and row["Day"] <= 15)
                or (row["Month"] == 9 and row["Day"] <= 15)
                or (row["Month"] == 12 and row["Day"] >= 20)
            )

    df["Is_Peak_Season"] = df.apply(is_peak_season, axis=1)

    df["Hold time during call"] = np.where(
        df["Is_Peak_Season"],
        np.random.exponential(180, actual_num_calls),
        np.random.exponential(45, actual_num_calls),
    )
    df["Hold time during call"] = df["Hold time during call"].astype(int)

    start_times = []
    for i in range(actual_num_calls):
        st = df.loc[i, "Arrival time"] + timedelta(
            seconds=int(df.loc[i, "Hold time during call"])
        )
        start_times.append(st)
    df["Start time"] = start_times

    df["DURATION OF CALL"] = np.random.normal(
        df["Avg handle time by expert"], 120
    ).astype(int)
    df["DURATION OF CALL"] = df["DURATION OF CALL"].clip(lower=60)

    end_times = []
    for i in range(actual_num_calls):
        et = df.loc[i, "Start time"] + timedelta(
            seconds=int(df.loc[i, "DURATION OF CALL"])
        )
        end_times.append(et)
    df["End time"] = end_times

    df["Answered?"] = np.where(df["Hold time during call"] > 600, "No", "Yes")

    df["Transfer Flag"] = np.random.choice(
        ["Yes", "No"], actual_num_calls, p=[0.15, 0.85]
    )
    df.loc[df["Answered?"] == "No", "Transfer Flag"] = "No"

    df["Number of transfers in session"] = np.where(
        df["Transfer Flag"] == "Yes",
        np.random.randint(1, 3, actual_num_calls),
        0,
    )
    df["Transfer destination"] = np.where(
        df["Transfer Flag"] == "Yes",
        np.random.choice(
            ["Tier_2", "Billing", "Tech_Support"], actual_num_calls
        ),
        None,
    )

    df["Resolution outcome"] = np.where(
        df["Answered?"] == "No",
        "Abandoned",
        np.where(df["Transfer Flag"] == "Yes", "Transferred", "Resolved"),
    )

    df["First Call Resolution"] = np.where(
        df["Resolution outcome"] == "Resolved", "Yes", "No"
    )

    df["Customer History / Contact history"] = np.random.randint(
        0, 10, actual_num_calls
    )

    self_service_opts = ["Web FAQ", "Chatbot", "Knowledge Base", "None"]
    df["Self service attempts before calling"] = np.random.choice(
        self_service_opts, actual_num_calls, p=[0.2, 0.3, 0.1, 0.4]
    )

    df["Post resolution behavior"] = np.where(
        df["Resolution outcome"] == "Resolved",
        "No Action",
        "Callback within 24h",
    )

    df["OCCUPANCY"] = np.where(
        df["Is_Peak_Season"],
        np.round(np.random.uniform(0.85, 0.99, actual_num_calls), 2),
        np.round(np.random.uniform(0.60, 0.80, actual_num_calls), 2),
    )

    df["Arrival time"] = df["Arrival time"].dt.strftime("%m/%d/%Y %H:%M:%S")
    df["Start time"] = pd.to_datetime(df["Start time"]).dt.strftime(
        "%m/%d/%Y %H:%M:%S"
    )
    df["End time"] = pd.to_datetime(df["End time"]).dt.strftime(
        "%m/%d/%Y %H:%M:%S"
    )

    final_columns = [
        "Date",
        "Arrival time",
        "Start time",
        "End time",
        "Customer ID",
        "Expert ID",
        "Answered?",
        "Communication channel",
        "Product group",
        "Skill certifications",
        "Avg handle time by expert",
        "Resolution Rate By expert",
        "Transfer Rate by expert",
        "Satisfaction score given from customers",
        "Customer History / Contact history",
        "Self service attempts before calling",
        "Session ID / Contact ID",
        "Expert assigned",
        "Resolution outcome",
        "Transfer destination",
        "Number of transfers in session",
        "Post resolution behavior",
        "Transfer Flag",
        "First Call Resolution",
        "Hold time during call",
        "DURATION OF CALL",
        "OCCUPANCY",
    ]

    return df[final_columns]


if __name__ == "__main__":
    df_mock = generate_intuit_scale_data(
        start_date_str="2023-01-01",
        days=730,
        target_calls=500000
    )

    filename = "mock_intuit_2year_data.csv"
    df_mock.to_csv(filename, index=False)

    print(f"\nData saved to {filename}")
    print(f"Total rows: {len(df_mock):,}")
    print("\nProduct distribution:")
    print(df_mock["Product group"].value_counts())

    df_mock["Year"] = pd.to_datetime(
        df_mock["Date"], format="%m/%d/%Y"
    ).dt.year
    df_mock["Month"] = pd.to_datetime(
        df_mock["Date"], format="%m/%d/%Y"
    ).dt.month
    print("\nCalls by year:")
    print(df_mock.groupby("Year").size())
    print("\nTurboTax calls by month (all years combined):")
    print(
        df_mock[df_mock["Product group"] == "TurboTax"].groupby("Month").size()
    )
    print("\nQuickBooks calls by month (all years combined):")
    print(
        df_mock[df_mock["Product group"] == "QuickBooks"]
        .groupby("Month")
        .size()
    )
