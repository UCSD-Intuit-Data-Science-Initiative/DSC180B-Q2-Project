import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random

def generate_intuit_scale_data(start_date_str='2024-01-01', days=730):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = start_date + timedelta(days=days)
    
    # --- 1. Generate Expert Pool ---
    num_experts = 150 # Increased staff for higher volume
    expert_ids = [str(uuid.uuid4()) for _ in range(num_experts)]
    
    expert_data = {
        'Expert ID': expert_ids,
        'Skill certifications': [random.choice(['TurboTax_L1', 'QuickBooks_Pro', 'Bilingual_Span', 'Tax_Expert_Sr']) for _ in range(num_experts)],
        'Avg handle time by expert': np.random.normal(600, 100, num_experts).astype(int), 
        'Resolution Rate By expert': np.round(np.random.uniform(0.70, 0.99, num_experts), 2),
        'Transfer Rate by expert': np.round(np.random.uniform(0.05, 0.20, num_experts), 2),
        'Satisfaction score given from customers': np.round(np.random.uniform(4.0, 5.0, num_experts), 2)
    }
    df_experts = pd.DataFrame(expert_data)

    # --- 2. Generate Time Intervals (With Tax Season Logic) ---
    timestamps = []
    current = start_date
    
    while current < end_date:
        month = current.month
        day = current.day
        hour = current.hour
        weekday = current.weekday() # 0=Mon, 6=Sun

        # -- SEASONALITY LOGIC --
        # Tax Season (Jan - April 15): High Volume
        if 1 <= month <= 3:
            season_multiplier = 3.0
        elif month == 4 and day <= 15:
            season_multiplier = 5.0 # Peak Tax Day Rush
        elif month == 4 and day > 15:
            season_multiplier = 1.5 # Post-deadline drop
        elif month == 10 and day >= 1: 
            season_multiplier = 1.2 # Extension deadline mini-peak
        else:
            season_multiplier = 0.5 # Off-season (Summer/Fall)

        # -- TIME OF DAY LOGIC --
        if 8 <= hour <= 17:
            time_weight = 1.0
        elif 18 <= hour <= 23:
            time_weight = 0.4
        else:
            time_weight = 0.05
            
        # Weekend dampener
        if weekday >= 5:
            time_weight *= 0.6

        # Calculate arrival probability
        # We adjust the 'step' (time between calls) based on volume
        # Higher multiplier = smaller step = more calls
        base_step_seconds = 600 # Base: 1 call every 10 mins
        adjusted_step = base_step_seconds / (season_multiplier * time_weight)
        
        # Add some randomness to the step
        step = np.random.exponential(scale=adjusted_step)
        
        # Ensure step isn't zero
        step = max(1, step)
        
        current += timedelta(seconds=step)
        
        if current < end_date:
            timestamps.append(current)
        
    actual_num_calls = len(timestamps)
    
    # --- 3. Generate Main Transactional Data ---
    data = {
        'Session ID / Contact ID': [str(uuid.uuid4()) for _ in range(actual_num_calls)],
        'Customer ID': [str(uuid.uuid4()) for _ in range(actual_num_calls)],
        'Communication channel': ['Inbound Calls'] * actual_num_calls,
        'Product group': np.random.choice(['TurboTax', 'QuickBooks'], actual_num_calls, p=[0.7, 0.3]), # TT heavy
        'Arrival time': timestamps,
    }

    df = pd.DataFrame(data)

    # --- 4. Join Expert Data ---
    assigned_experts = df_experts.sample(n=actual_num_calls, replace=True).reset_index(drop=True)
    df = pd.concat([df, assigned_experts], axis=1)
    df['Expert assigned'] = df['Expert ID']

    # --- 5. Generate Calculated Fields ---
    df['Date'] = df['Arrival time'].dt.strftime('%m/%d/%Y')
    
    # Hold times increase during Tax Season (Capacity constraints)
    # We can infer "busy season" from the month of the Arrival Time
    df['Month'] = df['Arrival time'].dt.month
    df['Is_Tax_Season'] = df['Month'].isin([1, 2, 3, 4])
    
    # Base hold time + penalty for tax season
    df['Hold time during call'] = np.where(df['Is_Tax_Season'], 
                                           np.random.exponential(180, actual_num_calls), # Longer hold in season
                                           np.random.exponential(45, actual_num_calls))  # Short hold off season
    df['Hold time during call'] = df['Hold time during call'].astype(int)

    start_times = []
    for i in range(actual_num_calls):
        st = df.loc[i, 'Arrival time'] + timedelta(seconds=int(df.loc[i, 'Hold time during call']))
        start_times.append(st)
    df['Start time'] = start_times

    df['DURATION OF CALL'] = np.random.normal(df['Avg handle time by expert'], 120).astype(int)
    df['DURATION OF CALL'] = df['DURATION OF CALL'].clip(lower=60) 

    end_times = []
    for i in range(actual_num_calls):
        et = df.loc[i, 'Start time'] + timedelta(seconds=int(df.loc[i, 'DURATION OF CALL']))
        end_times.append(et)
    df['End time'] = end_times

    df['Answered?'] = np.where(df['Hold time during call'] > 600, 'No', 'Yes') # Higher tolerance for hold in tax season?
    
    df['Transfer Flag'] = np.random.choice(['Yes', 'No'], actual_num_calls, p=[0.15, 0.85])
    df.loc[df['Answered?'] == 'No', 'Transfer Flag'] = 'No'
    
    df['Number of transfers in session'] = np.where(df['Transfer Flag'] == 'Yes', np.random.randint(1, 3, actual_num_calls), 0)
    df['Transfer destination'] = np.where(df['Transfer Flag'] == 'Yes', np.random.choice(['Tier_2', 'Billing', 'Tech_Support'], actual_num_calls), np.nan)

    df['Resolution outcome'] = np.where(df['Answered?'] == 'No', 'Abandoned', 
                                        np.where(df['Transfer Flag'] == 'Yes', 'Transferred', 'Resolved'))

    df['First Call Resolution'] = np.where(df['Resolution outcome'] == 'Resolved', 'Yes', 'No')

    df['Customer History / Contact history'] = np.random.randint(0, 10, actual_num_calls)
    
    self_service_opts = ['Web FAQ', 'Chatbot', 'Knowledge Base', 'None']
    df['Self service attempts before calling'] = np.random.choice(self_service_opts, actual_num_calls, p=[0.2, 0.3, 0.1, 0.4])

    df['Post resolution behavior'] = np.where(df['Resolution outcome'] == 'Resolved', 'No Action', 'Callback within 24h')

    # Occupancy is higher in tax season
    df['OCCUPANCY'] = np.where(df['Is_Tax_Season'],
                               np.round(np.random.uniform(0.85, 0.99, actual_num_calls), 2),
                               np.round(np.random.uniform(0.60, 0.80, actual_num_calls), 2))

    df['Arrival time'] = df['Arrival time'].dt.strftime('%m/%d/%Y %H:%M:%S')
    df['Start time'] = pd.to_datetime(df['Start time']).dt.strftime('%m/%d/%Y %H:%M:%S')
    df['End time'] = pd.to_datetime(df['End time']).dt.strftime('%m/%d/%Y %H:%M:%S')

    final_columns = [
        "Date", "Arrival time", "Start time", "End time", "Customer ID", "Expert ID",
        "Answered?", "Communication channel", "Product group", "Skill certifications",
        "Avg handle time by expert", "Resolution Rate By expert", "Transfer Rate by expert",
        "Satisfaction score given from customers", "Customer History / Contact history",
        "Self service attempts before calling", "Session ID / Contact ID", "Expert assigned",
        "Resolution outcome", "Transfer destination", "Number of transfers in session",
        "Post resolution behavior", "Transfer Flag", "First Call Resolution",
        "Hold time during call", "DURATION OF CALL", "OCCUPANCY"
    ]
    
    return df[final_columns]

if __name__ == "__main__":
    # Generate 2 years (730 days) of data
    df_mock = generate_intuit_scale_data(days=730)
    
    filename = 'mock_intuit_2year_data.csv'
    df_mock.to_csv(filename, index=False)