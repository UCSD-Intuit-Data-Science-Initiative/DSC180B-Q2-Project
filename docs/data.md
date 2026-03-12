---
layout: default
title: Data
nav_order: 1
---

# Data

_Our datasets for this project contain comprehensive daily call center operational metrics with complete data coverage across two years of call center operations, spanning from **November 27, 2023, to November 15, 2025**._


## Data Inventory

| Dataset | Source | Location | Notes |
| --- | --- | --- | --- |
| Call-Related Data | Routing / Telephony Logs | `data/raw/call_related_data.parquet` | Granular interaction leg details. Contains `customer_id` and `expert_id` (needs masking). |
| Expert Metadata | WFM / HR Systems | `data/raw/expert_metadata.parquet` | Expert lifecycle, status, and aggregated performance metrics. |
| Historical Outcomes | CRM / Ticketing System | `data/raw/historical_outcomes.parquet` | Session-level resolution, transfer counts, and outcomes. |
| Expert State & Interval | Presence / Activity Logs | `data/raw/expert_state_interval.parquet` | 30-minute interval state, availability, and occupancy tracking. |

_Keep this table current as new datasets are added or existing ones change._

---

## Data Dictionaries

### 1. Call-Related Data
_Contains granular metrics for individual interaction legs._

| Data Field | Data Type | Description |
| :--- | :--- | :--- |
| `tax_year` | INT | Reporting year bucket. |
| `cc_id` | STRING/ID | Canonical contact identifier. |
| `engagement_id` | STRING | Engagement/workflow identifier. |
| `interaction_date` | DATE | Date anchor for the interaction. |
| `arrival_time_utc` | TIMESTAMP | Earliest observed arrival. |
| `start_time_utc` | TIMESTAMP | Earliest agent-leg start. |
| `end_time_utc` | TIMESTAMP | Latest agent-leg end. |
| `customer_id` | STRING | Customer identifier. |
| `customer_id_source` | STRING | Source of `customer_id`. |
| `expert_id` | STRING | Handling expert identifier. |
| `answered_flag` | STRING | Yes/No answered indicator. |
| `product_group_sku` | STRING | Product/routing grouping. |
| `communication_channel_type` | STRING | Channel/initiation descriptor. |
| `communication_leg_direction` | STRING | Interaction leg direction. |

### 2. Expert Metadata
_Captures the lifecycle, status, and aggregated metrics of call center experts._

| Data Field | Data Type | Description |
| :--- | :--- | :--- |
| `tax_year` | INT | Reporting year bucket. |
| `expert_id` | STRING/ID | Canonical expert identifier. |
| `first_started_date` | DATE | Lifecycle start proxy date. |
| `first_active_date` | DATE | Active/production proxy date. |
| `latest_termination_date` | DATE | Lifecycle end date. |
| `tenure_as_of_date` | DATE | Date used as tenure boundary. |
| `tenure_is_ongoing_flg` | INT | Ongoing-tenure indicator. |
| `tenure_days_since_start` | INT | Days from `first_started_date`. |
| `tenure_days_since_first_active` | INT | Days from `first_active_date`. |
| `agent_status` | STRING | Current lifecycle status label. |
| `active_flg` | STRING | Current active indicator. |
| `business_segment` | STRING | Lifecycle business segment. |
| `expert_segment` | STRING | Lifecycle expert segment. |
| `access_rule` | STRING | Lifecycle access rule/profile. |
| `routing_profiles_seen_raw` | STRING | Observed routing profiles (raw). |
| `routing_profiles_seen_clean` | STRING | Observed routing profiles (clean). |
| `skill_certifications` | STRING | Combined skill descriptor. |
| `contacts` | BIGINT | Count of contact-level rows. |
| `answered_contacts` | BIGINT | Count of answered contacts. |
| `average_handle_time_seconds` | DOUBLE | Average handle seconds. |
| `average_hold_time_seconds` | DOUBLE | Average hold seconds. |
| `resolution_rate` | DOUBLE | Resolution proxy percentage. |
| `transfer_rate` | DOUBLE | Transfer percentage. |

### 3. Historical Outcomes
_Records the results, transfers, and resolution behaviors of customer sessions._

| Data Field | Data Type | Description |
| :--- | :--- | :--- |
| `tax_year` | INT | Reporting year bucket. |
| `calendar_year` | INT | Calendar year. |
| `session_contact_id` | STRING | Canonical session key. |
| `expert_assigned_id` | STRING | Final assigned expert ID. |
| `resolution_outcome` | STRING | Outcome classification. |
| `transfer_destination` | STRING | Destination queue/profile. |
| `transfer_count` | BIGINT | Count of transfer events. |
| `post_resolution_behavior` | STRING | Repeat-contact indicator. |
| `transfer_flag` | STRING | Session-level transfer indicator. |
| `first_call_resolution` | STRING | First-call resolution proxy. |
| `hold_time_seconds` | DOUBLE | Total hold time seconds. |
| `duration_of_call_minutes` | DOUBLE | Session duration in minutes. |
| `cc_id` | STRING | Contact ID context. |
| `engagement_id` | STRING | Engagement context. |
| `conversation_id` | STRING | Conversation context. |
| `case_number` | STRING | Case identifier field. |
| `customer_id` | STRING | Customer identifier. |
| `expert_assigned_id_source` | STRING | Source of assigned expert. |
| `expert_from_assignment_id` | STRING | Assignment-summary expert ID. |
| `expert_from_interaction_id` | STRING | Interaction-derived expert ID. |
| `expert_id_source_mismatch_flg` | INT | Source mismatch flag. |
| `expert_assigned_id_in_lifecycle_flg` | INT | Lifecycle membership flag. |
| `expert_assigned_id_domain_status` | STRING | Domain status classification. |
| `session_start_time_utc` | TIMESTAMP | Session start timestamp. |
| `session_end_time_utc` | TIMESTAMP | Session end timestamp. |
| `interaction_date` | DATE | Session interaction date. |

### 4. Expert State & Interval
_Tracks adherence and activity states across standardized 30-minute intervals._

| Data Field | Data Type | Description |
| :--- | :--- | :--- |
| `tax_year` | INT | Reporting year bucket. |
| `date` | DATE | UTC interval date. |
| `time_interval_30m_utc` | STRING | 30-minute UTC label. |
| `expert_id` | STRING | Canonical expert ID. |
| `total_handle_time_seconds` | DOUBLE | Raw handle seconds. |
| `total_available_time_seconds` | DOUBLE | Raw available seconds. |
| `activity_break_meal_seconds` | DOUBLE | Raw break/meal seconds. |
| `activity_meeting_training_seconds` | DOUBLE | Raw meeting/training seconds. |
| `activity_offline_unavailable_seconds` | DOUBLE | Raw offline seconds. |
| `activity_uncategorized_seconds` | DOUBLE | Raw uncategorized seconds. |
| `primary_activity_category_30m` | STRING | Dominant activity category. |
| `occupancy_pct` | DOUBLE | Raw occupancy percentage. |
| `normalization_scale_factor` | DOUBLE | Scaling factor. |
| `total_handle_time_seconds_normalized` | DOUBLE | Normalized handle seconds. |
| `total_available_time_seconds_normalized` | DOUBLE | Normalized available seconds. |
| `activity_break_meal_seconds_normalized` | DOUBLE | Normalized break/meal seconds. |
| `activity_meeting_training_seconds_normalized` | DOUBLE | Normalized meeting seconds. |
| `activity_offline_unavailable_seconds_normalized` | DOUBLE | Normalized offline seconds. |
| `activity_uncategorized_seconds_normalized` | DOUBLE | Normalized uncategorized. |
| `occupancy_pct_normalized` | DOUBLE | Normalized occupancy. |
| `interval_start_utc` | TIMESTAMP | Interval start timestamp. |
| `interval_end_utc` | TIMESTAMP | Interval end timestamp. |
| `total_state_overlap_seconds` | DOUBLE | Raw total overlap seconds. |
| `total_state_overlap_seconds_normalized` | DOUBLE | Capped overlap seconds. |
| `interval_over_30m_flg` | INT | Raw overage flag. |
| `interval_over_30m_normalized_flg` | INT | Normalized overage flag. |
