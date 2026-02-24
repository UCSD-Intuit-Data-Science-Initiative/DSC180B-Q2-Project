import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, SCRIPT_DIR)  # keeps demand_forecasting_model importable

from main_module.workforce import CallCenterEmulator, EmulatorConfig, HybridForecaster
from demand_forecasting_model import CallDemandForecaster
from staffing_optimizer import (
    OptimizationThresholds,
    ShiftConstraints,
    StaffingOptimizer,
)

try:
    from business_analytics import BusinessAnalytics
    _ANALYTICS_AVAILABLE = True
except ImportError:
    _ANALYTICS_AVAILABLE = False

MODEL_PATH = os.path.join(SCRIPT_DIR, "demand_forecast_model.pkl")
HYBRID_MODEL_PATH = os.path.join(SCRIPT_DIR, "hybrid_forecast_model.pkl")
DATA_PATH = str(_ROOT / "data" / "interim" / "mock_intuit_2year_data.csv")


st.set_page_config(
    page_title="Workforce Optimization Dashboard",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_model_mtime(path):
    if os.path.exists(path):
        return os.path.getmtime(path)
    return None


@st.cache_resource
def load_single_forecaster(_model_mtime):
    forecaster = CallDemandForecaster()
    try:
        if os.path.exists(MODEL_PATH):
            forecaster.load_model(MODEL_PATH)
            return forecaster, True
        elif os.path.exists(DATA_PATH):
            st.info(
                "Training single model from data... This may take a minute."
            )
            forecaster.train(DATA_PATH)
            forecaster.save_model(MODEL_PATH)
            return forecaster, True
        else:
            return forecaster, False
    except Exception as e:
        st.error(f"Error loading single model: {e}")
        return forecaster, False


@st.cache_resource
def load_hybrid_forecaster(_model_mtime):
    forecaster = HybridForecaster()
    try:
        if os.path.exists(HYBRID_MODEL_PATH):
            forecaster.load_model(HYBRID_MODEL_PATH)
            return forecaster, True
        elif os.path.exists(DATA_PATH):
            st.info(
                "Training hybrid model from data... This may take a minute."
            )
            forecaster.train(DATA_PATH)
            forecaster.save_model(HYBRID_MODEL_PATH)
            return forecaster, True
        else:
            return forecaster, False
    except Exception as e:
        st.error(f"Error loading hybrid model: {e}")
        return forecaster, False


@st.cache_resource
def load_analytics():
    if not _ANALYTICS_AVAILABLE:
        return None, False
    try:
        if os.path.exists(DATA_PATH):
            return BusinessAnalytics(DATA_PATH), True
        return None, False
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        return None, False


def create_emulator_and_optimizer(
    config, shift_config, emulator_model="erlang_a"
):
    emulator_config = EmulatorConfig(
        avg_handle_time=config["avg_handle_time"],
        sla_threshold_seconds=config["sla_threshold"],
        avg_patience_time=config.get("avg_patience", 180),
        interval_duration_seconds=1800,
    )
    emulator = CallCenterEmulator(emulator_config, model=emulator_model)

    thresholds = OptimizationThresholds(
        max_avg_wait_time=config["max_wait"],
        min_sla_compliance=config["min_sla"],
        max_utilization=config["max_util"],
        max_abandonment_rate=config["max_abandon"],
    )

    shifts = ShiftConstraints(
        min_experts_per_interval=shift_config["min_experts"],
        max_experts_per_interval=shift_config["max_experts"],
        max_shift_changes_per_day=shift_config["max_changes"],
    )

    optimizer = StaffingOptimizer(emulator, thresholds, shifts)

    return emulator, optimizer


def forecast_demand_single(forecaster, target_date):
    target_date = pd.to_datetime(target_date).normalize()
    intervals = pd.date_range(start=target_date, periods=48, freq="30min")

    day_of_week = target_date.dayofweek

    predictions = []
    for interval in intervals:
        hour = interval.hour
        is_open = (day_of_week < 5) and (5 <= hour < 17)

        if is_open:
            pred = forecaster.predict_single_interval(interval)
        else:
            pred = 0

        predictions.append(
            {
                "interval_start": interval,
                "time": interval.strftime("%H:%M"),
                "hour": interval.hour,
                "predicted_calls": pred,
                "is_open": is_open,
            }
        )

    return pd.DataFrame(predictions)


def forecast_demand_hybrid(forecaster, target_date, reference_date=None):
    target_date = pd.to_datetime(target_date).normalize()
    intervals = pd.date_range(start=target_date, periods=48, freq="30min")

    day_of_week = target_date.dayofweek

    if reference_date is None:
        reference_date = forecaster.max_training_date

    days_ahead = (target_date - reference_date).days
    model_used = (
        "short-term"
        if days_ahead < forecaster.short_term_threshold_days
        else "long-term"
    )

    predictions = []
    for interval in intervals:
        hour = interval.hour
        is_open = (day_of_week < 5) and (5 <= hour < 17)

        if is_open:
            pred = forecaster.predict(interval, reference_date)
        else:
            pred = 0

        predictions.append(
            {
                "interval_start": interval,
                "time": interval.strftime("%H:%M"),
                "hour": interval.hour,
                "predicted_calls": pred,
                "is_open": is_open,
                "model_used": model_used,
            }
        )

    return pd.DataFrame(predictions)


def main():
    st.title("📞 Call Center Workforce Optimization Dashboard")
    st.markdown("---")

    st.sidebar.header("⚙️ Configuration")

    if "applied_config" not in st.session_state:
        st.session_state.applied_config = {
            "model_type": "Hybrid (Recommended)",
            "selected_date": datetime(2025, 4, 15),
            "avg_handle_time": 600,
            "sla_threshold": 60,
            "max_wait": 60,
            "min_sla": 80,
            "max_util": 85,
            "max_abandon": 5,
            "min_experts": 1,
            "max_experts": 50,
            "max_changes": 10,
            "emulator_model": "Erlang-A (Recommended)",
            "avg_patience": 180,
        }

    with st.sidebar.form("config_form"):
        st.subheader("🤖 Model Selection")
        model_type = st.radio(
            "Forecasting Model",
            ["Hybrid (Recommended)", "Single Model"],
            index=0 if st.session_state.applied_config["model_type"] == "Hybrid (Recommended)" else 1,
            help="Hybrid model uses short-term model for <7 days and long-term model for >=7 days",
        )

        st.subheader("📅 Date Selection")
        selected_date_input = st.date_input(
            "Forecast Date",
            value=st.session_state.applied_config["selected_date"],
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2027, 12, 31),
        )

        st.subheader("📊 Service Parameters")
        avg_handle_time_input = st.slider(
            "Avg Handle Time (seconds)",
            min_value=300,
            max_value=1200,
            value=st.session_state.applied_config["avg_handle_time"],
            step=30,
        )

        sla_threshold_input = st.slider(
            "SLA Threshold (seconds)",
            min_value=15,
            max_value=120,
            value=st.session_state.applied_config["sla_threshold"],
            step=5,
        )

        st.subheader("🎯 Optimization Thresholds")
        max_wait_input = st.slider("Max Avg Wait Time (s)", 15, 180, st.session_state.applied_config["max_wait"], 5)
        min_sla_input = st.slider("Min SLA Compliance (%)", 50, 99, st.session_state.applied_config["min_sla"], 5)
        max_util_input = st.slider("Max Utilization (%)", 50, 99, st.session_state.applied_config["max_util"], 5)
        max_abandon_input = st.slider("Max Abandonment Rate (%)", 1, 20, st.session_state.applied_config["max_abandon"], 1)

        st.subheader("👥 Shift Constraints")
        min_experts_input = st.number_input(
            "Min Experts per Interval", value=st.session_state.applied_config["min_experts"], min_value=0, max_value=10
        )
        max_experts_input = st.number_input(
            "Max Experts per Interval", value=st.session_state.applied_config["max_experts"], min_value=10, max_value=100
        )
        max_changes_input = st.number_input(
            "Max Shift Changes per Day", value=st.session_state.applied_config["max_changes"], min_value=1, max_value=48
        )

        st.subheader("⚙️ Emulator Settings")
        emulator_model_input = st.selectbox(
            "Queuing Model",
            ["Erlang-A (Recommended)", "Erlang-C"],
            index=0 if st.session_state.applied_config["emulator_model"] == "Erlang-A (Recommended)" else 1,
            help="Erlang-A models customer abandonment realistically. Erlang-C assumes infinite patience.",
        )
        avg_patience_input = st.slider(
            "Avg Customer Patience (s)",
            min_value=60,
            max_value=600,
            value=st.session_state.applied_config["avg_patience"],
            step=30,
            help="Average time a customer will wait before abandoning (Erlang-A only)",
        )

        submitted = st.form_submit_button("✅ Apply Changes", type="primary", use_container_width=True)

        if submitted:
            st.session_state.applied_config = {
                "model_type": model_type,
                "selected_date": selected_date_input,
                "avg_handle_time": avg_handle_time_input,
                "sla_threshold": sla_threshold_input,
                "max_wait": max_wait_input,
                "min_sla": min_sla_input,
                "max_util": max_util_input,
                "max_abandon": max_abandon_input,
                "min_experts": min_experts_input,
                "max_experts": max_experts_input,
                "max_changes": max_changes_input,
                "emulator_model": emulator_model_input,
                "avg_patience": avg_patience_input,
            }
            st.rerun()

    use_hybrid = st.session_state.applied_config["model_type"] == "Hybrid (Recommended)"
    selected_date = st.session_state.applied_config["selected_date"]
    avg_handle_time = st.session_state.applied_config["avg_handle_time"]
    sla_threshold = st.session_state.applied_config["sla_threshold"]
    max_wait = st.session_state.applied_config["max_wait"]
    min_sla = st.session_state.applied_config["min_sla"]
    max_util = st.session_state.applied_config["max_util"]
    max_abandon = st.session_state.applied_config["max_abandon"]
    min_experts = st.session_state.applied_config["min_experts"]
    max_experts = st.session_state.applied_config["max_experts"]
    max_changes = st.session_state.applied_config["max_changes"]
    emulator_model_key = "erlang_a" if "Erlang-A" in st.session_state.applied_config["emulator_model"] else "erlang_c"
    avg_patience = st.session_state.applied_config["avg_patience"]

    if use_hybrid:
        forecaster, model_loaded = load_hybrid_forecaster(
            get_model_mtime(HYBRID_MODEL_PATH)
        )
        model_name = "Hybrid"
    else:
        forecaster, model_loaded = load_single_forecaster(
            get_model_mtime(MODEL_PATH)
        )
        model_name = "Single"

    analytics, analytics_loaded = load_analytics()

    if not model_loaded:
        st.error(
            f"{model_name} model not loaded. Please ensure data file exists."
        )
        return

    if use_hybrid and forecaster.max_training_date:
        days_ahead = (
            pd.Timestamp(selected_date) - forecaster.max_training_date
        ).days
        if days_ahead < 0:
            horizon_info = "Historical date"
        elif days_ahead < 7:
            horizon_info = f"Short-term ({days_ahead} days ahead)"
        else:
            horizon_info = f"Long-term ({days_ahead} days ahead)"
        st.sidebar.caption(f"🎯 Forecast horizon: {horizon_info}")

    config = {
        "avg_handle_time": avg_handle_time,
        "sla_threshold": sla_threshold,
        "max_wait": max_wait,
        "min_sla": min_sla,
        "max_util": max_util,
        "max_abandon": max_abandon,
        "avg_patience": avg_patience,
    }

    shift_config = {
        "min_experts": min_experts,
        "max_experts": max_experts,
        "max_changes": max_changes,
    }

    emulator, optimizer = create_emulator_and_optimizer(
        config, shift_config, emulator_model_key
    )

    with st.spinner("Forecasting demand..."):
        if use_hybrid:
            demand_df = forecast_demand_hybrid(forecaster, selected_date)
        else:
            demand_df = forecast_demand_single(forecaster, selected_date)

    with st.spinner("Optimizing staffing..."):
        opt_result = optimizer.optimize_day(
            demand_df["predicted_calls"].tolist(), avg_handle_time
        )

    demand_df["optimal_experts"] = opt_result["optimal_experts_per_interval"]
    interval_metrics = opt_result["day_simulation"]["interval_metrics"]
    demand_df["avg_wait_time"] = [
        m.avg_wait_time if m.avg_wait_time != float("inf") else 999
        for m in interval_metrics
    ]
    demand_df["sla_compliance"] = [m.sla_compliance for m in interval_metrics]
    demand_df["utilization"] = [m.utilization_rate for m in interval_metrics]

    total_calls = demand_df["predicted_calls"].sum()
    peak_calls = demand_df["predicted_calls"].max()
    peak_experts = opt_result["peak_experts"]
    summary = opt_result["day_simulation"]["summary"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "📈 Demand Forecast",
            "👥 Staffing Schedule",
            "🔬 What-If Analysis",
            "📉 Business Analytics",
            "🤖 Model Info",
        ]
    )

    with tab1:
        st.header(f"Summary for {selected_date.strftime('%A, %B %d, %Y')}")

        day_of_week = selected_date.weekday()
        is_weekend = day_of_week >= 5
        is_tax_season = 1 <= selected_date.month <= 4
        is_tax_deadline = selected_date.month == 4 and selected_date.day <= 15

        if is_weekend:
            st.error(
                "🚫 CLOSED - Customer service is not available on weekends (Mon-Fri 5am-5pm PT only)"
            )
        elif is_tax_deadline:
            st.warning("⚠️ TAX DEADLINE PERIOD - Expect high call volume!")
        elif is_tax_season:
            st.info("📋 Tax Season - Elevated call volume expected")

        st.caption("📞 Operating Hours: Monday-Friday, 5:00 AM - 5:00 PM PT")

        if use_hybrid:
            if "model_used" in demand_df.columns:
                model_used = (
                    demand_df["model_used"].iloc[0]
                    if len(demand_df) > 0
                    else "unknown"
                )
                st.caption(f"🤖 Using: Hybrid Forecaster ({model_used} model)")
        else:
            st.caption("🤖 Using: Single Model Forecaster")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predicted Calls", f"{total_calls:,}")
        col2.metric("Peak Calls (30min)", f"{peak_calls}")
        col3.metric("Peak Experts Needed", f"{peak_experts}")
        col4.metric(
            "Total Expert-Intervals", f"{opt_result['total_experts_needed']}"
        )

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Avg Wait Time",
            f"{summary['avg_wait_time']:.1f}s",
            delta=f"Target: ≤{max_wait}s",
        )
        col2.metric(
            "SLA Compliance",
            f"{summary['avg_sla_compliance']:.1f}%",
            delta=f"Target: ≥{min_sla}%",
        )
        col3.metric(
            "Avg Utilization",
            f"{summary['avg_utilization']:.1f}%",
            delta=f"Target: ≤{max_util}%",
        )
        col4.metric(
            "Abandonment",
            f"{summary['abandonment_rate']:.1f}%",
            delta=f"Target: ≤{max_abandon}%",
        )

        st.markdown("---")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Call Volume & Staffing",
                "Wait Time",
                "SLA Compliance",
                "Utilization",
            ),
        )

        fig.add_trace(
            go.Bar(
                x=demand_df["time"],
                y=demand_df["predicted_calls"],
                name="Calls",
                marker_color="steelblue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=demand_df["time"],
                y=demand_df["optimal_experts"],
                name="Experts",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=demand_df["time"],
                y=demand_df["avg_wait_time"].clip(upper=300),
                fill="tozeroy",
                fillcolor="rgba(255,165,0,0.3)",
                line=dict(color="orange"),
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=max_wait, line_dash="dash", line_color="red", row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=demand_df["time"],
                y=demand_df["sla_compliance"],
                fill="tozeroy",
                fillcolor="rgba(0,128,0,0.3)",
                line=dict(color="green"),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=min_sla, line_dash="dash", line_color="red", row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=demand_df["time"],
                y=demand_df["utilization"],
                fill="tozeroy",
                fillcolor="rgba(128,0,128,0.3)",
                line=dict(color="purple"),
            ),
            row=2,
            col=2,
        )
        fig.add_hline(
            y=max_util, line_dash="dash", line_color="red", row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(tickangle=45, nticks=12)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Demand Forecast Details")

        colors = [
            "#1f77b4" if c < 10 else "#ff7f0e" if c < 20 else "#d62728"
            for c in demand_df["predicted_calls"]
        ]
        fig = go.Figure(
            go.Bar(
                x=demand_df["time"],
                y=demand_df["predicted_calls"],
                marker_color=colors,
                text=demand_df["predicted_calls"],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"Predicted Call Volume - {selected_date.strftime('%B %d, %Y')}",
            xaxis_title="Time",
            yaxis_title="Calls",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            hourly_df = (
                demand_df.groupby("hour")
                .agg({"predicted_calls": "sum"})
                .reset_index()
            )
            fig_hourly = px.bar(
                hourly_df,
                x="hour",
                y="predicted_calls",
                title="Calls by Hour",
                color="predicted_calls",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        with col2:
            st.subheader("📊 Key Statistics")
            st.write(
                f"**Peak Time:** {demand_df.loc[demand_df['predicted_calls'].idxmax(), 'time']}"
            )
            st.write(f"**Peak Volume:** {peak_calls} calls")
            st.write(f"**Total Daily Volume:** {total_calls} calls")
            if use_hybrid and "model_used" in demand_df.columns:
                st.write(f"**Model Used:** {demand_df['model_used'].iloc[0]}")

    with tab3:
        st.header("Optimized Staffing Schedule")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=demand_df["time"],
                y=demand_df["predicted_calls"],
                name="Calls",
                marker_color="lightblue",
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=demand_df["time"],
                y=demand_df["optimal_experts"],
                name="Experts",
                line=dict(color="red", width=3),
                mode="lines+markers",
            )
        )
        fig.update_layout(
            title="Staffing vs Demand",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig, use_container_width=True)

        schedule_df = demand_df[
            [
                "time",
                "predicted_calls",
                "optimal_experts",
                "avg_wait_time",
                "sla_compliance",
                "utilization",
            ]
        ].copy()
        schedule_df.columns = [
            "Time",
            "Calls",
            "Experts",
            "Wait (s)",
            "SLA %",
            "Util %",
        ]
        schedule_df = schedule_df.round(1)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        with col2:
            csv = schedule_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"staffing_schedule_{selected_date}.csv",
                "text/csv",
            )

    with tab4:
        st.header("What-If Analysis")

        @st.fragment
        def whatif_analysis_fragment():
            fixed_experts = st.slider("Number of Experts (constant)", 1, 30, 8, key="whatif_experts_slider")

            simulation = emulator.simulate_day(
                [fixed_experts] * len(demand_df),
                demand_df["predicted_calls"].tolist(),
                avg_handle_time,
            )
            sim_summary = simulation["summary"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Avg Wait Time",
                f"{sim_summary['avg_wait_time']:.1f}s"
                if sim_summary["avg_wait_time"] < 999
                else "∞",
            )
            col2.metric(
                "SLA Compliance", f"{sim_summary['avg_sla_compliance']:.1f}%"
            )
            col3.metric("Utilization", f"{sim_summary['avg_utilization']:.1f}%")
            col4.metric("Abandonment", f"{sim_summary['abandonment_rate']:.1f}%")

            meets_all = (
                sim_summary["avg_wait_time"] <= max_wait
                and sim_summary["avg_sla_compliance"] >= min_sla
                and sim_summary["avg_utilization"] <= max_util
                and sim_summary["abandonment_rate"] <= max_abandon
            )

            if meets_all:
                st.success(f"✅ {fixed_experts} experts meets all targets!")
            else:
                st.warning(f"⚠️ {fixed_experts} experts does NOT meet all targets.")

            st.markdown("---")

            scenarios_data = []
            for n in range(1, 21):
                sim = emulator.simulate_day(
                    [n] * len(demand_df),
                    demand_df["predicted_calls"].tolist(),
                    avg_handle_time,
                )
                s = sim["summary"]
                meets = (
                    s["avg_wait_time"] <= max_wait
                    and s["avg_sla_compliance"] >= min_sla
                    and s["avg_utilization"] <= max_util
                    and s["abandonment_rate"] <= max_abandon
                )
                scenarios_data.append(
                    {
                        "Experts": n,
                        "Wait (s)": min(s["avg_wait_time"], 999),
                        "SLA %": s["avg_sla_compliance"],
                        "Util %": s["avg_utilization"],
                        "Abandon %": s["abandonment_rate"],
                        "Meets": "✅" if meets else "❌",
                    }
                )

            scenarios_df = pd.DataFrame(scenarios_data)

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Wait Time", "SLA", "Utilization", "Abandonment"),
            )
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df["Experts"],
                    y=scenarios_df["Wait (s)"].clip(upper=300),
                    mode="lines+markers",
                    line=dict(color="orange"),
                ),
                row=1,
                col=1,
            )
            fig.add_hline(
                y=max_wait, line_dash="dash", line_color="red", row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df["Experts"],
                    y=scenarios_df["SLA %"],
                    mode="lines+markers",
                    line=dict(color="green"),
                ),
                row=1,
                col=2,
            )
            fig.add_hline(
                y=min_sla, line_dash="dash", line_color="red", row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df["Experts"],
                    y=scenarios_df["Util %"],
                    mode="lines+markers",
                    line=dict(color="purple"),
                ),
                row=2,
                col=1,
            )
            fig.add_hline(
                y=max_util, line_dash="dash", line_color="red", row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=scenarios_df["Experts"],
                    y=scenarios_df["Abandon %"],
                    mode="lines+markers",
                    line=dict(color="red"),
                ),
                row=2,
                col=2,
            )
            fig.add_hline(
                y=max_abandon, line_dash="dash", line_color="red", row=2, col=2
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Scenario Comparison Table")
            st.dataframe(
                scenarios_df.round(2), use_container_width=True, hide_index=True
            )

        whatif_analysis_fragment()

    with tab5:
        st.header("📉 Business Analytics & Insights")

        if not analytics_loaded:
            st.warning("Analytics data not available.")
            return

        st.subheader("🎯 Product Performance")
        product_stats = analytics.get_product_analysis()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(product_stats, use_container_width=True)
        with col2:
            fig = px.bar(
                product_stats.reset_index(),
                x="Product group",
                y=["Abandonment Rate", "Transfer Rate"],
                barmode="group",
                title="Abandonment & Transfer Rates by Product",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("👤 Expert Performance")

        expert_stats = analytics.get_expert_performance()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 10 Performers**")
            st.dataframe(
                expert_stats.head(10)[
                    [
                        "Total Calls",
                        "Avg Handle Time",
                        "FCR Rate",
                        "Avg CSAT",
                        "Performance Score",
                    ]
                ],
                use_container_width=True,
            )
        with col2:
            fig = px.histogram(
                expert_stats,
                x="Performance Score",
                nbins=20,
                title="Expert Performance Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔄 Repeat Caller Analysis")

        repeat_stats = analytics.get_repeat_caller_analysis()

        col1, col2, col3 = st.columns(3)
        col1.metric("Repeat Call Rate", f"{repeat_stats['repeat_call_rate']}%")
        col2.metric(
            "Unique Customers", f"{repeat_stats['unique_customers']:,}"
        )
        col3.metric("Repeat Callers", f"{repeat_stats['repeat_callers']:,}")

        st.write("**FCR & Satisfaction by Contact History**")
        history_df = repeat_stats["by_contact_history"].reset_index()
        fig = px.bar(
            history_df,
            x="Customer History / Contact history",
            y=["FCR Rate", "Avg CSAT"],
            barmode="group",
            title="First Call Resolution & CSAT by Prior Contacts",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 Self-Service Analysis")

        ss_stats = analytics.get_self_service_analysis()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                ss_stats.reset_index(),
                values="% of Total",
                names="Self service attempts before calling",
                title="Self-Service Attempts Before Calling",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(
                ss_stats[
                    ["Call Count", "Avg Handle Time", "FCR Rate", "% of Total"]
                ],
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("📞 Transfer Analysis")

        transfer_stats = analytics.get_transfer_analysis()

        col1, col2, col3 = st.columns(3)
        col1.metric("Transfer Rate", f"{transfer_stats['transfer_rate']}%")
        col2.metric(
            "Total Transfers", f"{transfer_stats['total_transfers']:,}"
        )
        col3.metric(
            "Avg Transfers/Session",
            f"{transfer_stats['avg_transfers_per_session']}",
        )

        col1, col2 = st.columns(2)
        with col1:
            dest_df = transfer_stats["by_destination"].reset_index()
            fig = px.pie(
                dest_df,
                values="% of Transfers",
                names="Transfer destination",
                title="Transfer Destinations",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            skill_df = pd.DataFrame(
                list(transfer_stats["by_skill"].items()),
                columns=["Skill", "Transfer Rate %"],
            )
            fig = px.bar(
                skill_df,
                x="Skill",
                y="Transfer Rate %",
                title="Transfer Rate by Skill",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Potential Savings Opportunities")

        savings = analytics.calculate_potential_savings()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**🔄 Repeat Call Reduction**")
            st.write(
                f"Current: {savings['repeat_call_reduction']['current_repeat_calls']:,} calls ({savings['repeat_call_reduction']['current_rate']})"
            )
            st.write(
                f"If reduced by 10%: **{savings['repeat_call_reduction']['if_reduced_by_10%']:,} fewer calls**"
            )
            st.write(
                f"Time saved: **{savings['repeat_call_reduction']['time_saved_hours']} hours**"
            )

            st.write("")
            st.write("**🤖 Self-Service Improvement**")
            st.write(
                f"Calls without self-service: {savings['self_service_improvement']['calls_without_self_service']:,} ({savings['self_service_improvement']['current_rate']})"
            )
            st.write(
                f"If 10% deflected: **{savings['self_service_improvement']['if_10%_deflected']:,} fewer calls**"
            )
            st.write(
                f"Time saved: **{savings['self_service_improvement']['time_saved_hours']} hours**"
            )

        with col2:
            st.write("**📞 Transfer Reduction**")
            st.write(
                f"Current: {savings['transfer_reduction']['current_transfers']:,} transfers ({savings['transfer_reduction']['current_rate']})"
            )
            st.write(
                f"If reduced by 20%: **{savings['transfer_reduction']['if_reduced_by_20%']:,} fewer transfers**"
            )
            st.write(
                f"Time saved: **{savings['transfer_reduction']['time_saved_hours']} hours**"
            )

            st.write("")
            st.write("**⚠️ Abandonment Recovery**")
            st.write(
                f"Current abandoned: {savings['abandonment_recovery']['current_abandoned']:,} ({savings['abandonment_recovery']['current_rate']})"
            )
            st.write(
                f"Potential to recover: **{savings['abandonment_recovery']['potential_recovered']:,} customers**"
            )

        st.markdown("---")
        st.subheader("📄 Download Full Report")

        report = analytics.generate_business_report()
        st.download_button(
            "📥 Download Business Analytics Report",
            report,
            "business_analytics_report.txt",
            "text/plain",
        )

    with tab6:
        st.header("🤖 Model Information")

        st.subheader("Current Model Selection")

        if use_hybrid:
            st.success("**Hybrid Forecaster** (Recommended)")
            st.write("""
            The hybrid forecaster uses two specialized models optimized for different forecast horizons:

            - **Short-term model** (< 7 days ahead): Optimized for predictions using recent data
              - Uses lag features (what happened 30 min, 1 hour, 1 day ago)
              - Uses rolling statistics (averages, trends from recent intervals)
              - Best for operational planning

            - **Long-term model** (≥ 7 days ahead): Optimized for predictions without recent data
              - Uses historical patterns (same day of week, same month, same hour)
              - Uses seasonal indicators (tax season, holidays)
              - Best for capacity planning
            """)

            if forecaster.max_training_date:
                st.write(
                    f"**Training data ends:** {forecaster.max_training_date.strftime('%Y-%m-%d')}"
                )
                st.write(
                    f"**Short-term threshold:** {forecaster.short_term_threshold_days} days"
                )
                st.write(
                    f"**Short-term features:** {len(forecaster.short_term_features)}"
                )
                st.write(
                    f"**Long-term features:** {len(forecaster.long_term_features)}"
                )
        else:
            st.info("**Single Model Forecaster**")
            st.write("""
            The single model uses one ensemble model for all forecast horizons:

            - Combines Gradient Boosting, Random Forest, and Ridge Regression
            - Uses 70+ engineered features
            - Works well for short-term predictions
            - May be less accurate for long-term predictions (relies on analogous historical data)
            """)

        st.markdown("---")
        st.subheader("Model Comparison")

        comparison_data = {
            "Aspect": [
                "Short-term accuracy",
                "Long-term accuracy",
                "Interpretability",
                "Training time",
                "Adaptability",
            ],
            "Single Model": [
                "⭐⭐⭐⭐⭐ Excellent",
                "⭐⭐⭐ Good",
                "⭐⭐⭐ Moderate",
                "⭐⭐⭐⭐ Fast",
                "⭐⭐⭐ Moderate",
            ],
            "Hybrid Model": [
                "⭐⭐⭐⭐⭐ Excellent",
                "⭐⭐⭐⭐ Very Good",
                "⭐⭐⭐⭐ Good",
                "⭐⭐⭐ Moderate",
                "⭐⭐⭐⭐ Good",
            ],
        }
        st.dataframe(
            pd.DataFrame(comparison_data),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.subheader("When to Use Each Model")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Use Single Model when:**")
            st.write("- Predicting for the next few days")
            st.write("- Recent data closely matches expected patterns")
            st.write("- Faster training is needed")

        with col2:
            st.write("**Use Hybrid Model when:**")
            st.write("- Predicting weeks or months ahead")
            st.write("- Planning for specific dates (tax season, holidays)")
            st.write("- Need better long-range capacity planning")

        st.markdown("---")
        st.subheader("Retrain Models")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Retrain Single Model"):
                with st.spinner("Retraining single model..."):
                    new_forecaster = CallDemandForecaster()
                    new_forecaster.train(DATA_PATH)
                    new_forecaster.save_model(MODEL_PATH)
                    st.success("Single model retrained!")
                    st.cache_resource.clear()
                    st.rerun()

        with col2:
            if st.button("🔄 Retrain Hybrid Model"):
                with st.spinner("Retraining hybrid model..."):
                    new_forecaster = HybridForecaster()
                    new_forecaster.train(DATA_PATH)
                    new_forecaster.save_model(HYBRID_MODEL_PATH)
                    st.success("Hybrid model retrained!")
                    st.cache_resource.clear()
                    st.rerun()


if __name__ == "__main__":
    main()
