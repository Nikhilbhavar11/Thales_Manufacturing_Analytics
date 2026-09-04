import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Thales Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

RISK_COLORS = {'High': '#c1272d', 'Medium': '#f4a261', 'Low': '#2dc653', 'Not Active': '#888888'}

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_manufacturing_data.csv', parse_dates=['datetime'])
    active = pd.read_csv('data/active_anomaly_data.csv', parse_dates=['datetime'])
    hr_machines = pd.read_csv('data/high_risk_machines.csv')
    return df, active, hr_machines

df, active_df, hr_machines = load_data()

all_machines = sorted(df['Machine_ID'].unique().tolist())
all_modes = sorted(df['Operation_Mode'].unique().tolist())

st.sidebar.title("⚙️ Filters")
selected_machines = st.sidebar.multiselect(
    "Select Machines", options=all_machines,
    default=all_machines[:10],
    help="Select one or more Machine IDs to analyze"
)
risk_threshold = st.sidebar.slider(
    "High Risk Threshold", min_value=0.5, max_value=0.95,
    value=0.7, step=0.05,
    help="Anomaly score above this value is flagged as High Risk"
)
selected_modes = st.sidebar.multiselect(
    "Operation Mode", options=all_modes,
    default=all_modes
)
min_date = df['datetime'].min().date()
max_date = df['datetime'].max().date()
date_range = st.sidebar.date_input(
    "Time Window", value=[min_date, max_date],
    min_value=min_date, max_value=max_date
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "Predictive maintenance analytics for 6G-integrated smart manufacturing. "
    "Isolation Forest anomaly detection across 50 machines, Jan–Mar 2025."
)

if len(date_range) == 2:
    filt = (
        (df['Machine_ID'].isin(selected_machines)) &
        (df['Operation_Mode'].isin(selected_modes)) &
        (df['datetime'].dt.date >= date_range[0]) &
        (df['datetime'].dt.date <= date_range[1])
    )
    filt_active = (
        (active_df['Machine_ID'].isin(selected_machines)) &
        (active_df['Operation_Mode'].isin(selected_modes)) &
        (active_df['datetime'].dt.date >= date_range[0]) &
        (active_df['datetime'].dt.date <= date_range[1])
    )
else:
    filt = df['Machine_ID'].isin(selected_machines)
    filt_active = active_df['Machine_ID'].isin(selected_machines)

filtered_df = df[filt].copy()
filtered_active = active_df[filt_active].copy()

filtered_active['risk_level'] = filtered_active['anomaly_score'].apply(
    lambda s: 'High' if s >= risk_threshold else ('Medium' if s >= 0.4 else 'Low')
)

st.title("Predictive Maintenance & Anomaly Detection")
st.markdown("**Thales Group — 6G-Integrated Smart Manufacturing | Jan–Mar 2025**")
st.markdown("---")

if filtered_active.empty:
    st.warning("No data for selected filters.")
    st.stop()

total_records = len(filtered_active)
high_risk = (filtered_active['risk_level'] == 'High').sum()
medium_risk = (filtered_active['risk_level'] == 'Medium').sum()
avg_score = filtered_active['anomaly_score'].mean()
machines_at_risk = filtered_active[filtered_active['risk_level'] == 'High']['Machine_ID'].nunique()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Maintenance Overview",
    "🔍 Machine Anomaly Dashboard",
    "🚨 Maintenance Alert Panel",
    "📈 Historical Risk Analysis"
])

with tab1:
    st.subheader("Predictive Maintenance Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active Records Analyzed", f"{total_records:,}")
    c2.metric("High Risk Events", f"{high_risk:,}",
              delta=f"{high_risk/total_records*100:.1f}% of total",
              delta_color="inverse")
    c3.metric("Medium Risk Events", f"{medium_risk:,}",
              delta=f"{medium_risk/total_records*100:.1f}% of total",
              delta_color="off")
    c4.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    c5.metric("Machines with High Risk", f"{machines_at_risk}",
              delta=f"of {len(selected_machines)} selected",
              delta_color="inverse")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        risk_counts = filtered_active['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig_pie = px.pie(
            risk_counts, values='Count', names='Risk Level',
            color='Risk Level',
            color_discrete_map=RISK_COLORS,
            title='Risk Distribution Across Selected Machines'
        )
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        machine_risk_summary = filtered_active.groupby('Machine_ID').agg(
            avg_score=('anomaly_score', 'mean'),
            high_risk_count=('risk_level', lambda x: (x == 'High').sum()),
            total=('risk_level', 'count')
        ).reset_index()
        machine_risk_summary = machine_risk_summary.sort_values('avg_score', ascending=False)

        fig_bar = px.bar(
            machine_risk_summary.head(15), x='Machine_ID', y='avg_score',
            color='avg_score',
            color_continuous_scale=[[0, '#2dc653'], [0.5, '#f4a261'], [1, '#c1272d']],
            title='Top 15 Machines by Average Anomaly Score'
        )
        fig_bar.add_hline(y=risk_threshold, line_dash='dash', line_color='red',
                          annotation_text=f'High Risk ({risk_threshold})')
        fig_bar.add_hline(y=0.4, line_dash='dash', line_color='orange',
                          annotation_text='Medium Risk (0.4)')
        fig_bar.update_layout(height=360, coloraxis_showscale=False,
                              xaxis_title='Machine ID', yaxis_title='Avg Anomaly Score')
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=filtered_active['anomaly_score'], nbinsx=80,
        marker_color='steelblue', opacity=0.8
    ))
    fig_hist.add_vline(x=risk_threshold, line_dash='dash', line_color='red',
                       annotation_text=f'High Risk ({risk_threshold})',
                       annotation_font_color='red')
    fig_hist.add_vline(x=0.4, line_dash='dash', line_color='orange',
                       annotation_text='Medium Risk (0.4)',
                       annotation_font_color='orange')
    fig_hist.update_layout(height=320, title='Anomaly Score Distribution',
                           xaxis_title='Anomaly Score', yaxis_title='Count', showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    mode_risk = filtered_active.groupby(['Operation_Mode', 'risk_level']).size().reset_index(name='count')
    fig_mode = px.bar(
        mode_risk, x='Operation_Mode', y='count',
        color='risk_level', color_discrete_map=RISK_COLORS,
        barmode='group', title='Risk Events by Operation Mode'
    )
    fig_mode.update_layout(height=340, xaxis_title='Operation Mode', yaxis_title='Count')
    st.plotly_chart(fig_mode, use_container_width=True)

with tab2:
    st.subheader("Machine Anomaly Dashboard")

    machine_select = st.selectbox(
        "Select Machine to Inspect",
        options=sorted(filtered_active['Machine_ID'].unique().tolist()),
        format_func=lambda x: f"Machine {x}"
    )

    mdf = filtered_active[filtered_active['Machine_ID'] == machine_select].sort_values('datetime')

    m_high = (mdf['risk_level'] == 'High').sum()
    m_medium = (mdf['risk_level'] == 'Medium').sum()
    m_avg = mdf['anomaly_score'].mean()
    m_max = mdf['anomaly_score'].max()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("High Risk Events", f"{m_high}")
    mc2.metric("Medium Risk Events", f"{m_medium}")
    mc3.metric("Avg Anomaly Score", f"{m_avg:.3f}")
    mc4.metric("Peak Anomaly Score", f"{m_max:.3f}")

    fig_machine = go.Figure()
    fig_machine.add_trace(go.Scatter(
        x=mdf['datetime'], y=mdf['anomaly_score'],
        mode='lines', name='Anomaly Score',
        line=dict(color='steelblue', width=0.8)
    ))
    fig_machine.add_trace(go.Scatter(
        x=mdf[mdf['risk_level'] == 'High']['datetime'],
        y=mdf[mdf['risk_level'] == 'High']['anomaly_score'],
        mode='markers', name='High Risk',
        marker=dict(color='#c1272d', size=5)
    ))
    fig_machine.add_hline(y=risk_threshold, line_dash='dash', line_color='red',
                          annotation_text=f'High Risk ({risk_threshold})')
    fig_machine.add_hline(y=0.4, line_dash='dash', line_color='orange',
                          annotation_text='Medium Risk (0.4)')
    fig_machine.update_layout(
        height=380, title=f'Machine {machine_select} — Anomaly Score Timeline',
        xaxis_title='Datetime', yaxis_title='Anomaly Score',
        yaxis=dict(range=[0, 1.05]), hovermode='x unified'
    )
    st.plotly_chart(fig_machine, use_container_width=True)

    dev_cols = [c for c in mdf.columns if '_deviation' in c]
    sensor_labels = [c.replace('_deviation', '').replace('_', ' ') for c in dev_cols]
    avg_deviations = mdf[dev_cols].mean().values

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=np.abs(avg_deviations), theta=sensor_labels,
        fill='toself', name=f'Machine {machine_select}',
        line=dict(color='steelblue')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
        title=f'Machine {machine_select} — Average Sensor Deviation (Z-score)',
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    sensor_cols_base = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
                        'Network_Latency_ms', 'Error_Rate_%']
    high_risk_mdf = mdf[mdf['risk_level'] == 'High']
    normal_mdf = mdf[mdf['risk_level'] == 'Low']

    if not high_risk_mdf.empty and not normal_mdf.empty:
        compare_data = []
        for col in sensor_cols_base:
            if col in mdf.columns:
                compare_data.append({
                    'Sensor': col.replace('_', ' '),
                    'High Risk Avg': high_risk_mdf[col].mean(),
                    'Normal Avg': normal_mdf[col].mean()
                })
        compare_df = pd.DataFrame(compare_data)
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='High Risk Avg', x=compare_df['Sensor'],
            y=compare_df['High Risk Avg'], marker_color='#c1272d'
        ))
        fig_compare.add_trace(go.Bar(
            name='Normal Avg', x=compare_df['Sensor'],
            y=compare_df['Normal Avg'], marker_color='#2dc653'
        ))
        fig_compare.update_layout(
            barmode='group', height=360,
            title=f'Machine {machine_select} — High Risk vs Normal Sensor Averages',
            xaxis_title='Sensor', yaxis_title='Value'
        )
        st.plotly_chart(fig_compare, use_container_width=True)

with tab3:
    st.subheader("Maintenance Alert Panel")

    high_risk_records = filtered_active[
        filtered_active['risk_level'] == 'High'
    ].sort_values('anomaly_score', ascending=False)

    if not high_risk_records.empty:
        st.warning(f"⚠️ {machines_at_risk} machine(s) require urgent inspection. "
                   f"{len(high_risk_records)} high risk events detected.")

    priority_table = (
        filtered_active[filtered_active['risk_level'] == 'High']
        .groupby('Machine_ID')
        .agg(
            high_risk_events=('risk_level', 'count'),
            avg_anomaly_score=('anomaly_score', 'mean'),
            peak_anomaly_score=('anomaly_score', 'max'),
            last_high_risk=('datetime', 'max')
        )
        .sort_values('peak_anomaly_score', ascending=False)
        .reset_index()
    )
    priority_table['avg_anomaly_score'] = priority_table['avg_anomaly_score'].round(4)
    priority_table['peak_anomaly_score'] = priority_table['peak_anomaly_score'].round(4)
    priority_table['Priority'] = priority_table['peak_anomaly_score'].apply(
        lambda x: '🔴 Critical' if x >= 0.85 else '🟠 High'
    )
    priority_table.columns = ['Machine ID', 'High Risk Events', 'Avg Score',
                               'Peak Score', 'Last High Risk Event', 'Priority']
    st.subheader("Priority Inspection Queue")
    st.dataframe(priority_table, use_container_width=True, hide_index=True)

    recent_alerts = high_risk_records[
        ['datetime', 'Machine_ID', 'Operation_Mode', 'anomaly_score',
         'Temperature_C', 'Vibration_Hz', 'Error_Rate_%', 'Predictive_Maintenance_Score']
    ].head(50).copy()
    recent_alerts['anomaly_score'] = recent_alerts['anomaly_score'].round(4)
    recent_alerts['datetime'] = recent_alerts['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    recent_alerts.columns = ['Datetime', 'Machine ID', 'Operation Mode', 'Anomaly Score',
                              'Temp (°C)', 'Vibration (Hz)', 'Error Rate (%)', 'Maintenance Score']
    st.subheader("Recent High Risk Events")
    st.dataframe(recent_alerts, use_container_width=True, hide_index=True)

    dpi_data = (
        filtered_active[filtered_active['risk_level'] == 'High']
        .groupby('Machine_ID')
        .agg(high_risk_events=('risk_level', 'count'))
        .reset_index()
    )
    dpi_data['estimated_risk_hours'] = (dpi_data['high_risk_events'] / 60).round(2)
    dpi_data = dpi_data.sort_values('estimated_risk_hours', ascending=False).head(10)

    fig_dpi = px.bar(
        dpi_data, x='Machine_ID', y='estimated_risk_hours',
        color='estimated_risk_hours',
        color_continuous_scale=[[0, '#f4a261'], [1, '#c1272d']],
        title='Estimated Risk Hours per Machine (Top 10)',
        labels={'estimated_risk_hours': 'Risk Hours', 'Machine_ID': 'Machine ID'}
    )
    fig_dpi.update_layout(height=350, coloraxis_showscale=False)
    st.subheader("Downtime Prevention Index")
    st.plotly_chart(fig_dpi, use_container_width=True)

with tab4:
    st.subheader("Historical Risk Analysis")

    daily_risk = (
        filtered_active
        .groupby([filtered_active['datetime'].dt.date, 'risk_level'])
        .size().unstack(fill_value=0).reset_index()
    )
    daily_risk.columns.name = None
    daily_risk['datetime'] = pd.to_datetime(daily_risk['datetime'])
    for col in ['Low', 'Medium', 'High']:
        if col not in daily_risk.columns:
            daily_risk[col] = 0

    fig_stack = go.Figure()
    fig_stack.add_trace(go.Scatter(
        x=daily_risk['datetime'], y=daily_risk['Low'],
        stackgroup='one', name='Low', fill='tonexty',
        line=dict(color='#2dc653', width=0)
    ))
    fig_stack.add_trace(go.Scatter(
        x=daily_risk['datetime'], y=daily_risk['Medium'],
        stackgroup='one', name='Medium', fill='tonexty',
        line=dict(color='#f4a261', width=0)
    ))
    fig_stack.add_trace(go.Scatter(
        x=daily_risk['datetime'], y=daily_risk['High'],
        stackgroup='one', name='High', fill='tonexty',
        line=dict(color='#c1272d', width=0)
    ))
    fig_stack.update_layout(height=380, title='Daily Risk Level Distribution Over Time',
                            xaxis_title='Date', yaxis_title='Event Count', hovermode='x unified')
    st.plotly_chart(fig_stack, use_container_width=True)

    top_machines_list = hr_machines.head(10)['Machine_ID'].tolist()
    selected_top = st.multiselect(
        "Select machines for escalation view",
        options=top_machines_list, default=top_machines_list[:5],
        format_func=lambda x: f"Machine {x}"
    )

    if selected_top:
        escalation_df = (
            filtered_active[filtered_active['Machine_ID'].isin(selected_top)]
            .groupby([filtered_active['datetime'].dt.date, 'Machine_ID'])['anomaly_score']
            .mean().reset_index()
        )
        escalation_df['datetime'] = pd.to_datetime(escalation_df['datetime'])
        fig_esc = px.line(
            escalation_df, x='datetime', y='anomaly_score', color='Machine_ID',
            title='Daily Average Anomaly Score — Selected Machines',
            labels={'anomaly_score': 'Avg Anomaly Score', 'datetime': 'Date', 'Machine_ID': 'Machine'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_esc.add_hline(y=risk_threshold, line_dash='dash', line_color='red',
                          annotation_text=f'High Risk ({risk_threshold})')
        fig_esc.add_hline(y=0.4, line_dash='dash', line_color='orange',
                          annotation_text='Medium Risk')
        fig_esc.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_esc, use_container_width=True)

    hourly_risk = (
        filtered_active
        .groupby([filtered_active['datetime'].dt.hour, 'risk_level'])
        .size().unstack(fill_value=0).reset_index()
    )
    hourly_risk.columns.name = None
    hourly_risk.rename(columns={'datetime': 'Hour'}, inplace=True)

    fig_hourly = go.Figure()
    for risk, color in [('High', '#c1272d'), ('Medium', '#f4a261'), ('Low', '#2dc653')]:
        if risk in hourly_risk.columns:
            fig_hourly.add_trace(go.Bar(
                x=hourly_risk['Hour'], y=hourly_risk[risk],
                name=risk, marker_color=color
            ))
    fig_hourly.update_layout(barmode='stack', height=350,
                             title='Risk Event Distribution by Hour of Day',
                             xaxis_title='Hour of Day', yaxis_title='Count')
    st.plotly_chart(fig_hourly, use_container_width=True)

    weekly_risk = (
        filtered_active[filtered_active['risk_level'] == 'High']
        .groupby('week')
        .agg(high_risk_count=('risk_level', 'count'),
             avg_score=('anomaly_score', 'mean'),
             machines_affected=('Machine_ID', 'nunique'))
        .reset_index()
    )
    if not weekly_risk.empty:
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Bar(
            x=weekly_risk['week'], y=weekly_risk['high_risk_count'],
            name='High Risk Events', marker_color='#c1272d', yaxis='y'
        ))
        fig_weekly.add_trace(go.Scatter(
            x=weekly_risk['week'], y=weekly_risk['machines_affected'],
            name='Machines Affected', line=dict(color='#1f4e8c', width=2),
            mode='lines+markers', yaxis='y2'
        ))
        fig_weekly.update_layout(
            height=360, title='Weekly High Risk Events and Machines Affected',
            xaxis_title='Week Number',
            yaxis=dict(title='High Risk Event Count'),
            yaxis2=dict(title='Machines Affected', overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

    