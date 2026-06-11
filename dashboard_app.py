import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

API_URL = "http://127.0.0.1:8000/analytics"

st.set_page_config(
    page_title="PersonaForge Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode professional styling
st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E2127;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

def fetch_data(endpoint):
    try:
        response = requests.get(f"{API_URL}/{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
    return {}

st.sidebar.title("PersonaForge AI")
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Performance", 
    "Identity Metrics", 
    "Benchmark Results", 
    "System Health"
])

st.sidebar.markdown("---")
st.sidebar.info("Dashboard auto-refreshes metrics when switching pages.")

if page == "Overview":
    st.title("Production Overview")
    data = fetch_data("overview")
    
    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Jobs", data.get("total_jobs", 0))
        col2.metric("Success Rate", f"{data.get('success_rate', 0)}%")
        col3.metric("Failed Jobs", data.get("failed_jobs", 0))
        col4.metric("Avg Processing Time", f"{data.get('average_processing_time', 0)}s")
        
        st.markdown("### Job Status Breakdown")
        success = data.get("total_jobs", 0) - data.get("failed_jobs", 0)
        fig = px.pie(
            values=[success, data.get("failed_jobs", 0)], 
            names=['Success', 'Failed'],
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Performance":
    st.title("Performance Analytics")
    data = fetch_data("performance")
    
    if data and data.get("job_ids"):
        df = pd.DataFrame({
            "Job ID": data["job_ids"],
            "Processing Time (s)": data["processing_times"],
            "Queue Length": data["queue_lengths"]
        })
        
        st.markdown("### Recent Processing Times")
        fig_time = px.line(df, x="Job ID", y="Processing Time (s)", markers=True)
        fig_time.update_traces(line_color='#AB63FA', line_width=3)
        fig_time.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_time, use_container_width=True)
        
        st.markdown("### Queue Length Trend")
        fig_queue = px.bar(df, x="Job ID", y="Queue Length")
        fig_queue.update_traces(marker_color='#FFA15A')
        fig_queue.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_queue, use_container_width=True)
    else:
        st.info("No recent performance data available.")

elif page == "Identity Metrics":
    st.title("Identity Score Trends")
    data = fetch_data("identity")
    
    if data and data.get("trends"):
        df = pd.DataFrame(data["trends"])
        
        st.markdown("### Identity Consistency Score across Jobs")
        # Color by drift
        fig = px.scatter(
            df, x="job_id", y="score", 
            color="drift_detected",
            color_discrete_map={True: "red", False: "lightgreen"},
            size=[10]*len(df)
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No identity reports generated yet.")

elif page == "Benchmark Results":
    st.title("Benchmark Results")
    st.markdown("""
    This page aggregates benchmark metrics from the Face Quality Assessment System.
    Future updates will pull direct quality scores from the pipeline.
    """)
    
    # Mock data for demonstration
    benchmark_df = pd.DataFrame({
        "Model": ["Ghost", "Inswapper", "SimSwap"],
        "Avg Quality": [92.5, 88.0, 81.2],
        "Avg FPS": [24.5, 30.1, 15.0]
    })
    
    fig = px.bar(benchmark_df, x="Model", y=["Avg Quality", "Avg FPS"], barmode="group")
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

elif page == "System Health":
    st.title("Live System Health")
    
    col1, col2, col3 = st.columns(3)
    
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()
    placeholder3 = col3.empty()
    
    st.markdown("### Real-Time Utilization")
    chart_placeholder = st.empty()
    
    # Keep history for chart
    history = {"Time": [], "CPU": [], "RAM": [], "GPU": []}
    
    # Run loop to update system health for a few iterations (or until page changes)
    for i in range(20): # Limiting loop so it doesn't run forever in the background
        sys_data = fetch_data("system")
        if sys_data:
            cpu = sys_data.get("cpu_utilization", 0)
            ram = sys_data.get("memory_utilization", 0)
            gpu = sys_data.get("gpu_utilization", 0)
            
            placeholder1.metric("CPU Utilization", f"{cpu}%")
            placeholder2.metric("RAM Utilization", f"{ram}%")
            placeholder3.metric("GPU Utilization", f"{gpu}%")
            
            history["Time"].append(i)
            history["CPU"].append(cpu)
            history["RAM"].append(ram)
            history["GPU"].append(gpu)
            
            df = pd.DataFrame(history)
            fig = px.line(df, x="Time", y=["CPU", "RAM", "GPU"])
            fig.update_layout(template="plotly_dark", yaxis_range=[0, 100], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        time.sleep(2)
