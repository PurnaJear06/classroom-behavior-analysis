"""
Streamlit dashboard for visualizing classroom engagement data
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

def load_summary(summary_path):
    """Load engagement summary data"""
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
    return summary_data

def main():
    st.set_page_config(page_title="Classroom Engagement Analysis", layout="wide")
    
    # App title and description
    st.title("Classroom Engagement Analysis Dashboard")
    st.write("Analyze student engagement and behavior patterns from your lecture recordings.")
    
    # Sidebar for navigation and file selection
    st.sidebar.header("Navigation")
    
    # Find available summary files
    output_dir = "output"
    summary_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith("engagement_summary_") and file.endswith(".json"):
                summary_files.append(os.path.join(output_dir, file))
    
    if not summary_files:
        st.warning("No engagement summary files found. Please run the analysis first.")
        st.info("To run analysis: `python main.py --video path/to/video.mp4`")
        return
    
    # File selection
    selected_file = st.sidebar.selectbox("Select a session to analyze:", summary_files)
    
    # Load selected summary data
    summary_data = load_summary(selected_file)
    
    # Extract data
    metadata = summary_data.get("metadata", {})
    class_summary = summary_data.get("class_summary", {})
    student_summaries = summary_data.get("student_summaries", {})
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Class Overview", "Student Analysis", "Timeline View"])
    
    with tab1:
        st.header("Class Overview")
        
        # Display session information
        st.subheader("Session Information")
        col1, col2 = st.columns(2)
        
        with col1:
            session_id = metadata.get("session_id", "Unknown")
            timestamp = metadata.get("timestamp", "Unknown")
            st.write(f"**Session ID:** {session_id}")
            st.write(f"**Timestamp:** {timestamp}")
        
        with col2:
            st.write(f"**Total Students:** {class_summary.get('total_students', 0)}")
            duration_mins = class_summary.get('session_duration', 0) / 60
            st.write(f"**Session Duration:** {duration_mins:.1f} minutes")
        
        # Display class summary
        st.subheader("Class Engagement Metrics")
        
        # Create overall engagement metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engagement_score = class_summary.get("engagement_score", 0)
            st.metric("Engagement Score", f"{engagement_score:.1f}%")
        
        with col2:
            most_common = class_summary.get("most_common_behavior", "unknown")
            st.metric("Most Common Behavior", most_common)
        
        with col3:
            disengaged = class_summary.get("behavior_percentages", {}).get("disengaged", 0)
            st.metric("Disengagement Rate", f"{disengaged:.1f}%")
        
        # Create behavior pie chart
        st.subheader("Behavior Distribution")
        
        behavior_percentages = class_summary.get("behavior_percentages", {})
        
        if behavior_percentages:
            fig = px.pie(
                names=list(behavior_percentages.keys()),
                values=list(behavior_percentages.values()),
                title="Class Behavior Distribution",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig)
        else:
            st.write("No behavior data available.")
            
        # Class behavior bar chart
        if behavior_percentages:
            behavior_df = pd.DataFrame({
                'Behavior': list(behavior_percentages.keys()),
                'Percentage': list(behavior_percentages.values())
            })
            
            fig = px.bar(
                behavior_df,
                x='Behavior',
                y='Percentage',
                title="Behavior Distribution",
                color='Behavior'
            )
            st.plotly_chart(fig)
    
    with tab2:
        st.header("Student Analysis")
        
        # Convert student data to DataFrame for easier manipulation
        student_data = []
        for student_id, data in student_summaries.items():
            student_data.append({
                "Student ID": student_id,
                "Total Detections": data.get("total_detections", 0),
                "Most Common Behavior": data.get("most_common_behavior", "unknown"),
                "Attentive (%)": data.get("behavior_percentages", {}).get("attentive", 0),
                "Disengaged (%)": data.get("behavior_percentages", {}).get("disengaged", 0),
                "Other (%)": data.get("behavior_percentages", {}).get("other_behavior", 0),
                "Latest Snapshot": data.get("latest_snapshot", None)
            })
        
        if student_data:
            student_df = pd.DataFrame(student_data)
            
            # Allow filtering and sorting
            st.write("### Student Engagement Data")
            st.dataframe(student_df)
            
            # Student comparison chart
            st.write("### Student Comparison")
            
            chart_type = st.radio("Chart Type", ["Bar Chart", "Scatter Plot"])
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    student_df,
                    x="Student ID",
                    y=["Attentive (%)", "Disengaged (%)", "Other (%)"],
                    title="Behavior Comparison Across Students",
                    barmode="group"
                )
                st.plotly_chart(fig)
            else:
                fig = px.scatter(
                    student_df,
                    x="Attentive (%)",
                    y="Disengaged (%)",
                    size="Total Detections",
                    color="Most Common Behavior",
                    hover_name="Student ID",
                    title="Student Engagement Scatter Plot"
                )
                st.plotly_chart(fig)
            
            # Student detail view
            st.write("### Student Detail View")
            selected_student = st.selectbox("Select a student to view details:", student_df["Student ID"].unique())
            
            if selected_student:
                student_info = student_summaries.get(str(selected_student), {})
                
                # Display snapshot if available
                snapshot_path = student_info.get("latest_snapshot")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if snapshot_path and os.path.exists(os.path.join(output_dir, snapshot_path)):
                        img = Image.open(os.path.join(output_dir, snapshot_path))
                        st.image(img, caption=f"Student {selected_student}")
                    else:
                        st.write("No snapshot available")
                
                with col2:
                    # Student behavior pie chart
                    behavior_percentages = student_info.get("behavior_percentages", {})
                    
                    if behavior_percentages:
                        fig = px.pie(
                            names=list(behavior_percentages.keys()),
                            values=list(behavior_percentages.values()),
                            title=f"Student {selected_student} Behavior Distribution",
                            color_discrete_sequence=px.colors.qualitative.Plotly
                        )
                        st.plotly_chart(fig)
                
                # Show behavior stats
                st.write("### Behavior Statistics")
                behavior_counts = student_info.get("behavior_counts", {})
                behavior_durations = student_info.get("behavior_durations", {})
                
                if behavior_counts and behavior_durations:
                    stats_df = pd.DataFrame({
                        'Behavior': list(behavior_counts.keys()),
                        'Count': list(behavior_counts.values()),
                        'Duration (s)': [behavior_durations.get(b, 0) for b in behavior_counts.keys()],
                        'Percentage (%)': [behavior_percentages.get(b, 0) for b in behavior_counts.keys()]
                    })
                    
                    st.dataframe(stats_df)
        else:
            st.write("No student data available.")
    
    with tab3:
        st.header("Timeline View")
        st.write("Analysis of behavior changes over time")
        
        # Mock timeline data (in a real implementation, this would come from the log data)
        # Extract timestamps and create bins
        timestamps = []
        behaviors = []
        
        # Find the log file
        log_file = metadata.get("log_file", "")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                entries = log_data.get("entries", [])
                
                # Aggregate data by time bins
                if entries:
                    # Sort entries by timestamp
                    entries = sorted(entries, key=lambda x: x.get("timestamp", 0))
                    
                    # Get time range
                    start_time = entries[0].get("timestamp", 0)
                    end_time = entries[-1].get("timestamp", 0)
                    duration = end_time - start_time
                    
                    # Create time bins (10 bins across the session)
                    num_bins = 10
                    bin_duration = duration / num_bins
                    
                    # Initialize data structures
                    time_bins = [start_time + i * bin_duration for i in range(num_bins + 1)]
                    time_labels = [f"{(t - start_time) / 60:.1f}" for t in time_bins]
                    behavior_bins = {
                        "attentive": [0] * num_bins,
                        "disengaged": [0] * num_bins,
                        "other_behavior": [0] * num_bins
                    }
                    
                    # Count behaviors in bins
                    for entry in entries:
                        timestamp = entry.get("timestamp", 0)
                        behavior = entry.get("behavior", "unknown")
                        
                        # Find bin index
                        bin_idx = min(int((timestamp - start_time) / bin_duration), num_bins - 1)
                        
                        # Increment behavior count for this bin
                        if behavior in behavior_bins:
                            behavior_bins[behavior][bin_idx] += 1
                    
                    # Create DataFrame for plotting
                    timeline_data = {
                        "Time (minutes)": time_labels[:-1],
                    }
                    
                    for behavior, counts in behavior_bins.items():
                        timeline_data[behavior] = counts
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    # Plot timeline
                    st.subheader("Behavior Changes Over Time")
                    fig = px.line(
                        timeline_df, 
                        x="Time (minutes)", 
                        y=list(behavior_bins.keys()),
                        title="Behavior Trends During Class Session",
                        markers=True
                    )
                    st.plotly_chart(fig)
                    
                    # Create stacked area chart
                    st.subheader("Behavior Composition Over Time")
                    fig = px.area(
                        timeline_df,
                        x="Time (minutes)",
                        y=list(behavior_bins.keys()),
                        title="Behavior Composition During Class Session"
                    )
                    st.plotly_chart(fig)
                else:
                    st.write("No timeline data available.")
        else:
            st.write("Log file not found. Cannot generate timeline view.")
    
    # Include instructions for reading the dashboard
    st.sidebar.subheader("How to use this dashboard")
    st.sidebar.write("""
    1. Select a session from the dropdown menu
    2. View overall class engagement metrics in the Class Overview tab
    3. Examine individual student behavior in the Student Analysis tab
    4. See how behaviors changed over time in the Timeline View tab
    """)
    
    # About section
    st.sidebar.subheader("About")
    st.sidebar.write("""
    This dashboard visualizes classroom engagement data captured by the 
    Classroom Behavior Analysis System using computer vision and YOLOv8.
    """)

if __name__ == "__main__":
    main() 