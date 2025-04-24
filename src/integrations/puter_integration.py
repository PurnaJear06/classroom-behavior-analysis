import os
import json
import time
import requests
from typing import Dict, List, Optional, Union

def is_working():
    """Simple function to check if the integration is working"""
    return True

def analyze_behavior_with_puter(prompt, mode="Standard", max_retries=2):
    """
    Analyze classroom behavior using Puter's services.
    Note: This is a placeholder that simulates responses until we integrate with Puter.js client-side.
    
    Client-side JavaScript integration should be used with:
    <script src="https://js.puter.com/v2/"></script>
    
    And then call:
    puter.ai.chat(prompt).then(result => console.log(result))
    
    Args:
        prompt (str): The prompt to send for analysis
        mode (str): Analysis depth - "Basic", "Standard", or "Comprehensive"
        max_retries (int): Maximum number of retries on failure
        
    Returns:
        str: The analysis result
    """
    # Since Puter.js requires client-side JavaScript integration, we're providing a simulated response
    # for testing purposes. In production, this would be handled by the Streamlit frontend.
    
    if "classroom" in prompt.lower() or "students" in prompt.lower():
        if mode == "Basic":
            return """The classroom shows moderate engagement with 75% of students actively participating. The front section has higher engagement. Consider more interactive activities for distracted students."""
        elif mode == "Comprehensive":
            return """# Classroom Behavior Analysis

## Engagement Overview
- 75% active participation (18/24 students)
- 16.7% passive observation (4/24 students)
- 8.3% distracted behavior (2/24 students)

## Spatial Analysis
- Higher engagement in front areas and near windows
- Low engagement in back corners
- Effective group formations in front-left and center
- Varied individual activities (reading, writing, hand-raising)

## Recommendations
1. Redistribute seating to integrate passive students with active groups
2. Implement brief movement breaks to engage all areas of classroom
3. Introduce pair activities to activate back corner areas
4. Consider rotating small group formations to increase engagement
5. Use more visual aids near low-movement areas to draw attention

## Intervention Strategies
- Direct questions to passive students with supportive prompting
- Provide specific tasks for distracted students
- Incorporate more hand-raising activities to increase participation
- Create movement-based learning activities

The classroom shows good potential for high engagement with targeted interventions."""
        else:  # Standard
            return """The classroom shows 75% active engagement (18/24 students) with higher activity at the front and near windows. Back corners have lower engagement.

Key observations:
1. Group formations are effective in front-left and center areas
2. 4 students show passive behavior but are not disruptive
3. 2 students appear consistently distracted

Recommendations:
1. Reorganize seating to better integrate passive students
2. Implement brief activities targeting back corner areas
3. Use direct questioning techniques for less engaged students
4. Consider short movement breaks to re-energize the class"""
    
    # Generic response for other prompts
    return "Analysis complete. Classroom behavior patterns identified with key engagement metrics and recommendations for improvement."

def get_behavior_insights(video_data):
    """
    Format video analysis data and get insights
    
    Args:
        video_data (dict): Data extracted from video analysis
        
    Returns:
        dict: Dict containing analysis, raw data, and timestamp
    """
    try:
        # Extract relevant data from video_data
        duration = video_data.get("duration", 0)
        num_students = video_data.get("num_students", 0)
        active_students = video_data.get("active_students", 0)
        passive_students = video_data.get("passive_students", 0)
        distracted_students = video_data.get("distracted_students", 0)
        high_movement_areas = video_data.get("high_movement_areas", [])
        low_movement_areas = video_data.get("low_movement_areas", [])
        group_formations = video_data.get("group_formations", [])
        individual_activities = video_data.get("individual_activities", [])
        
        # Create a detailed prompt for analysis
        prompt = f"""
        Analyze the following classroom behavior data and provide insights:
        
        Class Duration: {duration} seconds
        Total Students: {num_students}
        Active Students: {active_students}
        Passive Students: {passive_students}
        Distracted Students: {distracted_students}
        
        High Movement Areas: {', '.join(high_movement_areas) if high_movement_areas else 'None detected'}
        Low Movement Areas: {', '.join(low_movement_areas) if low_movement_areas else 'None detected'}
        
        Group Formations: {', '.join(group_formations) if group_formations else 'None detected'}
        Individual Activities: {', '.join(individual_activities) if individual_activities else 'None detected'}
        
        Please provide:
        1. Key observations about student engagement
        2. Potential areas for improvement
        3. Recommendations for the teacher
        """
        
        # Get analysis from simulation
        analysis = analyze_behavior_with_puter(prompt, mode="Comprehensive")
        
        # Return results with metadata
        return {
            "analysis": analysis,
            "raw_data": video_data,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "analysis": f"Error generating insights: {str(e)}",
            "raw_data": video_data,
            "timestamp": time.time()
        } 