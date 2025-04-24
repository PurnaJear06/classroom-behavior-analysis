"""
Summary generator for creating engagement summaries from behavior logs
"""

import os
import json
from collections import defaultdict
from datetime import datetime

class SummaryGenerator:
    def __init__(self, log_file):
        """
        Initialize the summary generator
        
        Args:
            log_file: Path to the behavior log file
        """
        self.log_file = log_file
        self.output_dir = os.path.dirname(os.path.dirname(log_file))
    
    def generate_summary(self):
        """
        Generate engagement summary from behavior logs
        
        Returns:
            summary_file: Path to the generated summary file
        """
        print(f"Generating engagement summary from {self.log_file}")
        
        # Load behavior log data
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
        
        # Extract metadata
        metadata = log_data.get("metadata", {})
        session_id = metadata.get("session_id", "unknown")
        
        # Process entries
        entries = log_data.get("entries", [])
        
        # Calculate statistics per student
        student_stats = self._calculate_student_stats(entries)
        
        # Calculate overall class statistics
        class_stats = self._calculate_class_stats(student_stats, entries)
        
        # Create summary data
        summary_data = {
            "metadata": {
                "session_id": session_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "log_file": self.log_file
            },
            "class_summary": class_stats,
            "student_summaries": student_stats
        }
        
        # Save summary data
        summary_file = os.path.join(self.output_dir, f"engagement_summary_{session_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Engagement summary saved to {summary_file}")
        
        return summary_file
    
    def _calculate_student_stats(self, entries):
        """
        Calculate statistics for each student
        
        Args:
            entries: List of behavior log entries
        
        Returns:
            student_stats: Dictionary with statistics per student
        """
        # Group entries by student (track_id)
        student_entries = defaultdict(list)
        for entry in entries:
            track_id = entry.get("track_id")
            student_entries[track_id].append(entry)
        
        # Calculate statistics for each student
        student_stats = {}
        for track_id, student_data in student_entries.items():
            # Count behaviors
            behavior_counts = defaultdict(int)
            behavior_durations = defaultdict(float)
            last_behavior = None
            last_timestamp = 0
            
            for entry in sorted(student_data, key=lambda x: x.get("timestamp", 0)):
                timestamp = entry.get("timestamp", 0)
                behavior = entry.get("behavior", "unknown")
                
                # Count occurrences
                behavior_counts[behavior] += 1
                
                # Calculate durations
                if last_behavior is not None and last_behavior == behavior:
                    duration = timestamp - last_timestamp
                    behavior_durations[behavior] += duration
                
                last_behavior = behavior
                last_timestamp = timestamp
            
            # Get most recent snapshot
            latest_entry = max(student_data, key=lambda x: x.get("timestamp", 0))
            latest_snapshot = latest_entry.get("snapshot_path")
            
            # Calculate percentage of time spent on each behavior
            total_time = sum(behavior_durations.values())
            behavior_percentages = {
                behavior: (duration / total_time * 100 if total_time > 0 else 0)
                for behavior, duration in behavior_durations.items()
            }
            
            # Create student summary
            student_stats[track_id] = {
                "track_id": track_id,
                "total_detections": len(student_data),
                "behavior_counts": dict(behavior_counts),
                "behavior_durations": dict(behavior_durations),
                "behavior_percentages": behavior_percentages,
                "most_common_behavior": max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else "unknown",
                "latest_snapshot": latest_snapshot
            }
        
        return student_stats
    
    def _calculate_class_stats(self, student_stats, entries):
        """
        Calculate overall class statistics
        
        Args:
            student_stats: Dictionary with statistics per student
            entries: List of behavior log entries
        
        Returns:
            class_stats: Dictionary with overall class statistics
        """
        # Get total times
        all_behaviors = set()
        total_durations = defaultdict(float)
        
        for student_data in student_stats.values():
            behavior_durations = student_data.get("behavior_durations", {})
            all_behaviors.update(behavior_durations.keys())
            
            for behavior, duration in behavior_durations.items():
                total_durations[behavior] += duration
        
        # Calculate total time
        total_time = sum(total_durations.values())
        
        # Calculate class behavior percentages
        class_percentages = {
            behavior: (duration / total_time * 100 if total_time > 0 else 0)
            for behavior, duration in total_durations.items()
        }
        
        # Calculate engagement score (percentage of "attentive" behavior)
        engagement_score = class_percentages.get("attentive", 0)
        
        # Calculate time range
        timestamps = [entry.get("timestamp", 0) for entry in entries]
        session_duration = max(timestamps) - min(timestamps) if timestamps else 0
        
        # Create class summary
        class_stats = {
            "total_students": len(student_stats),
            "total_behaviors_detected": len(entries),
            "session_duration": session_duration,
            "behavior_durations": dict(total_durations),
            "behavior_percentages": class_percentages,
            "engagement_score": engagement_score,
            "most_common_behavior": max(total_durations.items(), key=lambda x: x[1])[0] if total_durations else "unknown"
        }
        
        return class_stats 