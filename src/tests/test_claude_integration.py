import unittest
import time
from puter_integration import analyze_behavior_with_puter, get_behavior_insights, is_working

class TestClaudeIntegration(unittest.TestCase):
    def test_connection(self):
        """Test basic connection to Claude via Puter API"""
        working = is_working()
        self.assertTrue(working, "Claude integration should be working")
        
    def test_behavior_analysis(self):
        """Test analyzing behavior with Claude via Puter"""
        result = analyze_behavior_with_puter("How can teachers improve classroom engagement?", mode="Basic")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)  # Response should have meaningful content
        self.assertFalse(result.startswith("Error:"), f"API call failed: {result}")
        
    def test_insights_generation(self):
        """Test generating insights from video data"""
        test_data = {
            "duration": 300,
            "num_students": 24,
            "active_students": 18,
            "passive_students": 4,
            "distracted_students": 2,
            "high_movement_areas": ["Front of class", "Near windows"],
            "low_movement_areas": ["Back corners"],
            "group_formations": ["Front-left cluster", "Center pairs"],
            "individual_activities": ["Reading", "Writing", "Hand-raising"]
        }
        
        insights = get_behavior_insights(test_data)
        self.assertIn("analysis", insights)
        self.assertIn("raw_data", insights)
        self.assertIn("timestamp", insights)
        
        # Check timestamp is recent
        self.assertGreater(insights["timestamp"], time.time() - 60)
        
        # Check raw data is preserved
        self.assertEqual(insights["raw_data"]["num_students"], 24)
        self.assertEqual(insights["raw_data"]["active_students"], 18)
        
        # Check analysis is meaningful
        self.assertGreater(len(insights["analysis"]), 100)
        self.assertFalse(insights["analysis"].startswith("Error:"))

if __name__ == "__main__":
    print("Running Claude integration tests via Puter.js...")
    unittest.main() 