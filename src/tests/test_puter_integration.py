import unittest
from puter_integration import analyze_behavior_with_puter, get_behavior_insights

class TestPuterIntegration(unittest.TestCase):
    def test_behavior_analysis(self):
        # Test data
        test_data = {
            "timestamp": "2024-04-20 10:00:00",
            "detected_actions": [
                {"action": "raising_hand", "count": 5},
                {"action": "looking_away", "count": 3},
                {"action": "talking", "count": 2}
            ],
            "engagement_metrics": {
                "active_participation": 0.6,
                "distraction_level": 0.3,
                "overall_engagement": 0.7
            },
            "classroom_environment": {
                "lighting": "good",
                "noise_level": "moderate",
                "class_size": 25
            }
        }
        
        # Test behavior analysis
        result = analyze_behavior_with_puter(test_data)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
        # Test insights generation
        insights = get_behavior_insights(test_data)
        self.assertIn("analysis", insights)
        self.assertIn("raw_data", insights)
        self.assertIn("timestamp", insights)

if __name__ == "__main__":
    unittest.main() 