from typing import Dict, List, Optional
import pandas as pd
from .travel_types import TravelSuggestion, Itinerary
import logging
import os
from datetime import datetime

# Optional imports with fallbacks
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class LogManager:
    """Manages logging for the travel planner application"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"travel_planner_{timestamp}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger("TravelPlanner")
        self.logger.info("=== Starting New Travel Planning Session ===")
    
    def log_api_request(self, endpoint: str, payload: Dict):
        """Log API request details"""
        self.logger.info(f"\n=== API Request to {endpoint} ===")
        self.logger.info(f"Payload: {payload}")
    
    def log_api_response(self, endpoint: str, response: Dict):
        """Log API response details"""
        self.logger.info(f"\n=== API Response from {endpoint} ===")
        self.logger.info(f"Response: {response}")
    
    def log_error(self, error: Exception, context: str = None):
        """Log error details"""
        self.logger.error(f"\n=== Error in {context or 'Unknown Context'} ===")
        self.logger.error(f"Error: {str(error)}")
        self.logger.error(f"Stack trace:", exc_info=True)
    
    def log_info(self, message: str, data: Optional[Dict] = None):
        """Log informational message"""
        self.logger.info(f"\n{message}")
        if data:
            self.logger.info(f"Data: {data}")
    
    def log_warning(self, message: str, data: Optional[Dict] = None):
        """Log warning message"""
        self.logger.warning(f"\n{message}")
        if data:
            self.logger.warning(f"Data: {data}")
    
    def log_debug(self, message: str, data: Optional[Dict] = None):
        """Log debug message"""
        self.logger.debug(f"\n{message}")
        if data:
            self.logger.debug(f"Data: {data}")

# Create a global logger instance
logger = LogManager()

class TravelUtils:
    """Utility functions for travel-related operations"""
    
    @staticmethod
    def create_destination_map(suggestions: List[TravelSuggestion]):
        """Create interactive map with destination markers"""
        if not FOLIUM_AVAILABLE:
            return {"error": "Map visualization not available - folium not installed"}
        
        import folium
        # Default center (can be improved with geocoding)
        m = folium.Map(location=[20.0, 0.0], zoom_start=2)
        
        # Sample coordinates for demonstration (in real app, use geocoding API)
        sample_coords = {
            "Paris, France": [48.8566, 2.3522],
            "Tokyo, Japan": [35.6762, 139.6503],
            "New York, USA": [40.7128, -74.0060],
            "London, UK": [51.5074, -0.1278],
            "Sydney, Australia": [-33.8688, 151.2093]
        }
        
        for suggestion in suggestions:
            # Try to get coordinates for the destination
            coords = sample_coords.get(suggestion.destination, [0, 0])
            
            # Create popup content
            popup_content = f"""
            <b>{suggestion.destination}</b><br>
            {suggestion.description[:100]}...<br>
            <b>Best time:</b> {suggestion.best_time_to_visit}<br>
            <b>Budget:</b> {suggestion.estimated_budget}
            """
            
            folium.Marker(
                coords,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=suggestion.destination,
                icon=folium.Icon(color='blue', icon='plane')
            ).add_to(m)
        
        return m
    
    @staticmethod
    def create_budget_chart(itinerary: Itinerary) -> dict:
        """Create budget breakdown chart data"""
        if not itinerary.daily_plans:
            return {"error": "No daily plans available"}
        
        # Extract cost data from daily plans
        daily_costs = []
        for plan in itinerary.daily_plans:
            cost_str = plan.get("estimated_cost", "0")
            # Simple parsing (in real app, use better cost extraction)
            try:
                cost = float(cost_str.replace("$", "").replace("USD", "").split("-")[0])
            except:
                cost = 50.0  # Default cost
            
            daily_costs.append({
                "Day": f"Day {plan.get('day', 1)}",
                "Cost": cost
            })
        
        df = pd.DataFrame(daily_costs)
        
        return {
            "chart_data": df,
            "total_cost": sum([item["Cost"] for item in daily_costs])
        }
    
    @staticmethod
    def format_itinerary_for_display(itinerary: Itinerary) -> Dict:
        """Format itinerary for better display"""
        formatted_daily_plans = []
        
        for plan in itinerary.daily_plans:
            formatted_plan = {
                "day": plan.get("day", 1),
                "date": plan.get("date", f"Day {plan.get('day', 1)}"),
                "schedule": {
                    "Morning": plan.get("morning", "Free time"),
                    "Afternoon": plan.get("afternoon", "Free time"),
                    "Evening": plan.get("evening", "Free time")
                },
                "meals": plan.get("meals", []),
                "estimated_cost": plan.get("estimated_cost", "Not specified")
            }
            formatted_daily_plans.append(formatted_plan)
        
        return {
            "destination": itinerary.destination,
            "duration": f"{itinerary.total_days} days",
            "budget": itinerary.total_budget or "Not specified",
            "daily_plans": formatted_daily_plans,
            "accommodation": itinerary.accommodation_details,
            "transportation": itinerary.transportation_details,
            "packing_list": itinerary.packing_list,
            "emergency_contacts": itinerary.emergency_contacts,
            "important_notes": itinerary.important_notes
        }
    
    @staticmethod
    def get_travel_tips_by_region(destination: str) -> List[str]:
        """Get region-specific travel tips"""
        # Sample tips (in real app, use comprehensive database)
        regional_tips = {
            "europe": [
                "Check visa requirements for Schengen area",
                "Public transport is excellent in most cities",
                "Tipping is not mandatory but appreciated",
                "Many places close on Sundays"
            ],
            "asia": [
                "Respect local customs and dress codes",
                "Street food is generally safe and delicious",
                "Remove shoes when entering homes or temples",
                "Bargaining is common in markets"
            ],
            "americas": [
                "Tipping is expected in restaurants (15-20%)",
                "Distances can be vast - plan accordingly",
                "Check vaccination requirements",
                "Emergency services: 911"
            ],
            "africa": [
                "Check health requirements and vaccinations",
                "Safari bookings should be made in advance",
                "Carry cash as cards may not be widely accepted",
                "Respect wildlife and follow guide instructions"
            ],
            "oceania": [
                "Sun protection is essential",
                "Quarantine laws are strict",
                "Outdoor activities are year-round",
                "Indigenous culture should be respected"
            ]
        }
        
        # Simple region detection (can be improved)
        destination_lower = destination.lower()
        
        for region, tips in regional_tips.items():
            if any(country in destination_lower for country in ["france", "italy", "spain", "germany", "uk"]):
                return regional_tips["europe"]
            elif any(country in destination_lower for country in ["japan", "china", "thailand", "india", "korea"]):
                return regional_tips["asia"]
            elif any(country in destination_lower for country in ["usa", "canada", "brazil", "mexico", "argentina"]):
                return regional_tips["americas"]
            elif any(country in destination_lower for country in ["south africa", "kenya", "egypt", "morocco"]):
                return regional_tips["africa"]
            elif any(country in destination_lower for country in ["australia", "new zealand", "fiji"]):
                return regional_tips["oceania"]
        
        # Default general tips
        return [
            "Research local customs and etiquette",
            "Keep copies of important documents",
            "Check weather conditions before departure",
            "Learn basic phrases in local language",
            "Stay aware of your surroundings"
        ]
    
    @staticmethod
    def validate_travel_preferences(preferences: dict) -> Dict[str, bool]:
        """Validate travel preferences input"""
        validation_results = {
            "budget_valid": True,
            "dates_valid": True,
            "group_size_valid": True,
            "preferences_complete": True
        }
        
        # Validate budget
        budget = preferences.get("budget_range")
        if budget and not any(keyword in budget.lower() for keyword in ["low", "medium", "high", "luxury", "$", "budget"]):
            validation_results["budget_valid"] = False
        
        # Validate group size
        group_size = preferences.get("group_size", 1)
        if not isinstance(group_size, int) or group_size < 1 or group_size > 20:
            validation_results["group_size_valid"] = False
        
        # Check completeness
        required_fields = ["budget_range", "travel_style", "interests"]
        missing_fields = [field for field in required_fields if not preferences.get(field)]
        if missing_fields:
            validation_results["preferences_complete"] = False
            validation_results["missing_fields"] = missing_fields
        
        return validation_results
    
    @staticmethod
    def get_travel_style_descriptions() -> Dict[str, str]:
        """Get descriptions for different travel styles"""
        return {
            "adventure": "Thrilling activities, outdoor experiences, and off-the-beaten-path destinations",
            "cultural": "Museums, historical sites, local traditions, and authentic experiences",
            "relaxation": "Beaches, spas, resorts, and peaceful environments",
            "business": "Professional accommodations, meeting facilities, and efficient transportation",
            "budget": "Cost-effective options, hostels, local transport, and free activities",
            "luxury": "High-end accommodations, fine dining, and premium experiences",
            "family": "Kid-friendly activities, safe environments, and family accommodations",
            "romantic": "Intimate settings, couples activities, and romantic dining"
        } 