from typing import List, Dict, Any
from travel_bot.models import TravelAgent, ProcessedInput, TravelPreferences, TravelSuggestion, Itinerary, AIModel
from travel_bot.models.ai_models import ModelFactory
import json

class SmartTravelAgent(TravelAgent):
    """Intelligent travel agent that creates suggestions and itineraries"""
    
    def __init__(self, model_type: str = "devstral"):
        self.ai_model: AIModel = ModelFactory.create_model(model_type)
    
    async def create_suggestions(self, 
                               processed_input: ProcessedInput,
                               preferences: TravelPreferences) -> List[TravelSuggestion]:
        """Create travel suggestions based on input and preferences"""
        
        # Create context from input and preferences
        context = self._build_context(processed_input, preferences)
        
        prompt = f"""
        Based on the following travel input and preferences, suggest 3-5 travel destinations with detailed information:

        Input Analysis: {processed_input.content}
        Extracted Entities: {processed_input.extracted_entities}
        
        User Preferences:
        - Budget: {preferences.budget_range or 'Not specified'}
        - Travel Style: {preferences.travel_style or 'Not specified'}
        - Interests: {', '.join(preferences.interests or [])}
        - Group Size: {preferences.group_size}
        - Language: {preferences.language_preference}
        
        For each destination, provide:
        1. Destination name
        2. Description (2-3 sentences)
        3. Best time to visit
        4. Estimated budget range
        5. Recommended duration
        6. Top 5 activities
        7. Accommodation suggestions
        8. Transportation options
        9. Local tips
        10. Weather information
        11. Safety information
        
        Format the response as a JSON array with this structure for each destination:
        {{
            "destination": "City, Country",
            "description": "...",
            "best_time_to_visit": "...",
            "estimated_budget": "...",
            "duration": "...",
            "activities": ["activity1", "activity2", ...],
            "accommodation_suggestions": ["hotel1", "hotel2", ...],
            "transportation": ["option1", "option2", ...],
            "local_tips": ["tip1", "tip2", ...],
            "weather_info": "...",
            "safety_info": "..."
        }}
        """
        
        response = await self.ai_model.generate_response(prompt, context)
        
        try:
            # Parse JSON response
            suggestions_data = json.loads(response)
            suggestions = []
            
            for data in suggestions_data:
                suggestion = TravelSuggestion(
                    destination=data.get("destination", "Unknown"),
                    description=data.get("description", ""),
                    best_time_to_visit=data.get("best_time_to_visit", ""),
                    estimated_budget=data.get("estimated_budget", ""),
                    duration=data.get("duration", ""),
                    activities=data.get("activities", []),
                    accommodation_suggestions=data.get("accommodation_suggestions", []),
                    transportation=data.get("transportation", []),
                    local_tips=data.get("local_tips", []),
                    weather_info=data.get("weather_info"),
                    safety_info=data.get("safety_info")
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except json.JSONDecodeError:
            # Fallback: create a single suggestion from raw response
            return [TravelSuggestion(
                destination="AI Generated Suggestion",
                description=response[:200] + "...",
                best_time_to_visit="Please check seasonal information",
                estimated_budget="Varies based on preferences",
                duration="3-7 days recommended",
                activities=["Explore local attractions", "Try local cuisine"],
                accommodation_suggestions=["Hotels", "Vacation rentals"],
                transportation=["Local transport", "Walking"],
                local_tips=["Research local customs", "Learn basic phrases"],
                weather_info="Check current weather conditions",
                safety_info="Follow standard travel safety precautions"
            )]
    
    async def create_itinerary(self,
                             destination: str,
                             days: int,
                             preferences: TravelPreferences) -> Itinerary:
        """Create detailed itinerary for specific destination"""
        
        prompt = f"""
        Create a detailed {days}-day itinerary for {destination} with the following preferences:
        
        - Budget: {preferences.budget_range or 'Moderate'}
        - Travel Style: {preferences.travel_style or 'Balanced'}
        - Interests: {', '.join(preferences.interests or [])}
        - Group Size: {preferences.group_size}
        - Dietary Restrictions: {', '.join(preferences.dietary_restrictions or [])}
        - Accommodation Type: {preferences.accommodation_type or 'Hotel'}
        
        Provide a comprehensive itinerary including:
        1. Daily plans with activities, meals, and timing
        2. Accommodation recommendations with details
        3. Transportation between locations
        4. Emergency contacts
        5. Packing list
        6. Important notes and tips
        
        Format as JSON with this structure:
        {{
            "destination": "{destination}",
            "total_days": {days},
            "total_budget": "estimated budget",
            "daily_plans": [
                {{
                    "day": 1,
                    "date": "Day 1",
                    "morning": "activity details",
                    "afternoon": "activity details", 
                    "evening": "activity details",
                    "meals": ["breakfast location", "lunch location", "dinner location"],
                    "estimated_cost": "daily budget"
                }}
            ],
            "accommodation_details": [
                {{
                    "name": "hotel name",
                    "type": "hotel/airbnb/etc",
                    "location": "address",
                    "price_range": "cost per night",
                    "amenities": ["wifi", "breakfast", "etc"]
                }}
            ],
            "transportation_details": [
                {{
                    "type": "flight/train/bus/car",
                    "details": "specific information",
                    "cost": "estimated cost"
                }}
            ],
            "emergency_contacts": [
                {{
                    "service": "Police/Hospital/Embassy",
                    "number": "phone number"
                }}
            ],
            "packing_list": ["item1", "item2", "item3"],
            "important_notes": ["note1", "note2", "note3"]
        }}
        """
        
        response = await self.ai_model.generate_response(prompt)
        
        try:
            itinerary_data = json.loads(response)
            
            return Itinerary(
                destination=itinerary_data.get("destination", destination),
                total_days=itinerary_data.get("total_days", days),
                total_budget=itinerary_data.get("total_budget"),
                daily_plans=itinerary_data.get("daily_plans", []),
                accommodation_details=itinerary_data.get("accommodation_details", []),
                transportation_details=itinerary_data.get("transportation_details", []),
                emergency_contacts=itinerary_data.get("emergency_contacts", []),
                packing_list=itinerary_data.get("packing_list", []),
                important_notes=itinerary_data.get("important_notes", [])
            )
            
        except json.JSONDecodeError:
            # Fallback itinerary
            daily_plans = []
            for day in range(1, days + 1):
                daily_plans.append({
                    "day": day,
                    "date": f"Day {day}",
                    "morning": "Explore local attractions",
                    "afternoon": "Cultural activities or sightseeing",
                    "evening": "Dinner and relaxation",
                    "meals": ["Local breakfast", "Local lunch", "Local dinner"],
                    "estimated_cost": "50-100 USD"
                })
            
            return Itinerary(
                destination=destination,
                total_days=days,
                total_budget="Varies based on preferences",
                daily_plans=daily_plans,
                accommodation_details=[{
                    "name": "Recommended accommodation",
                    "type": "Hotel",
                    "location": "City center",
                    "price_range": "100-200 USD/night",
                    "amenities": ["WiFi", "Breakfast", "AC"]
                }],
                transportation_details=[{
                    "type": "Local transport",
                    "details": "Use public transportation or walking",
                    "cost": "10-20 USD/day"
                }],
                emergency_contacts=[{
                    "service": "Emergency Services",
                    "number": "Check local emergency numbers"
                }],
                packing_list=["Comfortable shoes", "Weather-appropriate clothing", "Travel documents", "Camera", "Medications"],
                important_notes=["Check visa requirements", "Research local customs", "Keep copies of important documents"]
            )
    
    def _build_context(self, processed_input: ProcessedInput, preferences: TravelPreferences) -> str:
        """Build context string for AI model"""
        context = f"""
        You are a professional travel advisor with expertise in creating personalized travel experiences.
        You have access to comprehensive knowledge about destinations worldwide, including:
        - Cultural attractions and activities
        - Accommodation options across all budgets
        - Transportation methods and costs
        - Local customs and etiquette
        - Safety considerations
        - Weather patterns
        - Budget planning
        
        Always provide practical, actionable advice while being sensitive to cultural differences.
        Consider the user's language preference: {preferences.language_preference}
        """
        return context