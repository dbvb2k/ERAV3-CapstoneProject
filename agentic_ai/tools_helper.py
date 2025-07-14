import logging
import difflib
import json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_city_code(location_string: str, city_map: dict) -> tuple:
    """
    Find IATA code for a location using multiple matching strategies.
    Returns tuple of (iata_code, matching_method, matched_city)
    """
    if not location_string:
        logger.warning("Empty location string provided")
        return "BOM", "default", "Mumbai"  # Default to Mumbai
    
    # Method 1: Direct match
    if location_string in city_map:
        logger.info(f"Direct match found for '{location_string}'")
        return city_map[location_string], "direct", location_string
    
    # Method 2: Check if any city name is contained in the input string
    for city, code in city_map.items():
        # Case insensitive search
        if city.lower() in location_string.lower():
            logger.info(f"Substring match found: '{city}' in '{location_string}'")
            return code, "substring", city
    
    # Method 3: Fuzzy matching
    # Extract all words from the location string
    words = location_string.lower().split()
    best_match = 0.0
    best_ratio = 0.8  # Threshold for a good match
    best_city = None
    
    for word in words:
        if len(word) < 3:  # Skip short words
            continue
            
        for city in city_map.keys():
            ratio = difflib.SequenceMatcher(None, word.lower(), city.lower()).ratio()
            if ratio > best_ratio and ratio > best_match:
                best_match = ratio
                best_ratio = ratio
                best_city = city
    
    if best_city:
        logger.info(f"Fuzzy match found: '{best_city}' (ratio: {best_ratio:.2f}) for '{location_string}'")
        return city_map[best_city], "fuzzy", best_city
    
    # No match found, return default
    logger.warning(f"No city match found for '{location_string}', using default")
    return "BOM", "default", "Mumbai"  # Default to Mumbai

# Add this new function after the existing JSON parsing code (around line 432)
def repair_and_parse_json(content, location, duration, preferences):
    """Attempt to repair and parse malformed JSON from LLM output."""
    import re
    
    logger.info("Attempting to repair and parse potentially malformed JSON")
    
    # Step 1: Clean up the content
    # Remove markdown code block markers if present
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```", "", content)
    
    # Step 2: Try multiple repair strategies
    try:
        # First attempt: Direct parsing (might work if JSON is valid)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.info(f"Initial JSON parsing failed: {str(e)}")
        
        # Second attempt: Fix missing commas between objects in arrays
        fixed_content = re.sub(r'}\s*{', '},{', content)
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            logger.info("Failed to parse after fixing missing commas between objects")
        
        # Third attempt: Fix trailing commas (common LLM error)
        fixed_content = re.sub(r',\s*}', '}', content)
        fixed_content = re.sub(r',\s*]', ']', fixed_content)
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            logger.info("Failed to parse after removing trailing commas")
        
        # Fourth attempt: Balance braces and brackets
        def balance_json_string(json_str):
            # Count opening and closing braces/brackets
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            # Add missing closing braces/brackets
            json_str = json_str.rstrip()
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)
                
            return json_str
        
        fixed_content = balance_json_string(content)
        try:
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            logger.info("Failed to parse after balancing braces")
        
        # Fifth attempt: Use regex to extract key components and build a new JSON object
        result = {"destination": location, "duration": duration}
        
        # Extract best time to visit
        best_time_match = re.search(r'"best_time_to_visit"\s*:\s*"([^"]+)"', content)
        if best_time_match:
            result["best_time_to_visit"] = best_time_match.group(1)
        else:
            result["best_time_to_visit"] = "Year-round"
        
        # Extract budget
        budget_match = re.search(r'"estimated_budget"\s*:\s*"([^"]+)"', content)
        if budget_match:
            result["estimated_budget"] = budget_match.group(1)
        else:
            result["estimated_budget"] = f"{preferences.get('budget_range', 'Moderate')} budget"
        
        # Extract daily plans - this is more complex
        result["daily_plans"] = []
        
        # Look for day patterns in the content
        day_pattern = r'"day"\s*:\s*(\d+)[^{]*?(?:"morning"\s*:\s*(?:"([^"]+)"|(\{[^}]+\}))[^{]*?"afternoon"\s*:\s*(?:"([^"]+)"|(\{[^}]+\}))[^{]*?"evening"\s*:\s*(?:"([^"]+)"|(\{[^}]+\})))'
        day_matches = re.findall(day_pattern, content)
        
        if day_matches:
            for match in day_matches:
                day_num, morning_str, morning_obj, afternoon_str, afternoon_obj, evening_str, evening_obj = match
                
                day_plan = {"day": int(day_num)}
                
                # Process morning
                if morning_str:
                    day_plan["morning"] = morning_str
                elif morning_obj:
                    # Try to extract activity from the object
                    activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', morning_obj)
                    if activity_match:
                        morning_activity = {
                            "activity": activity_match.group(1)
                        }
                        # Try to extract time
                        time_match = re.search(r'"time"\s*:\s*"([^"]+)"', morning_obj)
                        if time_match:
                            morning_activity["time"] = time_match.group(1)
                        # Try to extract cost
                        cost_match = re.search(r'"cost"\s*:\s*"([^"]+)"', morning_obj)
                        if cost_match:
                            morning_activity["cost"] = cost_match.group(1)
                            
                        day_plan["morning"] = morning_activity
                    else:
                        day_plan["morning"] = f"Explore {location}"
                else:
                    day_plan["morning"] = f"Explore {location}"
                
                # Process afternoon (similar to morning)
                if afternoon_str:
                    day_plan["afternoon"] = afternoon_str
                elif afternoon_obj:
                    activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', afternoon_obj)
                    if activity_match:
                        afternoon_activity = {
                            "activity": activity_match.group(1)
                        }
                        # Extract time and cost similar to morning
                        time_match = re.search(r'"time"\s*:\s*"([^"]+)"', afternoon_obj)
                        if time_match:
                            afternoon_activity["time"] = time_match.group(1)
                        cost_match = re.search(r'"cost"\s*:\s*"([^"]+)"', afternoon_obj)
                        if cost_match:
                            afternoon_activity["cost"] = cost_match.group(1)
                            
                        day_plan["afternoon"] = afternoon_activity
                    else:
                        day_plan["afternoon"] = "Explore local attractions"
                else:
                    day_plan["afternoon"] = "Explore local attractions"
                
                # Process evening (similar to morning)
                if evening_str:
                    day_plan["evening"] = evening_str
                elif evening_obj:
                    activity_match = re.search(r'"activity"\s*:\s*"([^"]+)"', evening_obj)
                    if activity_match:
                        evening_activity = {
                            "activity": activity_match.group(1)
                        }
                        # Extract time and cost similar to morning
                        time_match = re.search(r'"time"\s*:\s*"([^"]+)"', evening_obj)
                        if time_match:
                            evening_activity["time"] = time_match.group(1)
                        cost_match = re.search(r'"cost"\s*:\s*"([^"]+)"', evening_obj)
                        if cost_match:
                            evening_activity["cost"] = cost_match.group(1)
                            
                        day_plan["evening"] = evening_activity
                    else:
                        day_plan["evening"] = "Enjoy local cuisine"
                else:
                    day_plan["evening"] = "Enjoy local cuisine"
                
                result["daily_plans"].append(day_plan)
        
        # If no day plans were extracted, create default ones
        if not result["daily_plans"]:
            for day in range(1, min(duration + 1, 8)):
                result["daily_plans"].append({
                    "day": day,
                    "morning": f"Explore popular attractions in {location}",
                    "afternoon": "Enjoy local cuisine for lunch",
                    "evening": "Experience local nightlife and dinner"
                })
        
        return result
            
    except Exception as e:
        logger.error(f"All JSON repair attempts failed: {str(e)}")
        return None


def advanced_json_repair(content, location, duration):
    """
    Advanced JSON repair using character-by-character parsing and fixing.
    This handles common LLM JSON generation errors.
    """
    import re
    
    logger.info("Attempting advanced JSON repair")
    
    # Clean the content first
    content = re.sub(r"```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```", "", content)
    content = content.strip()
    
    # Find the JSON object (everything between the first { and the last })
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        logger.error("Cannot find valid JSON boundaries")
        return None
        
    # Extract the JSON part
    json_content = content[start_idx:end_idx+1]
    
    # Step 1: Fix missing commas between objects
    json_content = re.sub(r'}\s*{', '},{', json_content)
    
    # Step 2: Fix trailing commas
    json_content = re.sub(r',\s*}', '}', json_content)
    json_content = re.sub(r',\s*]', ']', json_content)
    
    # Step 3: Fix broken nested objects
    # This corrects common LLM errors like unclosed nested objects
    def fix_nested_json(json_text):
        stack = []
        fixed_json = ""
        i = 0
        
        # Process character by character
        while i < len(json_text):
            char = json_text[i]
            
            # Handle opening brackets
            if char in ['{', '[']:
                stack.append(char)
                fixed_json += char
            
            # Handle closing brackets
            elif char in ['}', ']']:
                if not stack:  # Extra closing bracket
                    i += 1
                    continue
                    
                opening = stack.pop()
                # Ensure bracket types match
                if (opening == '{' and char == '}') or (opening == '[' and char == ']'):
                    fixed_json += char
                else:
                    # Fix mismatched brackets
                    if opening == '{':
                        fixed_json += '}'
                    else:
                        fixed_json += ']'
                    i -= 1  # Reprocess this character
                
            # Handle string literals
            elif char == '"':
                fixed_json += char
                i += 1
                
                # Add all characters until the closing quote
                while i < len(json_text) and json_text[i] != '"':
                    # Handle escape sequences
                    if json_text[i] == '\\' and i + 1 < len(json_text):
                        fixed_json += json_text[i:i+2]
                        i += 2
                    else:
                        fixed_json += json_text[i]
                        i += 1
                
                # Add the closing quote
                if i < len(json_text):
                    fixed_json += json_text[i]
                else:
                    fixed_json += '"'  # Add missing quote
                    
            # Handle comma between values
            elif char == ',':
                fixed_json += char
                
                # Fix double commas
                while i + 1 < len(json_text) and json_text[i+1] == ',':
                    i += 1
                    
            # Handle everything else
            else:
                fixed_json += char
                
            i += 1
            
        # Close any unclosed brackets
        for bracket in reversed(stack):
            if bracket == '{':
                fixed_json += '}'
            else:
                fixed_json += ']'
                
        return fixed_json
    
    # Apply the character-by-character fixing
    json_content = fix_nested_json(json_content)
    
    try:
        # Try parsing the fixed JSON
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Advanced JSON repair failed: {e}")
        
        # Final attempt: Try to build structured JSON from scratch using regex
        daily_plans = []
        
        # Extract individual days with their activities
        day_pattern = r'"day"\s*:\s*(\d+)[^{]*?(?:"morning"\s*:\s*({[^}]+})[^{]*?"afternoon"\s*:\s*({[^}]+})[^{]*?"evening"\s*:\s*({[^}]+}))'
        day_matches = re.findall(day_pattern, json_content)
        
        if day_matches:
            for day_num, morning, afternoon, evening in day_matches:
                try:
                    day_plan = {"day": int(day_num)}
                    
                    # Extract activity data from each period
                    for period, data in [("morning", morning), ("afternoon", afternoon), ("evening", evening)]:
                        activity = re.search(r'"activity"\s*:\s*"([^"]+)"', data)
                        time = re.search(r'"time"\s*:\s*"([^"]+)"', data)
                        cost = re.search(r'"cost"\s*:\s*"([^"]+)"', data)
                        
                        day_plan[period] = {
                            "activity": activity.group(1) if activity else f"Explore {location}",
                            "time": time.group(1) if time else "Various times",
                            "cost": cost.group(1) if cost else "Variable"
                        }
                    
                    daily_plans.append(day_plan)
                except Exception as inner_e:
                    logger.error(f"Error processing day {day_num}: {inner_e}")
        
        # If we found at least some days, create a basic itinerary
        if daily_plans:
            return {
                "destination": location,
                "duration": duration,
                "best_time_to_visit": "Year-round with seasonal variations",
                "estimated_budget": "Moderate budget (estimate unavailable)",
                "daily_plans": daily_plans,
                "transportation_tips": "Local transportation available"
            }
        
        # If all else fails, return None
        return None
    

city_to_iata = {
    # Indian Cities & Synonyms
    "Mumbai": "BOM",
    "Bombay": "BOM",
    "Delhi": "DEL",
    "New Delhi": "DEL",
    "Bengaluru": "BLR",
    "Bangalore": "BLR",
    "Chennai": "MAA",
    "Madras": "MAA",
    "Kolkata": "CCU",
    "Calcutta": "CCU",
    "Hyderabad": "HYD",
    "Secunderabad": "HYD",
    "Pune": "PNQ",
    "Ahmedabad": "AMD",
    "Goa": "GOI",
    "Panaji": "GOI",
    "Jaipur": "JAI",
    "Pink City": "JAI",
    "Kochi": "COK",
    "Cochin": "COK",
    "Trivandrum": "TRV",
    "Thiruvananthapuram": "TRV",
    "Lucknow": "LKO",
    "Varanasi": "VNS",
    "Bhubaneswar": "BBI",
    "Indore": "IDR",
    "Nagpur": "NAG",
    "Visakhapatnam": "VTZ",
    "Vizag": "VTZ",
    "Coimbatore": "CJB",
    "Mangalore": "IXE",
    "Amritsar": "ATQ",
    "Patna": "PAT",
    "Raipur": "RPR",
    "Ranchi": "IXR",
    "Srinagar": "SXR",
    "Guwahati": "GAU",
    "Chandigarh": "IXC",
    "Aurangabad": "IXU",
    "Agra": "AGR",
    "Vadodara": "BDQ",
    "Surat": "STV",
    "Tiruchirappalli": "TRZ",
    "Trichy": "TRZ",
    "Jodhpur": "JDH",
    "Udaipur": "UDR",
    "Dehradun": "DED",
    "Leh": "IXL",
    "Shillong": "SHL",

    # International Cities & Synonyms
    "New York": "JFK",
    "NYC": "JFK",
    "John F Kennedy": "JFK",
    "London": "LHR",
    "Heathrow": "LHR",
    "Gatwick": "LGW",
    "Paris": "CDG",
    "Charles de Gaulle": "CDG",
    "Orly": "ORY",
    "Tokyo": "HND",
    "Haneda": "HND",
    "Narita": "NRT",
    "Dubai": "DXB",
    "Singapore": "SIN",
    "Hong Kong": "HKG",
    "Sydney": "SYD",
    "Los Angeles": "LAX",
    "LA": "LAX",
    "San Francisco": "SFO",
    "SF": "SFO",
    "Amsterdam": "AMS",
    "Frankfurt": "FRA",
    "Toronto": "YYZ",
    "Bangkok": "BKK",
    "Istanbul": "IST",
    "Abu Dhabi": "AUH",
    "Doha": "DOH",
    "Melbourne": "MEL",
    "Brisbane": "BNE",
    "Perth": "PER",
    "Cape Town": "CPT",
    "Johannesburg": "JNB",
    "Beijing": "PEK",
    "Shanghai": "PVG",
    "Seoul": "ICN",
    "Kuala Lumpur": "KUL",
    "Jakarta": "CGK",
    "Madrid": "MAD",
    "Barcelona": "BCN",
    "Rome": "FCO",
    "Milan": "MXP",
    "Zurich": "ZRH",
    "Geneva": "GVA",
    "Vienna": "VIE",
    "Munich": "MUC",
    "Berlin": "BER",
    "Brussels": "BRU",
    "Athens": "ATH",
    "Lisbon": "LIS",
    "Dublin": "DUB",
    "Prague": "PRG",
    "Budapest": "BUD",
    "Warsaw": "WAW",
    "Moscow": "SVO",
    "St Petersburg": "LED",
    "Stockholm": "ARN",
    "Copenhagen": "CPH",
    "Oslo": "OSL",
    "Helsinki": "HEL",
    "Vienna": "VIE",
    "Venice": "VCE",
    "Miami": "MIA",
    "Chicago": "ORD",
    "Boston": "BOS",
    "Washington DC": "IAD",
    "Seattle": "SEA",
    "Dallas": "DFW",
    "Houston": "IAH",
    "Atlanta": "ATL",
    "San Diego": "SAN",
    "Las Vegas": "LAS",
    "Orlando": "MCO",
    "Montreal": "YUL",
    "Vancouver": "YVR",
    "Calgary": "YYC",
    "Mexico City": "MEX",
    "Rio de Janeiro": "GIG",
    "Sao Paulo": "GRU",
    "Buenos Aires": "EZE",
    "Lima": "LIM",
    "Bogota": "BOG",
    "Santiago": "SCL",
    "Dubai": "DXB",
    "Abu Dhabi": "AUH",
    "Doha": "DOH",
    "Jeddah": "JED",
    "Riyadh": "RUH",
    "Tel Aviv": "TLV",
    "Cairo": "CAI",
    "Nairobi": "NBO",
    "Cape Town": "CPT",
    "Johannesburg": "JNB",
    "Dar es Salaam": "DAR",
    "Addis Ababa": "ADD"
}