"""Travel tool — geocoding via Nominatim (geopy), routing via OSRM."""

import os
import requests
import logging
from geopy.geocoders import Nominatim

logger = logging.getLogger("v2a_demo")

# Required by Nominatim usage policy; requests without it get 403 Forbidden
HAILO_V2A_USER_AGENT = "HailoVoiceAssistant/1.0"
_geocoder = Nominatim(user_agent=HAILO_V2A_USER_AGENT, timeout=60)

# "home"/"work" are resolved from these addresses (set by the user), not geocoded literally.
HOME_ADDRESS = os.getenv("HOME_ADDRESS")
WORK_ADDRESS = os.getenv("WORK_ADDRESS")

# All the ways the LLM (or a user) might spell "here"/"current_location".
_CURRENT_LOCATION_ALIASES = {"current_location", "current location", "here"}

TOOL_PROMPT = (
    "Extract parameters from the user's travel time request as a JSON object.\n"
    "You MUST output ALL 3 fields in every response.\n"
    "\n"
    'CRITICAL RULE: If the user says "from home" or "home to", origin MUST be "home". NEVER use "current_location" when the user says "home".\n'
    "\n"
    "Parameters:\n"
    '- "origin": Starting location. Map: home/house/my place -> "home", work/office -> "work", here/current location -> "current_location". ONLY use "current_location" when the user does NOT mention any starting place. Default: "current_location".\n'
    '- "destination" (required): Destination place name. Preserve exact place names.\n'
    '- "mode": "driving", "walking", or "cycling". Map: car/drive -> "driving", walk/on foot -> "walking", bike/bicycle/cycling -> "cycling". Default: "driving".\n'
    "\n"
    "Examples:\n"
    '"How many minutes from home to Azrieli Mall?" -> {"origin": "home", "destination": "Azrieli Mall", "mode": "driving"}\n'
    '"How long from home to the airport?" -> {"origin": "home", "destination": "the airport", "mode": "driving"}\n'
    '"Travel time from home to work?" -> {"origin": "home", "destination": "work", "mode": "driving"}\n'
    '"How long to drive to the mall?" -> {"origin": "current_location", "destination": "mall", "mode": "driving"}\n'
    '"How long will it take me to get to Central Bus Station?" -> {"origin": "current_location", "destination": "Central Bus Station", "mode": "driving"}\n'
    '"How long to walk from Rothschild to the beach?" -> {"origin": "Rothschild", "destination": "the beach", "mode": "walking"}\n'
    '"Drive time from Azrieli Mall to the airport?" -> {"origin": "Azrieli Mall", "destination": "the airport", "mode": "driving"}\n'
    '"How long to cycle from home to the park?" -> {"origin": "home", "destination": "the park", "mode": "cycling"}\n'
    "\n"
    "Output ONLY the JSON object, nothing else."
)

TOOL_DESCRIPTIONS = [
    "How long does it take to get from one place to another?",
    "What is the travel time or ETA between two locations?",
    "How many minutes will the journey take?",
    "How fast can I get to a destination?",
    "Travel time questions for driving, walking, cycling, or public transport.",
    "Commute time from home to work or any other place.",
    "Time to reach a destination by car, bus, train, bike, or on foot.",
    "Estimated arrival time for a trip starting now, today, or tomorrow.",
    "Questions about journey duration or route time between an origin and a destination.",
]

# Maps user-facing mode names to FOSSGIS OSRM base URLs (separate instances per profile)
MODE_TO_OSRM_URL = {
    "driving": "https://routing.openstreetmap.de/routed-car/route/v1/driving",
    "walking": "https://routing.openstreetmap.de/routed-foot/route/v1/foot",
    "cycling": "https://routing.openstreetmap.de/routed-bike/route/v1/bicycle",
}

def _geocode(location: str) -> tuple[float, float] | None:
    """Geocode a location name to (longitude, latitude) using Nominatim."""
    try:
        result = _geocoder.geocode(location)
        if result:
            return (result.longitude, result.latitude)
    except Exception as e:
        logger.error(f"Geocoding failed for '{location}': {e}")
    return None


def _geocode_ip_location() -> tuple[float, float] | None:
    """Best-effort current location via IP geolocation (no GPS on this device)."""
    try:
        response = requests.get("http://ip-api.com/json/", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return (data["lon"], data["lat"])
    except Exception as e:
        logger.error(f"IP geolocation failed: {e}")
    return None


def _resolve_location(location: str) -> tuple[float, float] | None:
    """Resolve a location string to (longitude, latitude).

    Handles the "home"/"work"/"current_location" aliases the LLM extracts
    (see TOOL_PROMPT) before falling back to geocoding literal place names.
    Passing those aliases straight to Nominatim would never resolve, since
    they aren't real place names.
    """
    key = location.lower().strip()

    if key == "home":
        return _geocode(HOME_ADDRESS) if HOME_ADDRESS else None
    if key == "work":
        return _geocode(WORK_ADDRESS) if WORK_ADDRESS else None
    if key in _CURRENT_LOCATION_ALIASES:
        return _geocode_ip_location()

    return _geocode(location)


def _unresolved_location_message(location: str) -> str:
    """User-facing error for a location that couldn't be resolved."""
    key = location.lower().strip()
    if key == "home":
        return "I don't have your home address set. Please set the HOME_ADDRESS environment variable."
    if key == "work":
        return "I don't have your work address set. Please set the WORK_ADDRESS environment variable."
    if key in _CURRENT_LOCATION_ALIASES:
        return "I couldn't determine your current location."
    return f"I couldn't find {location}."


def _display_name(location: str) -> str:
    """Human-friendly name for a location when used in the spoken response."""
    if location.lower().strip() in _CURRENT_LOCATION_ALIASES:
        return "your current location"
    return location


def _format_duration(seconds: int) -> str:
    """Format seconds into a TTS-friendly duration string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60

    h = f"{hours} {'hour' if hours == 1 else 'hours'}"
    m = f"{minutes} {'minute' if minutes == 1 else 'minutes'}"
    if hours and minutes:
        return f"{h} and {m}"
    if hours:
        return h
    if minutes:
        return m
    return "less than a minute"


def _format_distance(meters: float) -> str:
    """Format meters into a TTS-friendly distance string."""
    if meters < 1000:
        m = round(meters)
        return f"{m} {'meter' if m == 1 else 'meters'}"
    km = round(meters / 1000)
    return f"{km} {'kilometer' if km == 1 else 'kilometers'}"


def get_travel_time(origin: str, destination: str, mode: str = "driving") -> str:
    """Get travel time and distance between two locations."""
    mode = mode.lower().strip()

    if mode not in MODE_TO_OSRM_URL:
        return f"Unknown travel mode: {mode}. You can ask for driving, walking, or cycling."

    orig_geocode = _resolve_location(origin)
    dest_geocode = _resolve_location(destination)

    if not orig_geocode:
        return _unresolved_location_message(origin)
    if not dest_geocode:
        return _unresolved_location_message(destination)

    base_url = MODE_TO_OSRM_URL[mode]
    url = (
        f"{base_url}/"
        f"{orig_geocode[0]},{orig_geocode[1]};{dest_geocode[0]},{dest_geocode[1]}"
    )

    try:
        request = requests.get(url, params={"overview": "false"}, timeout=60)
        request.raise_for_status()
        data = request.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return f"There is no {mode} route between {origin} and {destination}."

        route = data["routes"][0]
        duration = _format_duration(int(route["duration"]))
        distance = _format_distance(route["distance"])
        return (
            f"{mode.capitalize()} from {_display_name(origin)} to {_display_name(destination)} "
            f"takes about {duration}, covering {distance}."
        )

    except Exception as e:
        logger.error(f"Routing API error for {origin} -> {destination} ({mode}): {e}")
        return f"I couldn't get the travel time between {origin} and {destination}."
