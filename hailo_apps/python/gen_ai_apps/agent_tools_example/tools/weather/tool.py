"""
Real weather tool using Open-Meteo API.

Gets current temperature and weather data for any location worldwide.
No API key required!
"""

from __future__ import annotations

from typing import Any

from .api import get_current_temperature, get_weather_forecast

# Temperature unit configuration
TEMPERATURE_UNIT: str = "celsius"

name: str = "weather"

# User-facing description (shown in CLI tool list)
display_description: str = (
    "Get current weather and rain forecasts (supports future days) using the Open-Meteo API."
)

# LLM instruction description (includes warnings for model)
description: str = (
    "CRITICAL: Use this tool ONLY when the user explicitly asks about weather, temperature, or rain. "
    "If you don't know the location of the query, do not call this tool. Ask the user for the location."
    "Supported requests: current temperature, forecasts for future days, rain/precipitation queries. "
    "For dates: use the 'future_days' parameter (e.g., 'tomorrow' -> future_days=1, 'in 3 days' -> future_days=3, 'today' -> future_days=0). "
    "Set include_rain=true when the user asks about rain or precipitation.\n\n"
    "DEFAULT OPTION: If the user's request is not related to weather, or if you cannot determine the location, "
    "set 'default' to true. The tool will automatically generate an appropriate error message."
)

schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "Location in 'City' or 'City, Country' format. Required unless 'default' is used.",
        },
        "future_days": {
            "type": "integer",
            "description": (
                "Number of days in the future for forecast (0=today, 1=tomorrow, 2=in 2 days, etc.). "
                "Defaults to 0 if not specified."
            ),
        },
        "include_rain": {
            "type": "boolean",
            "description": (
                "If true, include precipitation (rain) totals and probability. "
                "Defaults to false if not specified."
            ),
        },
        "default": {
            "type": "boolean",
            "description": (
                "Set to true when the user's request is not related to weather or if you cannot determine the location. "
                "The tool will automatically generate an appropriate error message."
            ),
        },
    },
    "required": [],
}

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }
]


def _validate_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate input parameters."""
    # If default is provided, location is not required
    if payload.get("default") is True:
        return {"ok": True, "data": {"location": None, "future_days": 0, "include_rain": False}}

    try:
        from pydantic import BaseModel, Field

        class WeatherInput(BaseModel):
            location: str = Field(description="Location name")
            future_days: int = Field(
                default=0,
                description="Days in future for forecast (0=today, 1=tomorrow)",
                ge=0,
            )
            include_rain: bool = Field(
                default=False, description="Include precipitation data"
            )

        data = WeatherInput(**payload).model_dump()
        future_days = int(data.get("future_days", 0))
        if future_days < 0:
            future_days = 0
        data["future_days"] = future_days
        return {"ok": True, "data": data}
    except Exception:
        # Fallback validation without pydantic
        location = str(payload.get("location", "")).strip()
        future_days = payload.get("future_days", 0)
        try:
            future_days = int(future_days)
            if future_days < 0:
                future_days = 0
        except (ValueError, TypeError):
            future_days = 0
        include_rain = bool(payload.get("include_rain", False))
        if not location:
            return {"ok": False, "error": "Either 'location' or 'default' must be provided"}
        return {
            "ok": True,
            "data": {
                "location": location,
                "future_days": future_days,
                "include_rain": include_rain,
            },
        }


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Get weather data for a location, optionally with forecast and precipitation.

    Args:
        input_data: Dictionary with keys:
            - location: Location name (required unless default used)
            - future_days: Days in future for forecast (default: 0)
            - include_rain: Include precipitation data (default: False)
            - default: Message when request is not weather-related or location unknown

    Returns:
        Dictionary with 'ok' and weather data or 'error'.
    """
    # Check for default option first (user error - agent correctly used default)
    if input_data.get("default") is True:
        return {
            "ok": True,  # Agent used tool correctly (default option)
            "error": "This tool only handles weather queries and requires a location."
        }

    validated = _validate_input(input_data)
    if not validated.get("ok"):
        return validated

    data = validated["data"]
    location = data["location"]
    future_days = data.get("future_days", 0)
    include_rain = data.get("include_rain", False)

    try:
        if future_days > 0 or include_rain:
            result = get_weather_forecast(
                location=location,
                future_days=future_days,
                include_rain=include_rain,
                unit=TEMPERATURE_UNIT,
            )
        else:
            result = get_current_temperature(location, TEMPERATURE_UNIT)

        if result.startswith("Error:"):
            return {"ok": False, "error": result}

        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"Failed to fetch weather data: {str(e)}"}

