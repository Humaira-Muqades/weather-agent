"""
weather_agent.py
================
Your first AI agent, built from scratch (no agent framework).

What makes this an *agent* and not just a chatbot:
  1. The LLM is given TOOLS (get_weather, get_air_quality) and decides by itself
     which one to call, if any, and with which arguments.
  2. Our Python code runs the tool and hands the result back to the LLM.
  3. The LLM repeats this  think -> act -> observe  loop until it can answer.

Free pieces used:
  - Open-Meteo for weather and air-quality data (no API key needed for non-commercial use)
  - An LLM reached through the OpenAI-compatible chat API, from either:
      * Google Gemini  (free tier, needs an API key from Google AI Studio), or
      * Ollama         (runs a model on your own computer, no key, works offline)

Setup:
  pip install -r requirements.txt

  Create a file called  .env  in the same folder as this script containing:
      GEMINI_API_KEY=paste_your_key_here
      MODEL=gemini-3-flash-preview        (optional; run find_model.py to pick one)

  Run it:
      python weather_agent.py                          (interactive chat)
      python weather_agent.py "weather in Lahore for 3 days"   (one question)

To use Ollama instead of Gemini, put these lines in .env:
      PROVIDER=ollama
      MODEL=llama3.1:8b
"""

import json
import os
import re
import sys
from datetime import date, datetime

import requests
from openai import (OpenAI, APIConnectionError, APITimeoutError, AuthenticationError,
                    BadRequestError, NotFoundError, RateLimitError)

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads the .env file next to this script, if there is one
except ImportError:
    pass  # .env support is optional; normal environment variables also work


# ---------------------------------------------------------------------------
# 1. CONFIGURATION: which LLM the agent talks to
# ---------------------------------------------------------------------------
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Google renames and retires models often; if this one stops answering,
        # run  python find_model.py  and set MODEL=... in .env.
        "default_model": "gemini-3-flash-preview",
        "key_env": "GEMINI_API_KEY",
        "timeout": 60,  # seconds to wait for the model before giving up
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1:8b",
        "key_env": None,  # local model, no key needed
        "timeout": 180,  # local models on a laptop CPU can be slow, especially the first call
    },
}

PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()


def make_client(provider):
    """Create the LLM client and pick the model name."""
    if provider not in PROVIDERS:
        sys.exit(f"Unknown PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    if cfg["key_env"]:
        api_key = os.getenv(cfg["key_env"])
        if not api_key:
            sys.exit(
                f"Missing {cfg['key_env']}.\n"
                f"Create a file named .env next to this script containing:\n"
                f"    {cfg['key_env']}=paste_your_key_here"
            )
    else:
        api_key = "ollama"  # Ollama ignores the key, but the client needs some value

    model = os.getenv("MODEL", cfg["default_model"])
    # Without a timeout the library waits up to 10 minutes (and retries twice) on a bad connection,
    # which looks like the agent has frozen. Fail fast with a clear message instead.
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=cfg["timeout"], max_retries=1)
    return client, model


# ---------------------------------------------------------------------------
# 2. THE TOOLS: real Python code the agent is allowed to run
# ---------------------------------------------------------------------------
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo returns numeric WMO weather codes; this turns them into words.
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    97: "Heavy thunderstorm", 99: "Thunderstorm with heavy hail",
}


def describe(code):
    """Turn a WMO weather code into a readable description."""
    if code is None:
        return "Unknown"
    return WEATHER_CODES.get(int(code), f"Weather code {code}")


def _at(values, i):
    """Safely read values[i], returning None if it is missing."""
    if values and i < len(values):
        return values[i]
    return None


def find_location(city):
    """Look up a city's coordinates. Accepts 'Lahore' or 'Lahore, Pakistan'."""
    name, _, hint = city.partition(",")
    name, hint = name.strip(), hint.strip().lower()

    resp = requests.get(
        GEOCODE_URL,
        params={"name": name, "count": 5, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return None

    # If the user gave a country or province after the comma, prefer a matching result.
    if hint:
        for place in results:
            fields = (place.get("country", ""), place.get("country_code", ""), place.get("admin1", ""))
            if hint in (f.lower() for f in fields):
                return place
    return results[0]


def clamp_days(days):
    """Turn whatever the model sent into a whole number of days from 1 to 7."""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 1
    return max(1, min(days, 7))


def place_label(place):
    """Build 'Lahore, Punjab, Pakistan' without repeating names."""
    parts = []
    for p in (place.get("name"), place.get("admin1"), place.get("country")):
        if p and p not in parts:
            parts.append(p)
    return ", ".join(parts)


def not_found(city):
    return {"error": f"Could not find a place called '{city}'. "
                     f"Ask the user to check the spelling or add the country."}


def get_weather(city, days=1):
    """Current weather plus a daily forecast (1-7 days) for a city."""
    days = clamp_days(days)

    place = find_location(city)
    if place is None:
        return not_found(city)

    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})
    daily = data.get("daily", {})

    forecast = []
    for i, day_str in enumerate(daily.get("time", [])):
        forecast.append({
            "date": day_str,
            "day": datetime.strptime(day_str, "%Y-%m-%d").strftime("%A"),  # e.g. "Friday"
            "conditions": describe(_at(daily.get("weather_code"), i)),
            "max_temp_c": _at(daily.get("temperature_2m_max"), i),
            "min_temp_c": _at(daily.get("temperature_2m_min"), i),
            "chance_of_rain_pct": _at(daily.get("precipitation_probability_max"), i),
        })

    return {
        "location": place_label(place),
        "local_time": current.get("time"),
        "now": {
            "conditions": describe(current.get("weather_code")),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
        },
        "forecast": forecast,
    }


# US AQI bands (US EPA). Each entry is (upper limit of the band, name of the band).
AQI_CATEGORIES = [
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for sensitive groups"),
    (200, "Unhealthy"),
    (300, "Very unhealthy"),
    (float("inf"), "Hazardous"),
]


def aqi_category(aqi):
    """Name of the US AQI band a value falls in, e.g. 162 -> 'Unhealthy'."""
    if aqi is None:
        return "Unknown"
    for upper, name in AQI_CATEGORIES:
        if aqi <= upper:
            return name
    return "Hazardous"


def get_air_quality(city, days=1):
    """Current air quality plus the worst level expected on each day (1-7 days) for a city."""
    days = clamp_days(days)

    place = find_location(city)
    if place is None:
        return not_found(city)

    resp = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide,dust,uv_index",
            "hourly": "us_aqi,pm2_5",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    current = data.get("current", {})
    hourly = data.get("hourly", {})

    # The API gives hourly values; summarise each day by its worst hour.
    by_day = {}
    for i, stamp in enumerate(hourly.get("time", [])):
        day = by_day.setdefault(stamp[:10], {"aqi": [], "pm2_5": []})
        aqi, pm = _at(hourly.get("us_aqi"), i), _at(hourly.get("pm2_5"), i)
        if aqi is not None:
            day["aqi"].append(aqi)
        if pm is not None:
            day["pm2_5"].append(pm)

    forecast = []
    for day_str, values in by_day.items():
        worst = max(values["aqi"]) if values["aqi"] else None
        forecast.append({
            "date": day_str,
            "day": datetime.strptime(day_str, "%Y-%m-%d").strftime("%A"),
            "worst_us_aqi": worst,
            "worst_category": aqi_category(worst),
            "max_pm2_5_ugm3": max(values["pm2_5"]) if values["pm2_5"] else None,
        })

    now_aqi = current.get("us_aqi")
    return {
        "location": place_label(place),
        "local_time": current.get("time"),
        "now": {
            "us_aqi": now_aqi,
            "category": aqi_category(now_aqi),
            "pm2_5_ugm3": current.get("pm2_5"),
            "pm10_ugm3": current.get("pm10"),
            "ozone_ugm3": current.get("ozone"),
            "nitrogen_dioxide_ugm3": current.get("nitrogen_dioxide"),
            "dust_ugm3": current.get("dust"),
            "uv_index": current.get("uv_index"),
        },
        "forecast": forecast,
        "scale": "US AQI: 0-50 Good, 51-100 Moderate, 101-150 Unhealthy for sensitive groups, "
                 "151-200 Unhealthy, 201-300 Very unhealthy, 301+ Hazardous",
    }


# ---------------------------------------------------------------------------
# 3. TOOL DESCRIPTIONS: what the LLM reads to decide which tool to call, and how
# ---------------------------------------------------------------------------
# Both tools take the same arguments, so the schema is shared.
LOCATION_PARAMETERS = {
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name, optionally with country, e.g. 'Lahore' or 'Paris, France'.",
        },
        "days": {
            "type": "integer",
            "description": "Number of forecast days from 1 to 7, counting today as day 1. "
                           "Use 1 if the user only asks about today.",
        },
    },
    "required": ["city"],
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather and a daily forecast for a city: temperature, "
                           "feels-like, humidity, wind, conditions and chance of rain. "
                           "Not for air pollution; use get_air_quality for that.",
            "parameters": LOCATION_PARAMETERS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "Get current air quality and a daily air-quality forecast for a city: "
                           "US AQI with its category, PM2.5, PM10, ozone, nitrogen dioxide, dust "
                           "and UV index. Use for questions about air quality, pollution, smog, "
                           "haze, dust, masks, or whether the air is safe for outdoor activity.",
            "parameters": LOCATION_PARAMETERS,
        },
    },
]

# Maps the tool name the LLM uses to the real Python function.
TOOL_FUNCTIONS = {"get_weather": get_weather, "get_air_quality": get_air_quality}


def call_tool(name, arguments_json):
    """Run the tool the LLM asked for. Errors are returned to the LLM so it can recover."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return {"error": f"Tool arguments were not valid JSON: {arguments_json}"}
    try:
        return func(**args)
    except requests.RequestException as e:
        return {"error": f"Data service request failed: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# 4. THE AGENT LOOP: think -> act -> observe, until the LLM gives a final answer
# ---------------------------------------------------------------------------
def build_system_prompt():
    """The agent's instructions. Today's date is included so it can work out 'Friday' or 'tomorrow'."""
    today = date.today()
    return f"""You are a helpful weather and air-quality assistant.
Today is {today.strftime('%A')}, {today.isoformat()}.

Choosing tools:
- Weather, temperature, rain, wind or forecasts -> call get_weather.
- Air quality, pollution, smog, haze, dust or masks -> call get_air_quality.
- "Is it a good day for a run / walk / outdoor plans?" depends on both -> call both.
- Comparing several cities -> call the tool once per city.
- Never guess or invent data. If a tool returns an error, explain it plainly.

Dates:
- If the user names a day ("tomorrow", "Friday"), set `days` so the forecast reaches that day
  (today counts as day 1), then report that day using the "day" field in the results.
- The tools can only see 7 days ahead; say so if the user asks about a later day.
- If the user does not name a place, ask which city they mean.

Answer:
- Start straight with the report, with no preamble like "OK" or "Sure".
- Weather: location and local time; current temperature, feels-like, humidity and wind;
  the requested days including chance of rain.
- Air quality: the US AQI number with its category, the main pollutant (usually PM2.5),
  and the worst level expected on the requested days.
- End with one practical tip (umbrella, water, a mask, or the best time to go outside).
- Use °C and km/h. Keep it under 150 words unless the user asks for more."""

# Some local models (e.g. qwen3) print their reasoning inside <think> tags; hide it.
THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


def run_agent(client, model, messages, max_steps=5, verbose=True):
    """Keep calling the LLM, running any tools it asks for, until it answers in plain text."""
    for _ in range(max_steps):
        # THINK: send the whole conversation plus the tool list to the LLM.
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        message = response.choices[0].message

        # Record the LLM's turn. Tool calls are kept exactly as returned, because some
        # providers (e.g. Gemini) attach fields that must be sent back unchanged.
        assistant_turn = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_turn["tool_calls"] = [c.model_dump(exclude_none=True) for c in message.tool_calls]
        messages.append(assistant_turn)

        # No tool requested means the LLM has its final answer.
        if not message.tool_calls:
            return THINK_TAGS.sub("", message.content or "").strip()

        # ACT + OBSERVE: run each requested tool and feed the result back.
        for call in message.tool_calls:
            fn = getattr(call, "function", None)
            if fn is None:
                result = {"error": "Only function tools are supported."}
            else:
                if verbose:
                    print(f"  [agent calls tool] {fn.name}({fn.arguments})")
                result = call_tool(fn.name, fn.arguments)
                if verbose and "error" in result:
                    print(f"  [tool error] {result['error']}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "Sorry, I could not finish that request. Please try rephrasing it."


def explain_error(e):
    """Turn common API errors into a plain-English hint."""
    # Gemini reports a bad key as "400 Bad Request" rather than the usual 401.
    if isinstance(e, AuthenticationError) or (
            isinstance(e, BadRequestError) and "api key" in str(e).lower()):
        return ("The API key was rejected. Check GEMINI_API_KEY in your .env file: "
                "it should appear only once, with no spaces or quotes.")
    if isinstance(e, NotFoundError):
        return ("The model name was not found. Set MODEL in your .env to a model "
                "listed by your provider (Google AI Studio, or `ollama list`).")
    if isinstance(e, RateLimitError):
        return "Free-tier rate limit reached. Wait a minute and try again."
    if isinstance(e, APITimeoutError):
        return ("The model took too long to answer. Your connection may have dropped: "
                "run `python check_connection.py` to test it, then try again.")
    if isinstance(e, APIConnectionError):
        if PROVIDER == "ollama":
            return "Could not reach Ollama. Make sure the Ollama app is running."
        return "Could not reach the API. Check your internet connection."
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
def main():
    client, model = make_client(PROVIDER)
    messages = [{"role": "system", "content": build_system_prompt()}]

    # One-question mode:  python weather_agent.py "weather in Lahore for 3 days"
    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        try:
            print(run_agent(client, model, messages))
        except Exception as e:
            sys.exit(f"[error] {explain_error(e)}")
        return

    # Interactive chat mode
    print(f"Weather agent ready ({PROVIDER} / {model}).")
    print("Ask about weather or air quality anywhere. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit", "bye"}:
            break
        if not user_input:
            continue

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            reply = run_agent(client, model, messages)
        except Exception as e:
            del messages[checkpoint:]  # drop the failed turn so the history stays clean
            print(f"\n[error] {explain_error(e)}")
            continue
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
