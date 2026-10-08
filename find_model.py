"""
find_model.py
Tests each piece the agent needs, so a failure can be pinned to one place.
It never prints your API key.
"""
import os
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("1) Weather API (Open-Meteo)...")
try:
    t = time.time()
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 24.86, "longitude": 67.01, "current": "temperature_2m"},
        timeout=30,
    )
    print(f"   OK in {time.time() - t:.1f}s | Karachi now: {r.json().get('current', {}).get('temperature_2m')} C")
except Exception as e:
    print("   FAILED:", type(e).__name__, "-", e)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
    timeout=30,
    max_retries=0,
)

print("2) Gemini API: connect and list models...")
available = []
try:
    t = time.time()
    available = [m.id.removeprefix("models/") for m in client.models.list()]
    flash = [n for n in available if "flash" in n and "tts" not in n and "image" not in n]
    print(f"   OK in {time.time() - t:.1f}s | Flash models on your key: {', '.join(flash)}")
except Exception as e:
    print("   FAILED:", type(e).__name__, "-", e)

print("3) Gemini API: which models actually answer? (up to 30s each)")
candidates = [
    os.getenv("MODEL", "gemini-flash-latest"),
    "gemini-3.8-flash",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]
results = []
for model in dict.fromkeys(candidates):  # keeps order, drops duplicates
    if available and model not in available:
        print(f"   {model}: not offered to your key, skipped")
        continue
    t = time.time()
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just the word: ready"}],
        )
        secs = time.time() - t
        text = (reply.choices[0].message.content or "").strip()
        print(f"   {model}: OK in {secs:.1f}s -> {text[:30]!r}")
        results.append((secs, model))
    except Exception as e:
        print(f"   {model}: FAILED after {time.time() - t:.1f}s - {type(e).__name__}: {str(e)[:150]}")

if results:
    best = min(results)[1]
    print(f"\nFastest working model: {best}")
    print(f"Put this line in your .env file:  MODEL={best}")
else:
    print("\nNo model answered. Wait a few minutes and run this again.")
