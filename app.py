"""
app.py
======
A web chat interface for the weather and air-quality agent, built with Gradio.

The agent itself lives in weather_agent.py. This file adds the interface:
  - every tool call the agent makes is shown as a collapsible panel, so visitors can
    watch it "think -> act -> observe" instead of only reading the final answer;
  - an interactive chart (from charts.py) follows each answer.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open  http://127.0.0.1:7860  in your browser.

Online, Render runs the same command (see render.yaml). It sets the PORT environment
variable, which tells this file to accept connections from the internet on that port.
"""

import json
import os

import gradio as gr

from charts import chart_for_call
from weather_agent import PROVIDER, build_system_prompt, explain_error, make_client, run_agent

client, MODEL = make_client(PROVIDER)

MAX_CHARTS = 3  # per answer, so a five-city comparison doesn't flood the chat

EXAMPLES = [
    "What's the air quality in Lahore today?",
    "Will it rain in Karachi on Friday?",
    "Is it a good day for a run in Islamabad?",
    "Weather in London, Canada for the next 3 days",
]


def _text(content):
    """Return a message's text, or "" for charts and other non-text messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _pretty(result_json):
    """Format a tool result for display."""
    try:
        return json.dumps(json.loads(result_json), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result_json)


def _tool_calls(new_messages):
    """Pair each tool call the agent made with its result, in order."""
    pairs, i = [], 0
    while i < len(new_messages):
        turn = new_messages[i]
        calls = turn.get("tool_calls") if turn.get("role") == "assistant" else None
        if not calls:
            i += 1
            continue
        results = new_messages[i + 1: i + 1 + len(calls)]
        pairs.extend(zip(calls, results))
        i += 1 + len(calls)
    return pairs


def chat(message, history):
    # 1. Rebuild the conversation for the model from the visible chat.
    #    Tool panels and charts are only for display, so they are skipped.
    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in history:
        if (m.get("metadata") or {}).get("title"):
            continue
        text = _text(m.get("content"))
        if text and m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": text})
    messages.append({"role": "user", "content": message})
    start = len(messages)

    # 2. Run exactly the same agent loop as the terminal version.
    try:
        reply = run_agent(client, MODEL, messages, verbose=False)
    except Exception as e:
        return f"⚠️ {explain_error(e)}"

    # 3. Show each tool call (arguments + result) as a collapsible panel, then the answer.
    shown, chart_calls = [], []
    for call, result in _tool_calls(messages[start:]):
        fn = call.get("function", {})
        content = result.get("content")
        shown.append(gr.ChatMessage(
            role="assistant",
            content=(
                f"**Arguments:** `{fn.get('arguments')}`\n\n"
                f"**Result:**\n```json\n{_pretty(content)}\n```"
            ),
            # "done" makes the panel start collapsed, so the answer stays in view.
            metadata={"title": f"🔧 Called {fn.get('name')}", "status": "done"},
        ))
        try:
            succeeded = "error" not in json.loads(content or "{}")
        except (TypeError, ValueError):
            succeeded = False
        key = (fn.get("name"), fn.get("arguments"))
        if succeeded and key not in chart_calls:
            chart_calls.append(key)
    shown.append(gr.ChatMessage(role="assistant", content=reply))

    # 4. Add a chart for each successful tool call. Charts are optional extras:
    #    if one can't be built, the answer above still stands.
    for name, arguments in chart_calls[:MAX_CHARTS]:
        fig = chart_for_call(name, arguments)
        if fig is not None:
            shown.append(gr.ChatMessage(role="assistant", content=gr.Plot(fig)))
    return shown


demo = gr.ChatInterface(
    fn=chat,
    title="🌦️ Weather & Air Quality Agent",
    description=(
        "An AI agent built from scratch: the language model decides which tool to call "
        "(weather, air quality, or both), reads the data, and writes a short report with a chart. "
        "Open the 🔧 panels to see each tool call. "
        "Data by [Open-Meteo.com](https://open-meteo.com/); air quality from "
        "[CAMS](https://atmosphere.copernicus.eu/) (Copernicus)."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    fill_height=True,
)

if __name__ == "__main__":
    port = os.getenv("PORT")
    if port:
        # Hosted (e.g. on Render): listen on all network interfaces at the given port.
        demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=int(port))
    else:
        # On your own computer: only reachable from this machine, at http://127.0.0.1:7860
        demo.launch(theme=gr.themes.Soft())
