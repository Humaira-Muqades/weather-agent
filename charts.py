"""
charts.py
=========
Interactive charts shown under the agent's answer in the web interface.

The agent's tools return short daily summaries, which is what the language model needs.
A chart needs hour-by-hour detail, so this file fetches that separately from the same
free Open-Meteo APIs, for the same city and number of days the agent asked for.

Design choices:
- Temperature and chance of rain are two stacked panels, not one chart with two y-axes:
  a second y-axis makes readers compare numbers that are not comparable.
- Air quality is drawn on the US AQI scale with the official bands shaded and labelled,
  so the colour is never the only way to tell "Good" from "Unhealthy".
- Transparent background and mid-grey text, so the charts read in light and dark mode.
"""

import json
from datetime import datetime

import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

from weather_agent import (AIR_QUALITY_URL, FORECAST_URL, aqi_category,
                           clamp_days, find_location, place_label)

# Colours checked for colour-blind separation and contrast on both light and dark backgrounds.
TEMPERATURE = "#d95926"  # orange
RAIN = "#3987e5"         # blue
AQI_LINE = "#3987e5"     # blue
TEXT = "#7d7b75"         # mid grey, readable on white and on dark navy
GRID = "rgba(125, 123, 117, 0.22)"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# US AQI bands shaded behind the line: (low, high, short label, colour).
# The full band names appear in the hover text; the labels sit in the right margin.
AQI_BANDS = [
    (0, 50, "Good", "#0ca30c"),
    (50, 100, "Moderate", "#fab219"),
    (100, 150, "Sensitive groups", "#ec835a"),
    (150, 200, "Unhealthy", "#d03b3b"),
    (200, 300, "Very unhealthy", "#d03b3b"),
    (300, 500, "Hazardous", "#d03b3b"),
]


def _base_layout(fig, title, height, right_margin=16):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=TEXT)),
        height=height,
        margin=dict(l=48, r=right_margin, t=56, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=TEXT),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT)),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID,
                     tickformat="%a %H:%M", nticks=8)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID)


def _mark_now(fig, now, rows, side="right"):
    """A thin vertical line at the current local time, labelled 'Now' on the given side."""
    if not now:
        return
    for row in rows:
        fig.add_vline(x=now, line=dict(color=TEXT, width=1), row=row, col=1)
    # Label it inside the top panel, so it never collides with the panel titles.
    fig.add_annotation(x=now, y=1, yref="y domain", text="Now", showarrow=False,
                       xanchor="left" if side == "right" else "right", yanchor="top",
                       xshift=4 if side == "right" else -4, font=dict(size=11, color=TEXT))


def _hours(stamp):
    """Hours since year 0 for an ISO time like '2026-10-08T12:30' (only used for distances)."""
    d = datetime.fromisoformat(stamp)
    return d.toordinal() * 24 + d.hour + d.minute / 60


def _place_top_label(times, label_x, now):
    """Keep a label printed above a marker clear of the 'Now' label at the top of the chart.

    Returns (text position for the marker label, side for the 'Now' label). When the two
    are close, they are pushed to opposite sides of the 'Now' line.
    """
    if not now or not times:
        return "top center", "right"
    span = max(_hours(times[-1]) - _hours(times[0]), 1)
    gap = (_hours(label_x) - _hours(now)) / span
    if abs(gap) > 0.08:                      # far apart: no clash
        return "top center", "right"
    if gap < 0:                              # marker just before now
        return "top left", "right"
    return "top right", "left"               # marker just after now


def weather_chart(city, days=1):
    """Hourly temperature (top) and chance of rain (bottom) for the requested days."""
    days = clamp_days(days)
    place = find_location(city)
    if place is None:
        return None
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m",
            "hourly": "temperature_2m,precipitation_probability",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rain = [r if r is not None else 0 for r in hourly.get("precipitation_probability", [])]
    if not times or not temps:
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        vertical_spacing=0.14,
                        subplot_titles=("Temperature (°C)", "Chance of rain (%)"))
    fig.add_trace(go.Scatter(
        x=times, y=temps, mode="lines", name="Temperature",
        line=dict(color=TEMPERATURE, width=2, shape="spline"),
        hovertemplate="%{y:.1f} °C<extra></extra>",
    ), row=1, col=1)

    # Label only the extremes: the warmest and coolest hour.
    now = data.get("current", {}).get("time")
    now_side = "right"
    valid = [(t, x) for t, x in zip(temps, times) if t is not None]
    if valid:
        hi, lo = max(valid), min(valid)
        hi_position, now_side = _place_top_label(times, hi[1], now)
        fig.add_trace(go.Scatter(
            x=[hi[1], lo[1]], y=[hi[0], lo[0]], mode="markers+text",
            marker=dict(size=8, color=TEMPERATURE),
            text=[f"high {hi[0]:.0f}°", f"low {lo[0]:.0f}°"],
            textposition=[hi_position, "bottom center"],
            textfont=dict(color=TEXT, size=11), hoverinfo="skip",
        ), row=1, col=1)
        span = max(hi[0] - lo[0], 4)
        fig.update_yaxes(range=[lo[0] - span * 0.35, hi[0] + span * 0.35], row=1, col=1)

    fig.add_trace(go.Bar(
        x=times, y=rain, name="Chance of rain", marker=dict(color=RAIN, line=dict(width=0)),
        hovertemplate="%{y:.0f}% chance of rain<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[0, 50, 100], row=2, col=1)
    if max(rain, default=0) < 5:
        fig.add_annotation(x=0.5, y=50, xref="x2 domain", yref="y2", showarrow=False,
                           text="No rain expected", font=dict(color=TEXT, size=12))

    label = "today" if days == 1 else f"next {days} days"
    _base_layout(fig, f"{place_label(place)} · {label}", height=430)
    fig.update_layout(bargap=0.15, barcornerradius=3 if days <= 2 else 0)
    # Panel titles: left-aligned, small, in the same grey as the rest of the text.
    fig.for_each_annotation(lambda a: a.update(x=0, xanchor="left", font=dict(size=12, color=TEXT))
                            if a.text in ("Temperature (°C)", "Chance of rain (%)") else None)
    _mark_now(fig, now, rows=(1, 2), side=now_side)
    return fig


def air_quality_chart(city, days=1):
    """Hourly US AQI for the requested days, over the shaded AQI bands."""
    days = clamp_days(days)
    place = find_location(city)
    if place is None:
        return None
    resp = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "us_aqi",
            "hourly": "us_aqi",
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    points = [(t, v) for t, v in zip(hourly.get("time", []), hourly.get("us_aqi", [])) if v is not None]
    if not points:
        return None
    times = [t for t, _ in points]
    values = [v for _, v in points]
    top = max(110, max(values) * 1.2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines", name="US AQI",
        line=dict(color=AQI_LINE, width=2, shape="spline"),
        customdata=[aqi_category(v) for v in values],
        hovertemplate="US AQI %{y:.0f} · %{customdata}<extra></extra>",
    ))
    now = data.get("current", {}).get("time")
    peak = max(points, key=lambda p: p[1])
    peak_position, now_side = _place_top_label(times, peak[0], now)
    fig.add_trace(go.Scatter(
        x=[peak[0]], y=[peak[1]], mode="markers+text", marker=dict(size=8, color=AQI_LINE),
        text=[f"peak {peak[1]:.0f}"], textposition=peak_position,
        textfont=dict(color=TEXT, size=11), hoverinfo="skip",
    ))
    fig.update_yaxes(range=[0, top], title=dict(text="US AQI", font=dict(color=TEXT)))

    # Shade each AQI band inside the visible range, and name it in the right margin
    # (outside the plot, so the line never runs over the labels).
    for low, high, label, colour in AQI_BANDS:
        if low >= top:
            break
        fig.add_hrect(y0=low, y1=min(high, top), fillcolor=colour, opacity=0.16,
                      line_width=0, layer="below")
        if min(high, top) - low >= top * 0.07:  # only label bands tall enough to hold text
            fig.add_annotation(x=1, xref="paper", xanchor="left", xshift=8,
                               y=(low + min(high, top)) / 2, text=label, showarrow=False,
                               font=dict(color=TEXT, size=11))

    label = "today" if days == 1 else f"next {days} days"
    _base_layout(fig, f"{place_label(place)} · air quality, {label}", height=340,
                 right_margin=118)
    _mark_now(fig, now, rows=(1,), side=now_side)
    return fig


CHARTS = {"get_weather": weather_chart, "get_air_quality": air_quality_chart}


def chart_for_call(name, arguments_json):
    """Build the chart for one tool call, or return None. Never raises: a chart is optional."""
    builder = CHARTS.get(name)
    if builder is None:
        return None
    try:
        args = json.loads(arguments_json or "{}")
        return builder(args.get("city", ""), args.get("days", 1))
    except Exception:
        return None
