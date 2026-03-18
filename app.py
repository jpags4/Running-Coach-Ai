from __future__ import annotations

import json
import mimetypes
import os
import re
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

from coach import Recommendation, average_easy_pace, coach_recommendation, pace_window, recent_mileage
from integrations import (
    OAuthError,
    build_strava_authorize_url,
    build_whoop_authorize_url,
    exchange_strava_code,
    exchange_whoop_code,
    fetch_strava_snapshot,
    strava_activity_preview,
    fetch_whoop_snapshot,
    generate_state,
    merge_live_data,
    profile_from_settings,
    safe_iso_today,
    snapshot_preview,
    strava_runs_to_model,
    strava_redirect_uri,
    valid_access_token,
    whoop_metrics_to_model,
    whoop_workout_preview,
    whoop_redirect_uri,
)
from sample_data import SAMPLE_METRICS, SAMPLE_PROFILE, SAMPLE_RUNS
from llm_coach import llm_recommendation
from storage import (
    init_storage,
    load_settings,
    load_states,
    load_tokens,
    load_weekly_plans,
    save_settings,
    save_states,
    save_tokens,
    save_weekly_plans,
    using_hosted_env,
)


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
FRONTEND_DIST_DIR = ROOT / "frontend" / "dist"
WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def previous_run_summary(runs: list) -> dict:
    if not runs:
        return {"day": "", "distance_miles": 0, "duration_minutes": 0, "workout_type": ""}

    latest_run = max(runs, key=lambda item: item.day)
    return {
        "day": latest_run.day,
        "distance_miles": latest_run.distance_miles,
        "duration_minutes": latest_run.duration_minutes,
        "workout_type": latest_run.workout_type.replace("_", " "),
    }


def sample_activity_preview(runs: list) -> list[dict]:
    ordered = sorted(runs, key=lambda item: item.day, reverse=True)[:8]
    return [
        {
            "source": "Strava",
            "name": run.workout_type.replace("_", " "),
            "day": run.day,
            "sport": "Run",
            "distance_miles": run.distance_miles,
            "duration_minutes": run.duration_minutes,
            "average_pace_min_per_mile": run.average_pace_min_per_mile,
        }
        for run in ordered
    ]


def _calendar_activity_kind(activity: dict) -> str:
    raw = str(activity.get("name") or activity.get("sport") or "").lower()
    normalized = "".join(char for char in raw if char.isalpha())
    if any(token in normalized for token in ("weight", "strength", "lift", "mobility", "stretch", "yoga", "pilates", "core")):
        return "strength"
    if "run" in normalized:
        return "run"
    return normalized


def _filter_calendar_activities(activity_feed: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    seen: set[tuple[str, str, int, str]] = set()

    for activity in activity_feed:
        kind = _calendar_activity_kind(activity)
        if kind not in {"run", "strength"}:
            continue

        duration = int(activity.get("duration_minutes") or 0)
        distance = str(activity.get("distance_miles") or 0)
        key = (str(activity.get("day") or ""), kind, duration, distance)
        if key in seen:
            continue

        seen.add(key)
        filtered.append(activity)

    return filtered


def _preferred_long_run_index(value: str) -> int:
    normalized = str(value or "").strip().lower()
    return WEEKDAY_NAMES.index(normalized) if normalized in WEEKDAY_NAMES else 6


def _parse_pace_bounds(text: str) -> tuple[float, float] | None:
    value = str(text or "").strip()
    range_match = re.search(r"(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})\s*(?:min/mi|/mi)", value, re.IGNORECASE)
    if range_match:
        low = int(range_match.group(1)) + int(range_match.group(2)) / 60
        high = int(range_match.group(3)) + int(range_match.group(4)) / 60
        return low, high

    single_match = re.search(r"(\d{1,2}):(\d{2})\s*(?:min/mi|/mi)", value, re.IGNORECASE)
    if single_match:
        pace = int(single_match.group(1)) + int(single_match.group(2)) / 60
        return pace, pace

    return None


def _format_pace_value(value: float) -> str:
    total_seconds = max(270, int(round(value * 60)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def _shift_pace_bounds(bounds: tuple[float, float], faster: float = 0.0, slower: float = 0.0) -> str:
    low, high = bounds
    shifted_low = max(4.5, low - faster)
    shifted_high = max(shifted_low, high + slower - faster)
    return f"{_format_pace_value(shifted_low)}-{_format_pace_value(shifted_high)}/mi"


def _pace_text_for_type(workout_type: str, recommendation) -> str:
    base = str(recommendation.run_pace_guidance or "").strip()
    bounds = _parse_pace_bounds(base)
    if bounds:
        if workout_type == "quality":
            return _shift_pace_bounds(bounds, faster=0.7, slower=-0.2)
        if workout_type == "steady":
            return _shift_pace_bounds(bounds, faster=0.3, slower=-0.05)
        if workout_type == "long":
            return _shift_pace_bounds(bounds, faster=-0.1, slower=0.2)
        return _shift_pace_bounds(bounds)

    if workout_type == "quality":
        return "8:30-8:55/mi"
    if workout_type == "steady":
        return "8:55-9:20/mi"
    if workout_type == "long":
        return "9:25-9:55/mi"
    return base or "9:15-9:45/mi"


def _lift_focus_for_day(weekday: int, long_run_day: int) -> str:
    quality_day = (long_run_day - 5) % 7
    steady_day = (long_run_day - 3) % 7
    easy_day = (quality_day - 1) % 7

    if weekday == easy_day:
        return "Single-Leg Strength + Glutes"
    if weekday == steady_day:
        return "Upper Body + Core"
    if weekday == long_run_day:
        return "Posterior Chain + Core"
    return ""


def _run_blueprints(long_run_day: int, recommendation) -> dict[int, dict]:
    quality_day = (long_run_day - 5) % 7
    steady_day = (long_run_day - 3) % 7
    easy_day = (quality_day - 1) % 7
    aerobic_day = (steady_day + 1) % 7

    run_blueprints = {
        easy_day: {
            "weight": 0.18,
            "duration": 42,
            "intensity": "easy",
            "pace_text": _pace_text_for_type("easy", recommendation),
        },
        quality_day: {
            "weight": 0.20,
            "duration": 50,
            "intensity": "hard",
            "pace_text": _pace_text_for_type("quality", recommendation),
        },
        steady_day: {
            "weight": 0.22,
            "duration": 48,
            "intensity": "moderate",
            "pace_text": _pace_text_for_type("steady", recommendation),
        },
        aerobic_day: {
            "weight": 0.16,
            "duration": 40,
            "intensity": "easy",
            "pace_text": _pace_text_for_type("easy", recommendation),
        },
        long_run_day: {
            "weight": 0.24,
            "duration": 70,
            "intensity": "moderate",
            "pace_text": _pace_text_for_type("long", recommendation),
        },
    }
    return run_blueprints


def _projected_day_template(projection_date, long_run_day: int, blueprint: dict | None, distance_miles: float) -> list[dict]:
    weekday = projection_date.weekday()
    if not blueprint:
        return []

    activities = [
        {
            "source": "Projection",
            "name": "Run",
            "day": projection_date.isoformat(),
            "sport": "Projected Run",
            "distance_miles": round(max(2.5, distance_miles), 1),
            "duration_minutes": max(20, int(blueprint["duration"])),
            "average_pace_min_per_mile": 0,
            "pace_text": blueprint["pace_text"],
            "intensity": blueprint["intensity"],
            "projected": True,
        }
    ]

    lift_focus = _lift_focus_for_day(weekday, long_run_day)
    if lift_focus:
        activities.append(
            {
                "source": "Projection",
                "name": "Lift",
                "day": projection_date.isoformat(),
                "sport": "Projected Strength",
                "distance_miles": 0,
                "duration_minutes": 30,
                "average_pace_min_per_mile": 0,
                "lift_focus": lift_focus,
                "intensity": "easy" if blueprint["intensity"] == "easy" else "moderate",
                "projected": True,
            }
        )

    return activities


def projected_calendar_entries(anchor, recommendation, end_day, profile) -> dict[str, list[dict]]:
    if not recommendation or recommendation.run_distance_miles <= 0:
        return {}

    projections: dict[str, list[dict]] = {}
    long_run_day = _preferred_long_run_index(getattr(profile, "preferred_long_run_day", "Sunday"))
    run_blueprints = _run_blueprints(long_run_day, recommendation)
    current_week_start = anchor - timedelta(days=anchor.weekday())
    base_weekly_target = max(30.0, float(getattr(profile, "weekly_mileage_target", 0) or 0))
    projection_date = anchor + timedelta(days=1)

    while projection_date <= end_day:
        week_start = projection_date - timedelta(days=projection_date.weekday())
        week_end = min(end_day, week_start + timedelta(days=6))
        week_offset = max(0, (week_start - current_week_start).days // 7)
        target_week_miles = round(base_weekly_target * (1.1 ** week_offset), 1)

        week_days = [
            day
            for day in (week_start + timedelta(days=offset) for offset in range((week_end - week_start).days + 1))
            if day > anchor
        ]
        eligible_days = [day for day in week_days if day.weekday() in run_blueprints]
        total_weight = sum(run_blueprints[day.weekday()]["weight"] for day in eligible_days) or 1.0
        remaining_miles = max(0.0, target_week_miles - (recommendation.run_distance_miles if week_offset == 0 else 0.0))

        for day in week_days:
            blueprint = run_blueprints.get(day.weekday())
            distance_miles = 0.0
            if blueprint:
                distance_miles = remaining_miles * (blueprint["weight"] / total_weight)
            projections[day.isoformat()] = _projected_day_template(day, long_run_day, blueprint, distance_miles)

        projection_date = week_end + timedelta(days=1)
    return projections


def _today_plan_entries(anchor, recommendation) -> list[dict]:
    if not recommendation:
        return []

    activities: list[dict] = []
    if recommendation.run_distance_miles > 0:
        activities.append(
            {
                "source": "Projection",
                "name": "Run",
                "day": anchor.isoformat(),
                "sport": "Projected Run",
                "distance_miles": round(max(0.0, recommendation.run_distance_miles), 1),
                "duration_minutes": max(20, recommendation.duration_minutes),
                "average_pace_min_per_mile": 0,
                "pace_text": recommendation.run_pace_guidance,
                "intensity": recommendation.intensity,
                "projected": True,
            }
        )

    if str(recommendation.lift_focus or "").strip().lower() not in {"no lifting", "today is a lifting off-day"}:
        activities.append(
            {
                "source": "Projection",
                "name": "Lift",
                "day": anchor.isoformat(),
                "sport": "Projected Strength",
                "distance_miles": 0,
                "duration_minutes": 35,
                "average_pace_min_per_mile": 0,
                "lift_focus": recommendation.lift_focus,
                "intensity": "easy" if str(recommendation.intensity).lower() == "easy" else "moderate",
                "projected": True,
            }
        )

    return activities


def _week_plan_key(day_value) -> str:
    week_start = day_value - timedelta(days=day_value.weekday())
    return week_start.isoformat()


def _generate_weekly_plan(anchor, profile, runs, metrics) -> dict[str, list[dict]]:
    metric_dates = sorted(
        (datetime.strptime(item.day, "%Y-%m-%d").date() for item in metrics if getattr(item, "day", "")),
        reverse=True,
    )
    planning_day = next((day for day in metric_dates if day <= anchor), metric_dates[0] if metric_dates else anchor)
    baseline_recommendation = coach_recommendation(profile, runs, metrics, today=planning_day)
    if baseline_recommendation.run_distance_miles <= 0:
        easy_pace = average_easy_pace(runs)
        baseline_recommendation = Recommendation(
            date=anchor.isoformat(),
            workout="Baseline aerobic run",
            intensity="easy",
            duration_minutes=48,
            run_distance_miles=round(max(4.0, min(6.0, float(getattr(profile, "weekly_mileage_target", 28) or 28) * 0.18)), 1),
            run_pace_guidance=pace_window(easy_pace, slower=0.7),
            lift_focus="Single-Leg Strength + Core",
            lift_guidance="Baseline weekly structure only.",
            recap=[],
            explanation=["Baseline week structure generated so the calendar remains stable even if today becomes a recovery day."],
            explanation_sections={
                "overall": "Baseline week structure generated so the calendar remains stable even if today becomes a recovery day.",
                "run": "The baseline run distance seeds the weekly structure before daily adjustments are applied.",
                "pace": "The baseline pace uses your recent easy running so future days keep a practical pace band.",
                "lift": "Lift slots stay in the weekly structure, but daily guardrails can still remove them.",
                "recovery": "Daily recovery can still override today's training without erasing the rest of the week.",
            },
            warnings=[],
            confidence="medium",
        )
    end_day = (anchor - timedelta(days=anchor.weekday())) + timedelta(days=13)
    plan = projected_calendar_entries(anchor, baseline_recommendation, end_day, profile)
    return plan


def _load_or_create_weekly_plan(anchor, profile, runs, metrics) -> dict[str, list[dict]]:
    plans = load_weekly_plans()
    key = _week_plan_key(anchor)
    plan = plans.get(key)
    if isinstance(plan, dict) and plan:
        return plan

    plan = _generate_weekly_plan(anchor, profile, runs, metrics)
    plans[key] = plan
    save_weekly_plans(plans)
    return plan


def calendar_days(activity_feed: list[dict], metrics: list, recommendation=None, today: str = "", profile=None, weekly_plan=None) -> list[dict]:
    anchor = datetime.strptime(today, "%Y-%m-%d").date() if today else datetime.utcnow().date()
    feed_by_day: dict[str, list[dict]] = {}
    for item in activity_feed:
        day = item.get("day", "")
        if not day:
            continue
        feed_by_day.setdefault(day, []).append(item)

    start_day = anchor - timedelta(days=anchor.weekday())
    end_day = start_day + timedelta(days=13)
    projected_by_day = weekly_plan or {}

    cards: list[dict] = []
    current_day = start_day
    while current_day <= end_day:
        iso_day = current_day.isoformat()
        activities = feed_by_day.get(iso_day, [])
        projected = False
        if not activities and current_day == anchor:
            activities = _today_plan_entries(anchor, recommendation)
            projected = bool(activities)
        elif not activities and current_day > anchor:
            activities = projected_by_day.get(iso_day, [])
            projected = bool(activities)

        cards.append(
            {
                "day": iso_day,
                "activities": sorted(
                    activities,
                    key=lambda item: (item.get("projected", False), item.get("duration_minutes", 0)),
                    reverse=True,
                ),
                "is_today": iso_day == anchor.isoformat(),
                "is_current_month": current_day.month == anchor.month,
                "is_projection": projected,
            }
        )
        current_day += timedelta(days=1)

    return cards


def build_dashboard_payload(settings, tokens, subjective_feedback: dict | None = None, include_recommendation: bool = False) -> dict:
    connection_status = {
        "strava": bool(settings.get("strava", {}).get("client_id")) and bool(tokens.get("strava")),
        "whoop": bool(settings.get("whoop", {}).get("client_id")) and bool(tokens.get("whoop")),
    }
    profile = profile_from_settings(settings)
    runs = SAMPLE_RUNS
    metrics = SAMPLE_METRICS
    live_preview = {"mode": "sample"}
    activity_feed = sample_activity_preview(runs)

    try:
        live_strava = None
        live_whoop = None
        warnings: list[str] = []

        if tokens.get("strava"):
            refreshed = valid_access_token("strava", settings, tokens)
            if refreshed:
                tokens["strava"] = refreshed
                save_tokens(tokens)
                live_strava = fetch_strava_snapshot(
                    refreshed["access_token"],
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
            else:
                warnings.append("Strava token could not be refreshed.")

        if tokens.get("whoop"):
            refreshed = valid_access_token("whoop", settings, tokens)
            if refreshed:
                tokens["whoop"] = refreshed
                save_tokens(tokens)
                live_whoop = fetch_whoop_snapshot(
                    refreshed["access_token"],
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
            else:
                warnings.append("WHOOP token could not be refreshed.")

        if live_strava and live_whoop:
            merged = merge_live_data(live_strava, live_whoop, settings)
            if merged["runs"] and merged["metrics"]:
                profile = merged["profile"]
                runs = merged["runs"]
                metrics = merged["metrics"]
                activity_feed = strava_activity_preview(live_strava.get("activities", [])) + whoop_workout_preview(live_whoop)
                activity_feed = _filter_calendar_activities(sorted(activity_feed, key=lambda item: item.get("day", ""), reverse=True))[:20]
                live_preview = {
                    "mode": "live",
                    "strava_runs_found": len(runs),
                    "whoop_days_found": len(metrics),
                }
        else:
            if live_strava:
                runs = strava_runs_to_model(live_strava.get("activities", [])) or runs
                activity_feed = strava_activity_preview(live_strava.get("activities", []))
            if live_whoop:
                metrics = whoop_metrics_to_model(live_whoop) or metrics
                whoop_activities = whoop_workout_preview(live_whoop)
                if live_strava:
                    activity_feed = activity_feed + whoop_activities
                elif whoop_activities:
                    activity_feed = whoop_activities

            if live_strava or live_whoop:
                activity_feed = _filter_calendar_activities(sorted(activity_feed, key=lambda item: item.get("day", ""), reverse=True))[:20]
                live_preview = {
                    "mode": "mixed",
                    "strava_runs_found": len(runs) if live_strava else 0,
                    "whoop_days_found": len(metrics) if live_whoop else 0,
                    "warning": " ".join(warnings) if warnings else "One provider loaded live data while the other fell back.",
                }
    except Exception as exc:
        live_preview = {"mode": "sample", "warning": str(exc)}

    today_iso = safe_iso_today()
    today_date = datetime.strptime(today_iso, "%Y-%m-%d").date()
    weekly_plan = _load_or_create_weekly_plan(today_date, profile, runs, metrics)
    recommendation = None
    recommendation_meta = {"source": None, "model": None, "reason": None}
    if include_recommendation:
        recommendation, recommendation_meta = llm_recommendation(
            profile,
            runs,
            metrics,
            today=today_date,
            subjective_feedback=subjective_feedback,
        )

    payload = {
        "profile": {
            "name": profile.name,
            "goal_race_date": profile.goal_race_date,
            "weekly_mileage_target": profile.weekly_mileage_target,
            "preferred_long_run_day": profile.preferred_long_run_day,
        },
        "summary": {
            "recent_mileage": recent_mileage(runs),
            "latest_recovery": metrics[-1].recovery_score,
            "latest_sleep_hours": metrics[-1].sleep_hours,
            "latest_strain": metrics[-1].strain,
            "latest_resting_hr": metrics[-1].resting_hr,
            "latest_hrv": metrics[-1].hrv_ms,
            "previous_run": previous_run_summary(runs),
        },
        "recommendation": recommendation.to_dict() if recommendation else None,
        "recommendation_meta": recommendation_meta,
        "weekly_plan_key": _week_plan_key(today_date),
        "activity_feed": activity_feed,
        "activity_calendar": calendar_days(
            activity_feed,
            metrics,
            recommendation=recommendation,
            today=today_iso,
            profile=profile,
            weekly_plan=weekly_plan,
        ),
        "connections": {
            "status": connection_status,
            "setup_complete": {
                "strava": bool(settings.get("strava", {}).get("client_id")),
                "whoop": bool(settings.get("whoop", {}).get("client_id")),
            },
            "public_base_url": settings.get("public_base_url", ""),
            "strava_callback_url": strava_redirect_uri(settings.get("public_base_url", "")) if settings.get("public_base_url") else "",
            "whoop_callback_url": whoop_redirect_uri(settings.get("public_base_url", "")) if settings.get("public_base_url") else "",
        },
        "data_mode": live_preview,
        "today": today_iso,
    }
    return payload


def html_page(title: str, body: str) -> str:
    return f"""
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{escape(title)}</title>
        <style>
          body {{
            margin: 0;
            padding: 32px;
            font-family: Georgia, "Times New Roman", serif;
            background: linear-gradient(145deg, #f7f0e7, #e6efe8 60%, #dce8e4);
            color: #17211f;
          }}

          main {{
            max-width: 900px;
            margin: 0 auto;
          }}

          .card {{
            background: rgba(255, 251, 245, 0.94);
            border: 1px solid rgba(23, 33, 31, 0.12);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 18px;
          }}

          .button {{
            display: inline-block;
            padding: 12px 16px;
            border-radius: 999px;
            background: #d96c3f;
            color: white;
            text-decoration: none;
            margin-right: 10px;
          }}

          input {{
            width: 100%;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid rgba(23, 33, 31, 0.16);
            margin-top: 6px;
            margin-bottom: 14px;
          }}

          label {{
            font-size: 0.92rem;
          }}
        </style>
      </head>
      <body>
        <main>{body}</main>
      </body>
    </html>
    """


def setup_form(settings: dict, message: str = "") -> str:
    strava = settings.get("strava", {})
    whoop = settings.get("whoop", {})
    hosted_env = using_hosted_env()
    public_base_url = settings.get("public_base_url", "")
    return html_page(
        "Setup",
        f"""
        <div class="card">
          <h1>App Setup</h1>
          <p>This page stores your app IDs and secrets on your own computer so you do not need to type them into Terminal.</p>
          <p><strong>{'Hosted environment variables are active, so this page is now mainly for reference.' if hosted_env else 'Right now the app is using local settings saved on this machine.'}</strong></p>
          {f"<p><strong>{escape(message)}</strong></p>" if message else ""}
        </div>

        <form method="POST" action="/setup" class="card">
          <h2>About You</h2>
          <label>Athlete name
            <input name="athlete_name" value="{escape(settings.get("athlete_name", ""))}" />
          </label>
          <label>Goal race date (example: 2026-05-10)
            <input name="goal_race_date" value="{escape(settings.get("goal_race_date", ""))}" />
          </label>
          <label>Weekly mileage target
            <input name="weekly_mileage_target" value="{escape(str(settings.get("weekly_mileage_target", "28")))}" />
          </label>
          <label>Preferred long run day
            <input name="preferred_long_run_day" value="{escape(settings.get("preferred_long_run_day", "Sunday"))}" />
          </label>

          <h2>Strava</h2>
          <label>Strava client ID
            <input name="strava_client_id" value="{escape(strava.get("client_id", ""))}" {"readonly" if hosted_env else ""} />
          </label>
          <label>Strava client secret
            <input name="strava_client_secret" value="{escape(strava.get("client_secret", ""))}" {"readonly" if hosted_env else ""} />
          </label>

          <h2>WHOOP</h2>
          <label>WHOOP client ID
            <input name="whoop_client_id" value="{escape(whoop.get("client_id", ""))}" {"readonly" if hosted_env else ""} />
          </label>
          <label>WHOOP client secret
            <input name="whoop_client_secret" value="{escape(whoop.get("client_secret", ""))}" {"readonly" if hosted_env else ""} />
          </label>
          <label>Public base URL (local example: https://your-name.ngrok-free.dev, hosted example: https://your-app.onrender.com)
            <input name="public_base_url" value="{escape(public_base_url)}" {"readonly" if hosted_env else ""} />
          </label>
          <label>
            <input type="checkbox" name="allow_insecure_ssl" {"checked" if settings.get("allow_insecure_ssl") else ""} style="width:auto; margin-right:8px;" />
            Allow insecure SSL for local development only
          </label>
          <p>This is only for cases where your network adds its own certificate and Python refuses the connection.</p>

          {'<button type="submit">Save setup</button>' if not hosted_env else '<p>Change hosted values in your hosting dashboard environment settings.</p>'}
        </form>

        <div class="card">
          <h2>Useful Callback URLs</h2>
          <p><strong>Strava:</strong> {escape(strava_redirect_uri(public_base_url) if public_base_url else "Add your public base URL above to generate this")}</p>
          <p><strong>WHOOP:</strong> {escape(whoop_redirect_uri(public_base_url) if public_base_url else "Add your public base URL above to generate this")}</p>
          <p><a class="button" href="/">Back to dashboard</a></p>
        </div>
        """,
    )


def callback_success_page(provider: str, preview: dict) -> str:
    latest = preview.get("latest_item")
    latest_html = f"<pre>{escape(json.dumps(latest, indent=2))}</pre>" if latest else "<p>No sample item was returned yet.</p>"
    return html_page(
        f"{provider} Connected",
        f"""
        <div class="card">
          <h1>{escape(provider)} connected</h1>
          <p>The app successfully completed the login return step and saved your token locally.</p>
        </div>
        <div class="card">
          <h2>What the app just imported</h2>
          <p><strong>Provider:</strong> {escape(preview.get("provider", provider))}</p>
          <p><strong>Name on account:</strong> {escape(preview.get("athlete_name", "") or "Not returned")}</p>
          <p><strong>Items found:</strong> {preview.get("items_found", 0)}</p>
          {latest_html}
        </div>
        <div class="card">
          <a class="button" href="/">Back to dashboard</a>
          <a class="button" href="/setup">Open setup</a>
        </div>
        """,
    )


def callback_warning_page(title: str, details: str) -> str:
    return html_page(
        title,
        f"""
        <div class="card">
          <h1>{escape(title)}</h1>
          <p>{escape(details)}</p>
          <p>This relaxed check is only for local development while we get your connection working.</p>
        </div>
        <div class="card">
          <a class="button" href="/">Back to dashboard</a>
          <a class="button" href="/setup">Open setup</a>
        </div>
        """,
    )


def error_page(title: str, details: str) -> str:
    return html_page(
        title,
        f"""
        <div class="card">
          <h1>{escape(title)}</h1>
          <p>{escape(details)}</p>
          <p><a class="button" href="/setup">Open setup</a> <a class="button" href="/">Back to dashboard</a></p>
        </div>
        """,
    )


def debug_query_summary(query: dict[str, list[str]]) -> str:
    safe_query = {key: values for key, values in query.items()}
    return escape(json.dumps(safe_query, indent=2))


class CoachHandler(BaseHTTPRequestHandler):
    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(200)
        resolved_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if resolved_type.startswith("text/") or resolved_type in {"application/javascript", "application/json"}:
            resolved_type = f"{resolved_type}; charset=utf-8"
        self.send_header("Content-Type", resolved_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html_file(self, filename: str) -> None:
        react_path = FRONTEND_DIST_DIR / filename
        static_path = STATIC_DIR / filename
        if react_path.exists():
            self._send_file(react_path, content_type="text/html")
            return
        self._send_file(static_path, content_type="text/html")

    def _send_html_text(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _read_form(self) -> dict[str, str]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        parsed = parse_qs(body)
        return {key: values[0] for key, values in parsed.items()}

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(body or "{}")

    def _connected_status(self, settings: dict, tokens: dict) -> dict:
        return {
            "strava": bool(settings.get("strava", {}).get("client_id")) and bool(tokens.get("strava")),
            "whoop": bool(settings.get("whoop", {}).get("client_id")) and bool(tokens.get("whoop")),
        }

    def do_POST(self) -> None:
        if self.path == "/api/recommendation":
            settings = load_settings()
            tokens = load_tokens()
            try:
                form = self._read_json()
            except Exception:
                self._send_json({"error": "Invalid JSON body."}, status=400)
                return

            payload = build_dashboard_payload(
                settings,
                tokens,
                subjective_feedback={
                    "physical_feeling": str(form.get("physical_feeling", "")).strip(),
                    "mental_feeling": str(form.get("mental_feeling", "")).strip(),
                    "notes": str(form.get("notes", "")).strip(),
                },
                include_recommendation=True,
            )
            self._send_json(payload)
            return

        if self.path != "/setup":
            self._send_html_text(error_page("Not found", "That form target does not exist."), status=404)
            return
        if using_hosted_env():
            self._send_html_text(setup_form(load_settings(), message="Hosted environment variables are active, so local setup changes are disabled."))
            return

        form = self._read_form()
        settings = {
            "athlete_name": form.get("athlete_name", "").strip(),
            "goal_race_date": form.get("goal_race_date", "").strip(),
            "weekly_mileage_target": form.get("weekly_mileage_target", "28").strip() or "28",
            "preferred_long_run_day": form.get("preferred_long_run_day", "Sunday").strip() or "Sunday",
            "public_base_url": form.get("public_base_url", "").strip(),
            "allow_insecure_ssl": form.get("allow_insecure_ssl") == "on",
            "strava": {
                "client_id": form.get("strava_client_id", "").strip(),
                "client_secret": form.get("strava_client_secret", "").strip(),
            },
            "whoop": {
                "client_id": form.get("whoop_client_id", "").strip(),
                "client_secret": form.get("whoop_client_secret", "").strip(),
            },
        }
        save_settings(settings)
        self._send_html_text(setup_form(settings, message="Saved. You can go back and press Connect now."))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        settings = load_settings()
        tokens = load_tokens()
        states = load_states()

        if parsed.path.startswith("/assets/"):
            asset_path = FRONTEND_DIST_DIR / parsed.path.lstrip("/")
            if asset_path.exists() and asset_path.is_file():
                self._send_file(asset_path)
                return

        if parsed.path == "/":
            self._send_html_file("index.html")
            return

        if parsed.path == "/setup":
            self._send_html_text(setup_form(settings))
            return

        if parsed.path == "/connect/strava":
            client_id = settings.get("strava", {}).get("client_id", "")
            public_base_url = settings.get("public_base_url", "")
            if not client_id:
                self._send_html_text(error_page("Missing Strava setup", "Open Setup and add your Strava client ID and client secret first."))
                return
            if not public_base_url:
                self._send_html_text(error_page("Missing public URL", "Open Setup and add your app's public base URL first."))
                return

            state = generate_state()
            states["strava"] = state
            save_states(states)
            self._redirect(build_strava_authorize_url(client_id, public_base_url, state))
            return

        if parsed.path == "/connect/whoop":
            client_id = settings.get("whoop", {}).get("client_id", "")
            public_base_url = settings.get("public_base_url", "")
            if not client_id or not public_base_url:
                self._send_html_text(error_page("Missing WHOOP setup", "Open Setup and add your WHOOP client ID, client secret, and ngrok public URL first."))
                return

            state = generate_state()
            states["whoop"] = state
            save_states(states)
            self._redirect(build_whoop_authorize_url(client_id, public_base_url, state))
            return

        if parsed.path == "/strava/callback":
            if query.get("state", [""])[0] != states.get("strava", ""):
                self._send_html_text(error_page("Strava state mismatch", "The app could not verify that this login return belongs to your current session. Please try Connect Strava again."))
                return

            code = query.get("code", [""])[0]
            if not code:
                self._send_html_text(error_page("Missing Strava code", "Strava returned without an authorization code."))
                return

            try:
                provider_settings = settings.get("strava", {})
                public_base_url = settings.get("public_base_url", "")
                token_payload = exchange_strava_code(
                    provider_settings["client_id"],
                    provider_settings["client_secret"],
                    strava_redirect_uri(public_base_url),
                    code,
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
                tokens["strava"] = token_payload
                save_tokens(tokens)
                snapshot = fetch_strava_snapshot(
                    token_payload["access_token"],
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
                self._send_html_text(callback_success_page("Strava", snapshot_preview("strava", snapshot)))
            except Exception as exc:
                self._send_html_text(error_page("Strava connection failed", str(exc)), status=500)
            return

        if parsed.path == "/whoop/callback":
            code = query.get("code", [""])[0]
            if not code:
                details = "WHOOP returned without an authorization code."
                debug_html = html_page(
                    "Missing WHOOP code",
                    f"""
                    <div class="card">
                      <h1>Missing WHOOP code</h1>
                      <p>{escape(details)}</p>
                      <p>This means WHOOP reached your app, but did not include the login code needed to finish the connection.</p>
                    </div>
                    <div class="card">
                      <h2>What WHOOP sent back</h2>
                      <pre>{debug_query_summary(query)}</pre>
                    </div>
                    <div class="card">
                      <a class="button" href="/">Back to dashboard</a>
                      <a class="button" href="/setup">Open setup</a>
                    </div>
                    """,
                )
                self._send_html_text(debug_html)
                return

            returned_state = query.get("state", [""])[0]
            expected_state = states.get("whoop", "")
            state_warning = ""
            if expected_state and returned_state and returned_state != expected_state:
                state_warning = "WHOOP returned a different state value than expected, so the app is continuing in relaxed local development mode."
            elif expected_state and not returned_state:
                state_warning = "WHOOP did not return a state value, so the app is continuing in relaxed local development mode."

            try:
                provider_settings = settings.get("whoop", {})
                redirect_uri = whoop_redirect_uri(settings.get("public_base_url", ""))
                token_payload = exchange_whoop_code(
                    provider_settings["client_id"],
                    provider_settings["client_secret"],
                    redirect_uri,
                    code,
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
                tokens["whoop"] = token_payload
                save_tokens(tokens)
                snapshot = fetch_whoop_snapshot(
                    token_payload["access_token"],
                    allow_insecure_ssl=bool(settings.get("allow_insecure_ssl")),
                    user_agent=settings.get("app_user_agent", "AdaptiveRunningCoach/0.1"),
                )
                body = callback_success_page("WHOOP", snapshot_preview("whoop", snapshot))
                if state_warning:
                    body = body.replace(
                        "<div class=\"card\">\n          <h2>What the app just imported</h2>",
                        f"<div class=\"card\"><p><strong>{escape(state_warning)}</strong></p></div><div class=\"card\">\n          <h2>What the app just imported</h2>",
                    )
                self._send_html_text(body)
            except Exception as exc:
                self._send_html_text(error_page("WHOOP connection failed", str(exc)), status=500)
            return

        if parsed.path == "/api/dashboard":
            query = parse_qs(parsed.query)
            include_recommendation = query.get("include_recommendation", ["0"])[0].lower() in {"1", "true", "yes"}
            payload = build_dashboard_payload(settings, tokens, include_recommendation=include_recommendation)
            self._send_json(payload)
            return

        frontend_index = FRONTEND_DIST_DIR / "index.html"
        if frontend_index.exists() and not parsed.path.startswith("/api/"):
            self._send_file(frontend_index, content_type="text/html")
            return

        self._send_html_text(error_page("Not found", "That page does not exist."), status=404)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), CoachHandler)
    print(f"Serving adaptive run coach at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    init_storage()
    default_host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    run_server(
        host=os.environ.get("HOST", default_host),
        port=int(os.environ.get("PORT", "8000")),
    )
