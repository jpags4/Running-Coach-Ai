from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from statistics import mean


@dataclass
class Run:
    day: str
    distance_miles: float
    duration_minutes: int
    effort: str
    workout_type: str
    average_pace_min_per_mile: float = 0.0
    source: str = "strava"


@dataclass
class RecoveryMetrics:
    day: str
    recovery_score: int
    sleep_hours: float
    resting_hr: int
    hrv_ms: int
    strain: float


@dataclass
class AthleteProfile:
    name: str
    goal_race_date: str
    weekly_mileage_target: int
    preferred_long_run_day: str


@dataclass
class Recommendation:
    date: str
    workout: str
    intensity: str
    duration_minutes: int
    run_distance_miles: float
    run_pace_guidance: str
    lift_focus: str
    lift_guidance: str
    recap: list[str]
    explanation: list[str]
    explanation_sections: dict[str, str]
    warnings: list[str]
    confidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def days_until_race(goal_race_date: str, today: date) -> int:
    return (parse_date(goal_race_date) - today).days


def recent_mileage(runs: list[Run], days: int = 7) -> float:
    if not runs:
        return 0.0

    latest_day = max(parse_date(run.day) for run in runs)
    cutoff = latest_day.toordinal() - days + 1
    return round(
        sum(run.distance_miles for run in runs if parse_date(run.day).toordinal() >= cutoff),
        1,
    )


def acute_load(runs: list[Run], days: int = 3) -> float:
    if not runs:
        return 0.0

    latest_day = max(parse_date(run.day) for run in runs)
    cutoff = latest_day.toordinal() - days + 1
    return round(
        sum(run.duration_minutes for run in runs if parse_date(run.day).toordinal() >= cutoff),
        1,
    )


def average_resting_hr(metrics: list[RecoveryMetrics], days: int = 7) -> float:
    if not metrics:
        return 0.0

    latest_day = max(parse_date(item.day) for item in metrics)
    cutoff = latest_day.toordinal() - days + 1
    values = [item.resting_hr for item in metrics if parse_date(item.day).toordinal() >= cutoff]
    return round(mean(values), 1) if values else 0.0


def previous_run(runs: list[Run], today: date) -> Run | None:
    previous_runs = [run for run in runs if parse_date(run.day) < today]
    return max(previous_runs, key=lambda item: item.day) if previous_runs else None


def average_easy_pace(runs: list[Run], days: int = 28) -> float:
    eligible = [run.average_pace_min_per_mile for run in runs if run.average_pace_min_per_mile > 0 and run.effort == "easy"]
    if eligible:
        return round(mean(eligible[-6:]), 2)

    all_paces = [run.average_pace_min_per_mile for run in runs if run.average_pace_min_per_mile > 0]
    return round(mean(all_paces[-6:]), 2) if all_paces else 10.0


def pace_window(base_pace: float, faster: float = 0.0, slower: float = 0.0) -> str:
    low = max(4.5, base_pace - faster)
    high = base_pace + slower
    return f"{low:.2f}-{high:.2f} min/mi"


def coach_recommendation(
    profile: AthleteProfile,
    runs: list[Run],
    metrics: list[RecoveryMetrics],
    today: date | None = None,
) -> Recommendation:
    today = today or max(parse_date(item.day) for item in metrics)
    today_str = today.isoformat()
    latest_metrics = next((item for item in metrics if item.day == today_str), None)

    if latest_metrics is None:
        raise ValueError(f"No Whoop-style metrics found for {today_str}")

    seven_day_miles = recent_mileage(runs, days=7)
    three_day_load = acute_load(runs, days=3)
    baseline_rhr = average_resting_hr(metrics, days=7)
    race_countdown = days_until_race(profile.goal_race_date, today)
    latest_run = max(runs, key=lambda item: item.day) if runs else None
    prior_run = previous_run(runs, today)
    easy_pace = average_easy_pace(runs)
    recent_quality_run = any(
        run.effort in {"hard", "very hard"}
        and 0 < (today - parse_date(run.day)).days <= 3
        for run in runs
    )

    explanation: list[str] = []
    warnings: list[str] = []
    recap: list[str] = []

    if race_countdown <= 14:
        phase = "taper"
    elif race_countdown <= 42:
        phase = "specific"
    else:
        phase = "base"

    recovery_low = latest_metrics.recovery_score < 40
    recovery_ok = 40 <= latest_metrics.recovery_score < 67
    sleep_low = latest_metrics.sleep_hours < 6.5
    strain_high = latest_metrics.strain >= 16
    elevated_rhr = baseline_rhr and latest_metrics.resting_hr >= baseline_rhr + 4
    load_high = three_day_load >= 180
    yesterday_hard = bool(latest_run and latest_run.effort in {"hard", "very hard"} and latest_run.day != today_str)

    if prior_run:
        recap.append(
            f"Most recent run: {prior_run.distance_miles:.1f} miles in {prior_run.duration_minutes} minutes at about {prior_run.average_pace_min_per_mile:.2f} min/mi."
        )
    recap.append(f"Seven-day running total: {seven_day_miles:.1f} miles.")
    recap.append(
        f"Latest WHOOP: recovery {latest_metrics.recovery_score}%, sleep {latest_metrics.sleep_hours:.1f} hours, strain {latest_metrics.strain:.1f}."
    )

    if recovery_low or (sleep_low and elevated_rhr):
        workout = "Recovery day"
        intensity = "very easy"
        duration = 30
        run_distance = 2.5
        run_pace = pace_window(easy_pace, slower=1.1)
        lift_focus = "Mobility and tissue care"
        lift_guidance = "Skip heavy lifting. If you want to train, do 20-30 minutes of mobility, light core work, and easy activation."
        explanation.append("Recovery signals are suppressed, so today's priority is absorbing training.")
    elif strain_high and load_high:
        workout = "Easy aerobic run"
        intensity = "easy"
        duration = 40
        run_distance = round(max(3.0, min(5.0, duration / max(easy_pace, 1))), 1)
        run_pace = pace_window(easy_pace, slower=0.8)
        lift_focus = "Upper body only"
        lift_guidance = "Keep lifting light to moderate. Avoid grinding lower-body work because your recent run load is already high."
        explanation.append("Short-term load is already high, so keeping intensity down lowers injury risk.")
    elif phase == "taper":
        workout = "Race-pace primer"
        intensity = "moderate"
        duration = 35
        run_distance = round(max(3.0, min(5.0, duration / max(easy_pace - 0.3, 1))), 1)
        run_pace = pace_window(max(5.0, easy_pace - 0.45), faster=0.1, slower=0.2)
        lift_focus = "Sharp but short"
        lift_guidance = "If lifting today, keep it brief. Use low volume, avoid soreness, and focus on movement quality rather than load."
        explanation.append("The race is close, so we preserve sharpness without adding much fatigue.")
    elif yesterday_hard or recent_quality_run or recovery_ok:
        workout = "Easy aerobic run"
        intensity = "easy"
        duration = 45
        run_distance = round(max(3.5, min(5.5, duration / max(easy_pace, 1))), 1)
        run_pace = pace_window(easy_pace, slower=0.7)
        lift_focus = "Technique-based strength"
        lift_guidance = "Keep lifting controlled. Choose moderate weights, leave a few reps in reserve, and avoid a second maximal stressor."
        explanation.append("Recent work suggests an aerobic support day is better than another quality session.")
    elif phase == "specific":
        workout = "Tempo session"
        intensity = "moderately hard"
        duration = 55
        run_distance = round(max(5.0, min(7.0, duration / max(easy_pace - 0.4, 1))), 1)
        run_pace = pace_window(max(5.0, easy_pace - 0.7), faster=0.15, slower=0.2)
        lift_focus = "Light supplemental lift"
        lift_guidance = "If you lift today, keep it secondary to the run: short session, lighter lower-body work, and no failure sets."
        explanation.append("You're in a race-specific window, so threshold work helps half marathon readiness.")
    else:
        workout = "Steady endurance run"
        intensity = "moderate"
        duration = 50
        run_distance = round(max(4.5, min(6.5, duration / max(easy_pace - 0.15, 1))), 1)
        run_pace = pace_window(max(5.0, easy_pace - 0.25), faster=0.1, slower=0.35)
        lift_focus = "Full-body strength"
        lift_guidance = "A normal strength session fits today. Emphasize quality reps, compound lifts, and stop before form breaks down."
        explanation.append("Current readiness supports building aerobic volume.")

    if prior_run and prior_run.duration_minutes >= 55:
        explanation.append("Your most recent run was already fairly substantial, so today's guidance stays grounded in recovery from that work.")
    if prior_run and prior_run.distance_miles >= 7:
        warnings.append("Recent mileage is coming from longer efforts, so be careful not to stack too much leg stress too quickly.")
    if three_day_load >= 150:
        warnings.append("Your last few days already carry meaningful load, so today's recommendation favors durability over forcing extra volume.")

    if sleep_low:
        warnings.append("Sleep dipped below 6.5 hours, which can reduce workout quality and recovery.")
    if elevated_rhr:
        warnings.append("Resting heart rate is elevated relative to your recent baseline.")
    if latest_metrics.recovery_score >= 80:
        explanation.append("Recovery is strong today, which supports quality if the broader load picture agrees.")
    elif latest_metrics.recovery_score <= 60:
        explanation.append("Recovery is only moderate today, so the recommendation leans more conservative even if your cardio feels ready for more.")

    if recovery_low or elevated_rhr:
        confidence = "medium"
    elif latest_metrics.recovery_score >= 67 and not strain_high:
        confidence = "high"
    else:
        confidence = "medium"

    return Recommendation(
        date=today_str,
        workout=workout,
        intensity=intensity,
        duration_minutes=duration,
        run_distance_miles=run_distance,
        run_pace_guidance=run_pace,
        lift_focus=lift_focus,
        lift_guidance=lift_guidance,
        recap=recap,
        explanation=explanation,
        explanation_sections={
            "overall": explanation[0] if explanation else "",
            "run": f"Today's run is set at {run_distance:.1f} miles to match your current training load and recovery picture.",
            "pace": f"The suggested pace band of {run_pace} is meant to keep the effort aligned with the day's goal rather than chasing speed.",
            "lift": lift_guidance,
            "recovery": f"Recovery is {latest_metrics.recovery_score}% with {latest_metrics.sleep_hours:.1f} hours of sleep and {latest_metrics.strain:.1f} strain, which shaped today's load.",
        },
        warnings=warnings,
        confidence=confidence,
    )
