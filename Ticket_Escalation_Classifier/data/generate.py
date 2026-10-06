"""Synthetic support-ticket dataset generator.

Structured features are drawn from class-conditional log-normal / count
distributions with deliberate overlap, so the task is NOT perfectly separable.
Also renders a free-text ticket body per row (used to test the text extractor).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "response_time", "error_count", "downtime", "users_affected",
    "previous_similar", "customer_priority", "ticket_age", "severity_indicators",
]
PRIORITIES = ["Low", "Medium", "High"]

SEVERITY_PHRASES = ["crash", "data loss", "outage", "cannot login", "corrupt"]
MILD_PHRASES = ["slow", "delay", "minor glitch", "cosmetic issue"]


def _render_text(row: dict, rng: np.random.Generator) -> str:
    parts = [f"The application took {row['response_time']:.1f} sec to respond."]
    if row["error_count"] > 0:
        parts.append(f"We saw {int(row['error_count'])} errors in the logs.")
    if row["users_affected"] > 0:
        parts.append(f"About {int(row['users_affected'])} users are affected.")
    if row["downtime"] > 0:
        parts.append(f"There was {int(round(row['downtime']))} min of downtime.")
    if row["previous_similar"] > 0:
        parts.append(f"This is similar to {int(row['previous_similar'])} previous tickets.")
    k = int(row["severity_indicators"])
    sev = list(rng.choice(SEVERITY_PHRASES, size=min(k, len(SEVERITY_PHRASES)), replace=False))
    mild = list(rng.choice(MILD_PHRASES, size=int(rng.integers(0, 2)), replace=False))
    words = sev + mild
    rng.shuffle(words)
    if words:
        parts.append("Symptoms: " + ", ".join(words) + ".")
    return " ".join(parts)


def generate(n: int = 8000, escalation_rate: float = 0.25, seed: int = 42,
             missing_rate: float = 0.03, with_text: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < escalation_rate).astype(int)

    def lognorm(mu0: float, mu1: float, sigma: float) -> np.ndarray:
        mu = np.where(y == 1, mu1, mu0)
        return np.exp(rng.normal(mu, sigma))

    # Wide sigmas -> substantial class overlap.
    response_time = lognorm(np.log(2.0), np.log(3.4), 0.80)                   # seconds
    error_count = np.round(lognorm(np.log(3.0), np.log(6.5), 1.0)).clip(0)
    downtime = np.round(lognorm(np.log(4.0), np.log(10.0), 1.2)).clip(0)     # minutes
    users_affected = np.round(lognorm(np.log(10.0), np.log(28.0), 1.15)).clip(1)
    previous_similar = rng.poisson(np.where(y == 1, 2.3, 1.5))
    ticket_age = np.round(lognorm(np.log(2.0), np.log(2.7), 0.9), 1)          # days

    p_high = np.where(y == 1, 0.38, 0.18)
    p_low = np.where(y == 1, 0.18, 0.34)
    u = rng.random(n)
    prio = np.where(u < p_low, "Low", np.where(u < 1 - p_high, "Medium", "High"))

    severity = np.clip(rng.poisson(np.where(y == 1, 1.2, 0.7)), 0, 5)

    df = pd.DataFrame({
        "response_time": response_time.round(2),
        "error_count": error_count.astype(int),
        "downtime": downtime.astype(int),
        "users_affected": users_affected.astype(int),
        "previous_similar": previous_similar,
        "customer_priority": prio,
        "ticket_age": ticket_age,
        "severity_indicators": severity,
    })

    if with_text:
        df["text"] = [_render_text(r, rng) for r in df.to_dict("records")]

    # Missing values in numeric structured features (never the label).
    if missing_rate > 0:
        for col in ["response_time", "error_count", "downtime", "users_affected", "ticket_age"]:
            mask = rng.random(n) < missing_rate
            df[col] = df[col].astype("float")
            df.loc[mask, col] = np.nan

    df["escalate"] = y
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--rate", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="E:/vs/ticket-gda/data/tickets.csv")
    a = ap.parse_args()
    df = generate(a.n, a.rate, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out} | escalation rate {df.escalate.mean():.1%}")
