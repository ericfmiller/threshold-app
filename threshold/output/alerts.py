"""Alert generation, email composition, and score history persistence.

Generates DCS-based alerts (HIGH CONVICTION, DIP-BUY, STRONG BUY),
composes HTML emails, and persists scoring results to JSON files.
"""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from threshold.config.schema import ThresholdConfig
from threshold.engine.scorer import ScoringResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def generate_scoring_alerts(
    scores: dict[str, ScoringResult],
    config: ThresholdConfig | None = None,
    portfolio_only: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate alerts from scoring results.

    Parameters:
        scores: {symbol: ScoringResult} from pipeline.
        config: ThresholdConfig for alert thresholds.
        portfolio_only: If provided, only alert on these tickers (not watchlist).

    Returns:
        List of alert dicts with: level, ticker, score, message, signal.
    """
    if config is None:
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()

    thresholds = config.alerts.thresholds
    strong_threshold = thresholds.get("dcs_strong", 80)
    conviction_threshold = thresholds.get("dcs_conviction", 70)
    buy_threshold = config.scoring.thresholds.buy_dip  # DCS >= 65

    alerts: list[dict[str, Any]] = []

    for ticker, result in scores.items():
        # Filter to portfolio tickers if specified
        if portfolio_only and ticker not in portfolio_only:
            continue

        dcs = result.get("dcs", 0)
        signal = result.get("dcs_signal", "")

        if dcs >= strong_threshold:
            alerts.append({
                "level": "STRONG BUY",
                "ticker": ticker,
                "score": dcs,
                "message": f"{ticker} DCS={dcs:.0f} — Strong dip-buy opportunity",
                "signal": signal,
            })
        elif dcs >= conviction_threshold:
            alerts.append({
                "level": "HIGH CONVICTION",
                "ticker": ticker,
                "score": dcs,
                "message": f"{ticker} DCS={dcs:.0f} — High conviction dip-buy",
                "signal": signal,
            })
        elif dcs >= buy_threshold:
            alerts.append({
                "level": "BUY DIP",
                "ticker": ticker,
                "score": dcs,
                "message": f"{ticker} DCS={dcs:.0f} — Standard dip-buy signal",
                "signal": signal,
            })

    # Sort by score descending
    alerts.sort(key=lambda a: a["score"], reverse=True)
    return alerts


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def build_scoring_email(
    scores: dict[str, ScoringResult],
    alerts: list[dict[str, Any]],
    vix_current: float,
    vix_regime: str,
    spy_pct: float,
    scored_count: int,
) -> tuple[str, str]:
    """Build HTML email for scoring results.

    Returns:
        (subject, body_html) tuple.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Subject
    if alerts:
        top_level = alerts[0]["level"]
        subject = f"[{top_level}] Weekly Scores — {date_str}"
    else:
        subject = f"Weekly Scores — {date_str}"

    # Build HTML body
    parts: list[str] = []
    parts.append("<html><body>")
    parts.append(f"<h2>Threshold Weekly Scores — {date_str}</h2>")
    parts.append(f"<p>VIX: {vix_current:.1f} ({vix_regime}), "
                 f"SPY vs 200d: {spy_pct:+.1%}, "
                 f"Tickers scored: {scored_count}</p>")

    # Alerts section
    if alerts:
        parts.append("<h3>DCS Alerts</h3>")
        parts.append("<table border='1' cellpadding='4' cellspacing='0'>")
        parts.append("<tr><th>Level</th><th>Ticker</th><th>DCS</th></tr>")
        for alert in alerts:
            color = "#006400" if "STRONG" in alert["level"] else "#228B22"
            parts.append(
                f"<tr><td style='color:{color}'><b>{alert['level']}</b></td>"
                f"<td>{alert['ticker']}</td>"
                f"<td>{alert['score']:.0f}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>No DCS alerts this run.</p>")

    # Top 5 scores
    top_scores = sorted(
        ((t, r.get("dcs", 0)) for t, r in scores.items() if "dcs" in r),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    if top_scores:
        parts.append("<h3>Top 5 DCS</h3>")
        parts.append("<table border='1' cellpadding='4' cellspacing='0'>")
        parts.append("<tr><th>Ticker</th><th>DCS</th><th>Signal</th></tr>")
        for ticker, dcs in top_scores:
            sig = scores[ticker].get("dcs_signal", "")
            parts.append(f"<tr><td>{ticker}</td><td>{dcs:.0f}</td><td>{sig}</td></tr>")
        parts.append("</table>")

    parts.append("<hr><p><em>Generated by Threshold</em></p>")
    parts.append("</body></html>")

    return subject, "\n".join(parts)


def send_email(
    subject: str,
    body_html: str,
    config: ThresholdConfig,
) -> bool:
    """Send HTML email via SMTP.

    Returns True if sent, False if failed or not configured.
    """
    email_cfg = config.alerts.email
    if not email_cfg.to or not email_cfg.from_addr or not email_cfg.app_password:
        logger.debug("Email not configured — skipping")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = email_cfg.from_addr
        msg["To"] = email_cfg.to

        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(email_cfg.smtp_server, email_cfg.smtp_port) as server:
            server.starttls()
            server.login(email_cfg.from_addr, email_cfg.app_password)
            server.send_message(msg)

        logger.info("Email sent: %s", subject)
        return True

    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Score history persistence
# ---------------------------------------------------------------------------

def save_score_history(
    scores: dict[str, ScoringResult],
    vix_current: float,
    vix_regime: str,
    spy_pct: float = 0.0,
    breadth_pct: float = 0.0,
    effective_bets: float = 0.0,
    market_regime_score: float = 0.0,
    run_metadata: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Persist scoring results to a JSON file.

    Parameters:
        scores: {symbol: ScoringResult} from pipeline.
        vix_current: Current VIX level.
        vix_regime: VIX regime string.
        spy_pct: SPY % from 200d SMA.
        breadth_pct: Market breadth %.
        effective_bets: Portfolio effective bets.
        market_regime_score: MR sub-score value.
        run_metadata: Optional RunTracker.to_dict().
        output_dir: Override output directory.

    Returns:
        Path to saved JSON file.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = Path("~/.threshold/history").expanduser()
    else:
        output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output dict
    output: dict[str, Any] = {
        "_metadata": {
            "schema_version": 12,
            "scored_at": datetime.now().isoformat(),
            "vix_current": vix_current,
            "vix_regime": vix_regime,
            "spy_pct_from_200d": round(spy_pct, 4),
            "breadth_pct": round(breadth_pct, 4),
            "effective_bets": round(effective_bets, 2),
            "market_regime_score": round(market_regime_score, 4),
        },
        "scores": {},
    }

    if run_metadata:
        output["run_metadata"] = run_metadata

    # Per-ticker scores
    for symbol, result in scores.items():
        ticker_entry: dict[str, Any] = {
            "dcs": result.get("dcs", 0),
            "dcs_signal": result.get("dcs_signal", ""),
            "sub_scores": result.get("sub_scores", {}),
            "is_etf": result.get("is_etf", False),
        }

        # Include technicals if present
        technicals = result.get("technicals")
        if technicals:
            ticker_entry["technicals"] = dict(technicals)

        # Include sell flags
        sell_flags = result.get("sell_flags")
        if sell_flags:
            ticker_entry["sell_flags"] = sell_flags

        # Include reversal signals
        for sig_key in ("reversal_confirmed", "bottom_turning",
                        "rsi_bullish_divergence", "quant_freshness_warning"):
            if sig_key in result:
                ticker_entry[sig_key] = result[sig_key]

        # Include falling knife cap data
        if "falling_knife_cap" in result:
            ticker_entry["falling_knife_cap"] = result["falling_knife_cap"]

        # Include revision momentum data
        if "revision_momentum" in result:
            ticker_entry["revision_momentum"] = result["revision_momentum"]

        # Include trend score
        if "trend_score" in result:
            ticker_entry["trend_score"] = result["trend_score"]

        # Include holdings/watchlist tags
        if "is_holding" in result:
            ticker_entry["is_holding"] = result["is_holding"]
        if "is_watchlist" in result:
            ticker_entry["is_watchlist"] = result["is_watchlist"]

        output["scores"][symbol] = ticker_entry

    # Write
    filepath = output_dir / f"weekly_scores_{date_str}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Score history saved: %s", filepath)
    return str(filepath)


def load_previous_scores(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the most recent score history file.

    Returns:
        {symbol: score_data} dict, or empty dict if not found.
    """
    if output_dir is None:
        output_dir = Path("~/.threshold/history").expanduser()
    else:
        output_dir = Path(output_dir).expanduser()

    if not output_dir.exists():
        return {}

    # Find most recent
    files = sorted(output_dir.glob("weekly_scores_*.json"), reverse=True)
    if not files:
        return {}

    try:
        with open(files[0]) as f:
            data = json.load(f)
        return data.get("scores", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load previous scores: %s", e)
        return {}


def load_grade_history(
    max_weeks: int = 8,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load up to N most recent score history files for revision momentum.

    Returns:
        List of parsed JSON objects (most recent first).
    """
    if output_dir is None:
        output_dir = Path("~/.threshold/history").expanduser()
    else:
        output_dir = Path(output_dir).expanduser()

    if not output_dir.exists():
        return []

    files = sorted(output_dir.glob("weekly_scores_*.json"), reverse=True)[:max_weeks]
    history: list[dict[str, Any]] = []

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            history.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return history
