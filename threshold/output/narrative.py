"""Markdown narrative report generator.

Produces a human-readable Markdown report organized by the Decision
Hierarchy.  Each section pulls data from ``PipelineResult`` and
optional allocation / drawdown enrichments.

Sections
--------
1. Header & Run Summary
2. Macro Backdrop (VIX, SPY, breadth)
3. Dip-Buy Opportunities (DCS >= 65)
2.5. Deployment Gate 3 — Parabolic Filter
4. Falling Knife Alerts
5. Watch Zone (DCS 50-64)
6. Reversal Signals
7. Sell Criteria & Flags
6b. Active Grace Periods
6c. Exemption Status
8. Drawdown Defense Composition
9. Correlation & Diversification
10. Sector Exposure
11. War Chest Status
12. Big Picture / Action Items
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from threshold.engine.pipeline import PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def _dcs_emoji(dcs: float) -> str:
    """Return a simple text indicator for a DCS score."""
    if dcs >= 80:
        return "**[STRONG]**"
    elif dcs >= 70:
        return "**[HC]**"
    elif dcs >= 65:
        return "[BUY]"
    elif dcs >= 50:
        return "[WATCH]"
    else:
        return "[WEAK]"


def _vix_emoji(regime: str) -> str:
    """Return an indicator for VIX regime."""
    return {
        "COMPLACENT": "LOW",
        "NORMAL": "NORMAL",
        "FEAR": "**FEAR**",
        "PANIC": "**PANIC**",
    }.get(regime, regime)


def _format_sell_flags(flags: list[str]) -> str:
    """Format sell flags as a comma-separated string."""
    if not flags:
        return "-"
    return ", ".join(flags)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(result: PipelineResult) -> str:
    """Section 0: Report header."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_scored = len(result.scores)
    return f"""# Threshold Scoring Report — {date_str}

**Run ID:** `{result.run_id}`
**Tickers Scored:** {n_scored}
**VIX:** {result.vix_current:.1f} ({_vix_emoji(result.vix_regime)})
**SPY vs 200d:** {_pct(result.spy_pct_from_200d)} ({'Above' if result.spy_above_200d else '**BELOW**'})
**Breadth:** {_pct(result.breadth_pct, 0)} above 200d
**Effective Bets:** {result.correlation.effective_bets:.1f}

---
"""


def _build_macro_section(result: PipelineResult) -> str:
    """Section 1: Macro Backdrop."""
    vix = result.vix_current
    regime = result.vix_regime
    spy_pct = result.spy_pct_from_200d
    breadth = result.breadth_pct

    lines = [
        "## 1. Macro Backdrop",
        "",
        "| Indicator | Value | Status |",
        "|-----------|-------|--------|",
        f"| VIX | {vix:.1f} | {_vix_emoji(regime)} |",
        f"| SPY vs 200d SMA | {_pct(spy_pct)} | {'Above' if result.spy_above_200d else '**BELOW**'} |",
        f"| Breadth (% > 200d) | {_pct(breadth, 0)} | {'Healthy' if breadth > 0.5 else '**Weak**'} |",
        f"| Market Regime Score | {result.market_regime_score:.2f} | {'Risk-On' if result.market_regime_score >= 0.5 else '**Risk-Off**'} |",
        "",
    ]

    # Regime commentary
    if regime in ("FEAR", "PANIC"):
        lines.append(f"> **Elevated volatility regime ({regime}).** D-5 modifiers active: "
                     f"HEDGE +5, DEFENSIVE +3, CYCLICAL -3, AMPLIFIER -5.")
        lines.append("")
    elif regime == "COMPLACENT":
        lines.append("> VIX is complacent. Half-size new positions per deployment discipline.")
        lines.append("")

    return "\n".join(lines)


def _build_dipbuy_section(result: PipelineResult) -> str:
    """Section 2: Dip-Buy Opportunities (DCS >= 65)."""
    buys = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("dcs", 0) >= 65
    ]
    buys.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    lines = ["## 2. Dip-Buy Opportunities", ""]

    if not buys:
        lines.append("No tickers at DCS >= 65 this run.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | DCS | Signal | RSI | Sub-Scores | Flags |")
    lines.append("|--------|-----|--------|-----|------------|-------|")

    for ticker, r in buys:
        dcs = r.get("dcs", 0)
        signal = r.get("dcs_signal", "")
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        sub = r.get("sub_scores", {})
        sub_str = " ".join(f"{k}:{v:.0f}" for k, v in sorted(sub.items())
                           if isinstance(v, (int, float)))
        flags = _format_sell_flags(r.get("sell_flags", []))
        lines.append(f"| **{ticker}** | {dcs:.0f} | {signal} | {rsi:.0f} | {sub_str} | {flags} |")

    lines.append("")
    return "\n".join(lines)


def _build_falling_knife_section(result: PipelineResult) -> str:
    """Section 3: Falling Knife Alerts."""
    knives = []
    for ticker, r in result.scores.items():
        fk = r.get("falling_knife_cap")
        if fk and fk.get("cap_applied"):
            knives.append((ticker, r, fk))

    lines = ["## 3. Falling Knife Alerts", ""]

    if not knives:
        lines.append("No falling knife caps triggered.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | Original DCS | Capped DCS | Defense | Reason |")
    lines.append("|--------|-------------|------------|---------|--------|")

    for ticker, r, fk in knives:
        orig = fk.get("original_dcs", 0)
        capped = r.get("dcs", 0)
        defense = fk.get("classification", "")
        lines.append(f"| {ticker} | {orig:.0f} | {capped:.0f} | {defense} | Cap applied |")

    lines.append("")
    lines.append("> Falling knife filter limits DCS for tickers in steep downtrends. "
                 "Defense class determines the cap ceiling.")
    lines.append("")
    return "\n".join(lines)


def _build_watch_section(result: PipelineResult) -> str:
    """Section 4: Watch Zone (DCS 50-64)."""
    watch = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if 50 <= r.get("dcs", 0) < 65
    ]
    watch.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    lines = ["## 4. Watch Zone (DCS 50-64)", ""]

    if not watch:
        lines.append("No tickers in the watch zone.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | DCS | RSI | Trend | Notes |")
    lines.append("|--------|-----|-----|-------|-------|")

    for ticker, r in watch[:15]:
        dcs = r.get("dcs", 0)
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        trend = r.get("trend_score", 0)
        trend_str = "Up" if trend > 0.6 else ("Down" if trend < 0.4 else "Flat")
        notes_parts = []
        if r.get("rsi_bullish_divergence"):
            notes_parts.append("RSI Div")
        if r.get("bottom_turning"):
            notes_parts.append("Bottom Turning")
        notes = ", ".join(notes_parts) or "-"
        lines.append(f"| {ticker} | {dcs:.0f} | {rsi:.0f} | {trend_str} | {notes} |")

    if len(watch) > 15:
        lines.append(f"| ... | | | | +{len(watch)-15} more |")

    lines.append("")
    return "\n".join(lines)


def _build_reversal_section(result: PipelineResult) -> str:
    """Section 5: Reversal Signals."""
    reversals = []
    bottoms = []
    divergences = []
    quant_checks = []

    for ticker, r in result.scores.items():
        if r.get("reversal_confirmed"):
            reversals.append((ticker, r))
        if r.get("bottom_turning"):
            bottoms.append((ticker, r))
        if r.get("rsi_bullish_divergence"):
            divergences.append((ticker, r))
        if r.get("quant_freshness_warning"):
            quant_checks.append((ticker, r))

    total = len(reversals) + len(bottoms) + len(divergences) + len(quant_checks)

    lines = ["## 5. Reversal Signals", ""]

    if total == 0:
        lines.append("No reversal signals detected.")
        lines.append("")
        return "\n".join(lines)

    if reversals:
        lines.append("### Reversal Confirmed (DCS >= 65 + BB breach)")
        for ticker, r in reversals:
            dcs = r.get("dcs", 0)
            lines.append(f"- **{ticker}** DCS={dcs:.0f} — Full-size deployment signal")
        lines.append("")

    if bottoms:
        lines.append("### Bottom Turning (MACD rising + RSI < 30 + Q3+)")
        for ticker, r in bottoms:
            dcs = r.get("dcs", 0)
            lines.append(f"- **{ticker}** DCS={dcs:.0f} — Watchlist alert for entry timing")
        lines.append("")

    if divergences:
        lines.append("### RSI Bullish Divergence")
        for ticker, r in divergences:
            dcs = r.get("dcs", 0)
            lines.append(f"- **{ticker}** DCS={dcs:.0f} — +3 boost applied (walk-forward stable)")
        lines.append("")

    if quant_checks:
        lines.append("### Quant Freshness Warning")
        for ticker, r in quant_checks:
            dcs = r.get("dcs", 0)
            lines.append(f"- {ticker} DCS={dcs:.0f} — Verify quant score is current")
        lines.append("")

    return "\n".join(lines)


def _build_sell_criteria_section(result: PipelineResult) -> str:
    """Section 6: Sell Criteria & Flags."""
    flagged = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("sell_flags")
    ]
    flagged.sort(key=lambda x: len(x[1].get("sell_flags", [])), reverse=True)

    lines = ["## 6. Sell Criteria & Flags", ""]

    if not flagged:
        lines.append("No sell flags triggered.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**{len(flagged)}** tickers have sell flags:")
    lines.append("")
    lines.append("| Ticker | DCS | Flags | RSI | SMA Status |")
    lines.append("|--------|-----|-------|-----|------------|")

    for ticker, r in flagged:
        dcs = r.get("dcs", 0)
        flags = _format_sell_flags(r.get("sell_flags", []))
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        pct_200d = tech.get("pct_from_200d", 0)
        sma_status = f"{_pct(pct_200d)}" if pct_200d else "-"
        lines.append(f"| {ticker} | {dcs:.0f} | {flags} | {rsi:.0f} | {sma_status} |")

    lines.append("")
    lines.append("> **Rule:** Any 2 sell criteria = review required. "
                 "Grace period: 180-day hold window for weakening positions.")
    lines.append("")
    return "\n".join(lines)


def _build_grace_period_section(
    active_grace_periods: list[dict[str, Any]] | None,
) -> str:
    """Section 6b: Active Grace Periods."""
    lines = ["## 6b. Active Grace Periods", ""]

    if not active_grace_periods:
        lines.append("No active grace periods.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**{len(active_grace_periods)}** active grace period(s):")
    lines.append("")
    lines.append("| Ticker | Tier | Days Left | Reason | Expires |")
    lines.append("|--------|------|-----------|--------|---------|")

    for gp in active_grace_periods:
        symbol = gp.get("symbol", "")
        tier = gp.get("tier", 0)
        days = gp.get("days_remaining", 0)
        reason = gp.get("reason", "")
        expires = gp.get("expires_at", "")
        lines.append(
            f"| **{symbol}** | {tier}d | {days} | {reason} | {expires} |"
        )

    lines.append("")
    lines.append(
        "> Grace periods soften sell signals: SELL -> HOLD during the hold "
        "window. Ticker still tracked and DCS computed."
    )
    lines.append("")
    return "\n".join(lines)


def _build_exemption_section(
    exempt_tickers: dict[str, Any] | None,
) -> str:
    """Section 6c: Exemption Status."""
    lines = ["## 6c. Exemption Status", ""]

    if not exempt_tickers:
        lines.append("No tickers with exemptions.")
        lines.append("")
        return "\n".join(lines)

    crypto = []
    cash = []
    expired = []

    for ticker, exemption in exempt_tickers.items():
        # Handle both ExemptionResult objects and dicts
        ex_type = getattr(exemption, "exemption_type", "")
        is_expired = getattr(exemption, "is_expired", False)
        expires_at = getattr(exemption, "expires_at", "")

        if is_expired:
            expired.append((ticker, expires_at))
        elif ex_type == "crypto_halving":
            crypto.append((ticker, expires_at))
        elif ex_type == "cash":
            cash.append(ticker)

    if crypto:
        lines.append("### Crypto Halving Cycle (exempt from sell rules)")
        for ticker, expiry in crypto:
            expiry_note = f" — expires {expiry}" if expiry else ""
            lines.append(f"- **{ticker}**{expiry_note}")
        lines.append("")

    if cash:
        lines.append("### Cash / War Chest (permanent exemption)")
        for ticker in cash:
            lines.append(f"- **{ticker}**")
        lines.append("")

    if expired:
        lines.append("### Expired Exemptions")
        for ticker, expiry in expired:
            lines.append(f"- {ticker} — exemption expired {expiry}")
        lines.append("")

    return "\n".join(lines)


def _build_gate3_section(result: PipelineResult) -> str:
    """Section 2.5: Deployment Gate 3 — Parabolic Filter."""
    gate3_tickers = []
    for ticker, r in result.scores.items():
        dcs = r.get("dcs", 0)
        if dcs < 65:
            continue
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        ret_8w = tech.get("ret_8w", 0)

        # Check signal board for parabolic warning
        has_gate3 = False
        sizing = "FULL"
        for sig in r.get("signal_board", []):
            if sig.get("legacy_prefix", "").startswith("GATE3:"):
                has_gate3 = True
                sizing = sig.get("metadata", {}).get("sizing", "WAIT")
                break

        gate3_tickers.append((ticker, dcs, rsi, ret_8w, sizing, has_gate3))

    lines = ["## 2.5. Deployment Discipline — Gate 3", ""]

    if not gate3_tickers:
        lines.append("No buy candidates (DCS >= 65) this run.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | DCS | RSI | 8w Return | Sizing | Status |")
    lines.append("|--------|-----|-----|-----------|--------|--------|")

    for ticker, dcs, rsi, ret_8w, sizing, has_gate3 in gate3_tickers:
        status = f"**{sizing}**" if has_gate3 else "PASS"
        lines.append(
            f"| {ticker} | {dcs:.0f} | {rsi:.0f} | {_pct(ret_8w)} "
            f"| {sizing} | {status} |"
        )

    lines.append("")

    blocked = [t for t in gate3_tickers if t[5]]
    if blocked:
        lines.append(
            f"> **{len(blocked)} ticker(s) have Gate 3 deployment restrictions.** "
            "Do not deploy at full size until RSI < 80 or 8w return consolidates."
        )
        lines.append("")

    return "\n".join(lines)


def _build_drawdown_section(
    classifications: dict[str, str] | None,
) -> str:
    """Section 7: Drawdown Defense Composition."""
    lines = ["## 7. Drawdown Defense", ""]

    if not classifications:
        lines.append("No drawdown classification data available.")
        lines.append("")
        return "\n".join(lines)

    # Count by class
    class_order = ["HEDGE", "DEFENSIVE", "MODERATE", "CYCLICAL", "AMPLIFIER"]
    counts: dict[str, list[str]] = {c: [] for c in class_order}
    for ticker, cls in classifications.items():
        if cls in counts:
            counts[cls].append(ticker)

    total = len(classifications)
    offense = len(counts.get("CYCLICAL", [])) + len(counts.get("AMPLIFIER", []))
    defense = len(counts.get("HEDGE", [])) + len(counts.get("DEFENSIVE", []))

    lines.append(f"**Total:** {total} tickers classified | "
                 f"**Defense:** {defense} ({defense/total*100:.0f}%) | "
                 f"**Offense:** {offense} ({offense/total*100:.0f}%)")
    lines.append("")

    lines.append("| Class | Count | Tickers |")
    lines.append("|-------|-------|---------|")
    for cls in class_order:
        tickers = counts[cls]
        ticker_str = ", ".join(sorted(tickers)) if tickers else "-"
        lines.append(f"| {cls} | {len(tickers)} | {ticker_str} |")

    lines.append("")

    if offense / total > 0.5 if total > 0 else False:
        lines.append(f"> **Portfolio is {offense/total*100:.0f}% offense** "
                     f"(cyclical + amplifier). Consider defensive rotation "
                     f"if VIX enters FEAR/PANIC regime.")
        lines.append("")

    return "\n".join(lines)


def _build_correlation_section(result: PipelineResult) -> str:
    """Section 8: Correlation & Diversification."""
    corr = result.correlation
    lines = ["## 8. Correlation & Diversification", ""]

    lines.append(f"**Effective Bets:** {corr.effective_bets:.1f} | "
                 f"**Tickers:** {corr.n_tickers}")
    lines.append("")

    if corr.is_concentrated:
        lines.append("> **Concentrated portfolio** — effective bets below threshold. "
                     "New buy candidates with >0.70 correlation to existing holdings "
                     "will receive concentration warnings.")
        lines.append("")

    # High-correlation pairs
    if corr.high_corr_pairs:
        lines.append("### High-Correlation Pairs (> 0.80)")
        lines.append("")
        lines.append("| Pair | Correlation |")
        lines.append("|------|------------|")
        for a, b, c in corr.high_corr_pairs[:10]:
            lines.append(f"| {a} / {b} | {c:.3f} |")
        lines.append("")

    # Concentration warnings
    if result.concentration_warnings:
        lines.append("### Concentration Warnings")
        lines.append("")
        for w in result.concentration_warnings:
            lines.append(f"- **{w['ticker']}** correlated with "
                         f"{w['correlated_with']} ({w['correlation']:.3f})")
        lines.append("")

    return "\n".join(lines)


def _build_sector_section(
    result: PipelineResult,
    ticker_sectors: dict[str, str] | None = None,
) -> str:
    """Section 9: Sector Exposure."""
    lines = ["## 9. Sector Exposure", ""]

    if not ticker_sectors:
        lines.append("No sector mapping available.")
        lines.append("")
        return "\n".join(lines)

    # Group by sector
    sector_stats: dict[str, list[tuple[str, float]]] = {}
    for ticker, r in result.scores.items():
        sector = ticker_sectors.get(ticker, "Other")
        dcs = r.get("dcs", 0)
        sector_stats.setdefault(sector, []).append((ticker, dcs))

    lines.append("| Sector | Count | Avg DCS | Top Ticker |")
    lines.append("|--------|-------|---------|------------|")

    for sector in sorted(sector_stats.keys()):
        tickers = sector_stats[sector]
        count = len(tickers)
        avg_dcs = sum(d for _, d in tickers) / count if count > 0 else 0
        top = max(tickers, key=lambda x: x[1])
        lines.append(f"| {sector} | {count} | {avg_dcs:.0f} | {top[0]} ({top[1]:.0f}) |")

    lines.append("")
    return "\n".join(lines)


def _build_war_chest_section(
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    vix_regime: str = "NORMAL",
) -> str:
    """Section 10: War Chest Status."""
    lines = ["## 10. War Chest Status", ""]

    surplus = war_chest_pct - war_chest_target
    status = "ADEQUATE" if surplus >= 0 else "**BELOW TARGET**"

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| VIX Regime | {_vix_emoji(vix_regime)} |")
    lines.append(f"| Target | {_pct(war_chest_target)} |")
    lines.append(f"| Actual | {_pct(war_chest_pct)} |")
    lines.append(f"| Surplus/Gap | {_pct(surplus)} |")
    lines.append(f"| Status | {status} |")
    lines.append("")

    if surplus < 0:
        lines.append(f"> **War chest below {_vix_emoji(vix_regime)} target.** "
                     f"Consider reducing position sizes or deferring new deployments "
                     f"until cash reserves reach {_pct(war_chest_target)}.")
        lines.append("")

    return "\n".join(lines)


def _build_action_items(result: PipelineResult) -> str:
    """Section 11: Big Picture / Action Items."""
    lines = ["## 11. Action Items", ""]

    actions: list[str] = []

    # Count buy signals
    strong_buys = sum(1 for r in result.scores.values() if r.get("dcs", 0) >= 80)
    hc_buys = sum(1 for r in result.scores.values() if 70 <= r.get("dcs", 0) < 80)
    dip_buys = sum(1 for r in result.scores.values() if 65 <= r.get("dcs", 0) < 70)

    if strong_buys > 0:
        actions.append(f"- **{strong_buys} STRONG BUY** signal(s) — full size + lean in")
    if hc_buys > 0:
        actions.append(f"- **{hc_buys} HIGH CONVICTION** signal(s) — full size deployment")
    if dip_buys > 0:
        actions.append(f"- {dip_buys} BUY DIP signal(s) — standard deployment")

    # Sell flags
    flagged = sum(1 for r in result.scores.values() if r.get("sell_flags"))
    if flagged > 0:
        actions.append(f"- **{flagged} tickers with sell flags** — review required")

    # Reversal signals
    rev_count = sum(1 for r in result.scores.values() if r.get("reversal_confirmed"))
    if rev_count > 0:
        actions.append(f"- {rev_count} reversal confirmed — high-confidence entry point(s)")

    # Concentration
    if result.concentration_warnings:
        actions.append(f"- {len(result.concentration_warnings)} concentration warning(s) — "
                       f"check correlation before adding")

    # VIX regime
    if result.vix_regime in ("FEAR", "PANIC"):
        actions.append(f"- **{result.vix_regime} regime** — D-5 modifiers active, "
                       f"lean into hedges/defensives")
    elif result.vix_regime == "COMPLACENT":
        actions.append("- Complacent VIX — use half-size for new positions")

    if not actions:
        actions.append("- No immediate action items. Monitor positions at next review.")

    lines.extend(actions)
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by Threshold v0.4.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_narrative(
    result: PipelineResult,
    *,
    ticker_sectors: dict[str, str] | None = None,
    drawdown_classifications: dict[str, str] | None = None,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    output_dir: str | Path | None = None,
) -> str:
    """Generate the full Markdown narrative report.

    Parameters:
        result: PipelineResult from run_scoring_pipeline().
        ticker_sectors: {symbol: sector_name} mapping.
        drawdown_classifications: {symbol: class_name} from backtest.
        war_chest_pct: Current war chest % of portfolio.
        war_chest_target: VIX-regime target %.
        output_dir: Directory for output file.

    Returns:
        Path to generated Markdown file.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = Path("~/.threshold/narratives").expanduser()
    else:
        output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sections in order
    parts: list[str] = [
        _build_header(result),
        _build_macro_section(result),
        _build_dipbuy_section(result),
        _build_gate3_section(result),
        _build_falling_knife_section(result),
        _build_watch_section(result),
        _build_reversal_section(result),
        _build_sell_criteria_section(result),
        _build_grace_period_section(result.active_grace_periods),
        _build_exemption_section(result.exempt_tickers),
        _build_drawdown_section(drawdown_classifications),
        _build_correlation_section(result),
        _build_sector_section(result, ticker_sectors),
        _build_war_chest_section(war_chest_pct, war_chest_target, result.vix_regime),
        _build_action_items(result),
    ]

    content = "\n".join(parts)

    # Write file
    filepath = output_dir / f"narrative_{date_str}.md"
    with open(filepath, "w") as f:
        f.write(content)

    logger.info("Narrative generated: %s", filepath)
    return str(filepath)
