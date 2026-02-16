"""Markdown narrative report generator.

Produces a human-readable Markdown report organized by the Decision
Hierarchy.  Each section pulls data from ``PipelineResult`` and
optional allocation / drawdown enrichments.

Sections (23 total)
--------------------
 1. Header & Run Summary
 2. Macro Backdrop (VIX, SPY, breadth)
 3. Dip-Buy Opportunities — split Holdings / Watchlist
 2.5. Deployment Gate 3 — Parabolic Filter
 4. Falling Knife Alerts (non-defensive only)
 5. Hedges/Defensives in Downtrend (D-7 rule)
 6. Watch Zone (DCS 50-64)
 7. Bitcoin & Crypto (halving cycle)
 8. Reversal Signals (REV, BTM, DIV, QC)
 9. Sub-Score Driver Analysis (MQ/FQ/TO/MR/VC)
10. Relative Strength vs SPY (Antonacci dual momentum)
11. EPS Revision Momentum (Novy-Marx 2015)
12. OBV Divergence Analysis (Granville 1963)
13. Sell Criteria & Flags
14. Active Grace Periods
15. Exemption Status
16. Drawdown Defense (count + dollar-weighted)
17. Correlation & Diversification
18. Sector Exposure
19. War Chest Status (VIX-regime target)
20. Per-Account Holdings Health
21. Action Items
22. Quick Reference Summary
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


def _reversal_badges(r: dict[str, Any]) -> str:
    """Return space-separated reversal signal badges."""
    badges = []
    if r.get("reversal_confirmed"):
        badges.append("[REV]")
    if r.get("bottom_turning"):
        badges.append("[BTM]")
    if r.get("rsi_bullish_divergence"):
        badges.append("[DIV]")
    if r.get("quant_freshness_warning"):
        badges.append("[QC]")
    return " ".join(badges)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_header(result: PipelineResult) -> str:
    """Section 0: Report header."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_scored = len(result.scores)
    n_held = len(result.held_symbols)
    n_watchlist = n_scored - n_held
    return f"""# Threshold Scoring Report — {date_str}

**Run ID:** `{result.run_id}`
**Tickers Scored:** {n_scored} ({n_held} holdings, {n_watchlist} watchlist)
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


def _build_dipbuy_section(
    result: PipelineResult,
    held_symbols: set[str] | None = None,
) -> str:
    """Section 2: Dip-Buy Opportunities (DCS >= 65) split by Holdings / Watchlist.

    Research basis: DCS >= 65 validated at 58.9% win rate (124K obs backtest),
    bootstrap CI [57.7%, 60.1%], walk-forward stable (cal 60.8%, val 57.5%).
    """
    held = held_symbols or set()
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

    # Split by holdings vs watchlist
    holdings = [(t, r) for t, r in buys if t in held]
    watchlist = [(t, r) for t, r in buys if t not in held]

    for group_name, group in [("Portfolio Holdings", holdings), ("Watchlist Candidates", watchlist)]:
        if not group:
            continue
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| Ticker | DCS | Signal | RSI | % from 200d | Badges | Sub-Scores | Flags |")
        lines.append("|--------|-----|--------|-----|-------------|--------|------------|-------|")

        for ticker, r in group:
            dcs = r.get("dcs", 0)
            signal = r.get("dcs_signal", "")
            tech = r.get("technicals", {})
            rsi = tech.get("rsi_14", 0)
            pct_200d = tech.get("pct_from_200d", 0)
            badges = _reversal_badges(r)
            sub = r.get("sub_scores", {})
            sub_str = " ".join(f"{k}:{v:.0f}" for k, v in sorted(sub.items())
                               if isinstance(v, (int, float)))
            flags = _format_sell_flags(r.get("sell_flags", []))
            lines.append(
                f"| **{ticker}** | {dcs:.0f} | {signal} | {rsi:.0f} | "
                f"{_pct(pct_200d)} | {badges or '-'} | {sub_str} | {flags} |"
            )

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


def _build_falling_knife_section(
    result: PipelineResult,
    drawdown_classifications: dict[str, str] | None = None,
) -> str:
    """Section 3: Falling Knife Alerts (non-defensive tickers only).

    Defensive/hedge tickers in downtrend get their own section (D-7 rule).
    """
    dd = drawdown_classifications or {}
    defensive_classes = {"HEDGE", "DEFENSIVE"}
    knives = []
    for ticker, r in result.scores.items():
        fk = r.get("falling_knife_cap")
        if fk and fk.get("cap_applied"):
            ticker_class = dd.get(ticker, "")
            if ticker_class not in defensive_classes:
                knives.append((ticker, r, fk))

    lines = ["## 3. Falling Knife Alerts", ""]

    if not knives:
        lines.append("No falling knife caps triggered (non-defensive tickers).")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | Original DCS | Capped DCS | Defense | Reason |")
    lines.append("|--------|-------------|------------|---------|--------|")

    for ticker, r, fk in knives:
        orig = fk.get("original_dcs", 0)
        capped = r.get("dcs", 0)
        defense = fk.get("classification", dd.get(ticker, ""))
        lines.append(f"| {ticker} | {orig:.0f} | {capped:.0f} | {defense} | Cap applied |")

    lines.append("")
    lines.append("> Falling knife filter limits DCS for tickers in steep downtrends. "
                 "Defense class determines the cap ceiling.")
    lines.append("")
    return "\n".join(lines)


def _build_hedge_downtrend_section(
    result: PipelineResult,
    drawdown_classifications: dict[str, str] | None = None,
) -> str:
    """Section 4: Hedges/Defensives in Downtrend (D-7 rule).

    Research: D-7 rule — HEDGE/DEFENSIVE assets get 270-day grace during
    drawdowns because they have counter-cyclical value (DC < 0 means they
    gain when SPY falls). Selling hedges during drawdowns removes insurance
    exactly when it's needed.
    """
    dd = drawdown_classifications or {}
    defensive_classes = {"HEDGE", "DEFENSIVE"}

    hedges_down = []
    for ticker, r in result.scores.items():
        ticker_class = dd.get(ticker, "")
        if ticker_class not in defensive_classes:
            continue
        fk = r.get("falling_knife_cap")
        tech = r.get("technicals", {})
        pct_200d = tech.get("pct_from_200d", 0)
        # Include if falling knife capped OR significantly below 200d
        if (fk and fk.get("cap_applied")) or pct_200d < -0.03:
            hedges_down.append((ticker, r, ticker_class))

    lines = ["## 4. Hedges & Defensives in Downtrend", ""]

    if not hedges_down:
        lines.append("No hedge/defensive assets currently in downtrend.")
        lines.append("")
        return "\n".join(lines)

    lines.append("> **D-7 Rule:** HEDGE/DEFENSIVE assets provide drawdown insurance. "
                 "Consider extended grace period (270d) before selling. Counter-cyclical "
                 "value means they typically gain when SPY falls.")
    lines.append("")
    lines.append("| Ticker | DCS | Class | % from 200d | RSI | Sell Flags |")
    lines.append("|--------|-----|-------|-------------|-----|------------|")

    for ticker, r, cls in hedges_down:
        dcs = r.get("dcs", 0)
        tech = r.get("technicals", {})
        pct_200d = tech.get("pct_from_200d", 0)
        rsi = tech.get("rsi_14", 0)
        flags = _format_sell_flags(r.get("sell_flags", []))
        lines.append(f"| {ticker} | {dcs:.0f} | {cls} | {_pct(pct_200d)} | {rsi:.0f} | {flags} |")

    lines.append("")
    return "\n".join(lines)


def _build_watch_section(result: PipelineResult) -> str:
    """Section 5: Watch Zone (DCS 50-64)."""
    watch = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if 50 <= r.get("dcs", 0) < 65
    ]
    watch.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    lines = ["## 5. Watch Zone (DCS 50-64)", ""]

    if not watch:
        lines.append("No tickers in the watch zone.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | DCS | RSI | Trend | Badges | Notes |")
    lines.append("|--------|-----|-----|-------|--------|-------|")

    for ticker, r in watch[:15]:
        dcs = r.get("dcs", 0)
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        trend = r.get("trend_score", 0)
        trend_str = "Up" if trend > 0.6 else ("Down" if trend < 0.4 else "Flat")
        badges = _reversal_badges(r)
        held = "Held" if r.get("is_holding") else "WL"
        lines.append(f"| {ticker} | {dcs:.0f} | {rsi:.0f} | {trend_str} | {badges or '-'} | {held} |")

    if len(watch) > 15:
        lines.append(f"| ... | | | | | +{len(watch)-15} more |")

    lines.append("")
    return "\n".join(lines)


def _build_crypto_section(
    result: PipelineResult,
) -> str:
    """Section 6: Bitcoin & Crypto (halving cycle analysis).

    Research: 4-year halving cycle dynamics override standard momentum
    signals for Bitcoin and crypto-correlated assets. Golden cross
    (50d > 200d SMA) confirms cycle bottom.
    """
    crypto_tickers = []
    for ticker, ex in (result.exempt_tickers or {}).items():
        ex_type = getattr(ex, "exemption_type", "")
        if ex_type == "crypto_halving":
            r = result.scores.get(ticker, {})
            crypto_tickers.append((ticker, r, ex))

    lines = ["## 6. Bitcoin & Crypto", ""]

    if not crypto_tickers:
        lines.append("No crypto-exempt tickers in this run.")
        lines.append("")
        return "\n".join(lines)

    lines.append("> **Halving Cycle Hold:** These assets are exempt from standard "
                 "sell rules. Strategy: hold through 4-year cycle, accumulate at bottoms.")
    lines.append("")
    lines.append("| Ticker | DCS | RSI | % from 200d | Trend | Status |")
    lines.append("|--------|-----|-----|-------------|-------|--------|")

    for ticker, r, ex in crypto_tickers:
        dcs = r.get("dcs", 0)
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        pct_200d = tech.get("pct_from_200d", 0)
        trend = r.get("trend_score", 0)
        trend_str = "Up" if trend > 0.6 else ("Down" if trend < 0.4 else "Flat")
        expires_at = getattr(ex, "expires_at", "")
        status = f"Exempt until {expires_at}" if expires_at else "Exempt"
        lines.append(f"| **{ticker}** | {dcs:.0f} | {rsi:.0f} | {_pct(pct_200d)} | {trend_str} | {status} |")

    lines.append("")
    return "\n".join(lines)


def _build_reversal_section(result: PipelineResult) -> str:
    """Section 7: Reversal Signals (Phase 2 backtest-validated).

    Research: 124K obs, 625 tickers. RSI Divergence +2.2pp edge,
    Reversal Confirmed +4.6pp, Bottom Turning +4.4pp. All walk-forward stable.
    """
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

    lines = ["## 7. Reversal Signals", ""]

    if total == 0:
        lines.append("No reversal signals detected.")
        lines.append("")
        return "\n".join(lines)

    if reversals:
        lines.append("### Reversal Confirmed (DCS >= 65 + BB breach) — +4.6pp edge")
        for ticker, r in reversals:
            dcs = r.get("dcs", 0)
            held = "Held" if r.get("is_holding") else "Watchlist"
            lines.append(f"- **{ticker}** DCS={dcs:.0f} ({held}) — Full-size deployment signal")
        lines.append("")

    if bottoms:
        lines.append("### Bottom Turning (MACD rising + RSI < 30 + Q3+) — +4.4pp edge")
        for ticker, r in bottoms:
            dcs = r.get("dcs", 0)
            held = "Held" if r.get("is_holding") else "Watchlist"
            lines.append(f"- **{ticker}** DCS={dcs:.0f} ({held}) — Watchlist alert for entry timing")
        lines.append("")

    if divergences:
        lines.append("### RSI Bullish Divergence — +2.2pp edge (walk-forward stable)")
        for ticker, r in divergences:
            dcs = r.get("dcs", 0)
            lines.append(f"- **{ticker}** DCS={dcs:.0f} — +3 boost applied")
        lines.append("")

    if quant_checks:
        lines.append("### Quant Freshness Warning")
        lines.append("> 41% of Q4+ stocks at RSI < 30 drop below quant 4 at next observation.")
        for ticker, r in quant_checks:
            dcs = r.get("dcs", 0)
            lines.append(f"- {ticker} DCS={dcs:.0f} — Verify quant score is current")
        lines.append("")

    return "\n".join(lines)


def _build_subscore_drivers(result: PipelineResult) -> str:
    """Section 8: Sub-Score Driver Analysis.

    Purpose: Identify what's driving/dragging each top ticker's DCS.
    Sub-scores: MQ (momentum quality), FQ (fundamental quality),
    TO (technical opportunity), MR (market regime), VC (value composite).
    """
    scored = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("dcs", 0) >= 60 and r.get("sub_scores")
    ]
    scored.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    lines = ["## 8. Sub-Score Driver Analysis", ""]

    if not scored:
        lines.append("No tickers with DCS >= 60 and sub-score data.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Ticker | DCS | MQ | FQ | TO | MR | VC | Strongest | Weakest |")
    lines.append("|--------|-----|----|----|----|----|----|-----------|---------| ")

    for ticker, r in scored[:15]:
        dcs = r.get("dcs", 0)
        sub = r.get("sub_scores", {})
        mq = sub.get("MQ", 0)
        fq = sub.get("FQ", 0)
        to = sub.get("TO", 0)
        mr = sub.get("MR", 0)
        vc = sub.get("VC", 0)

        sub_vals = {"MQ": mq, "FQ": fq, "TO": to, "MR": mr, "VC": vc}
        strongest = max(sub_vals, key=sub_vals.get) if sub_vals else "-"
        weakest = min(sub_vals, key=sub_vals.get) if sub_vals else "-"

        lines.append(
            f"| **{ticker}** | {dcs:.0f} | {mq:.0f} | {fq:.0f} | "
            f"{to:.0f} | {mr:.0f} | {vc:.0f} | {strongest} | {weakest} |"
        )

    lines.append("")
    lines.append("> **Reading:** Each sub-score is 0-100. MQ=Momentum Quality, "
                 "FQ=Fundamental Quality, TO=Technical Opportunity, "
                 "MR=Market Regime, VC=Value Composite.")
    lines.append("")
    return "\n".join(lines)


def _build_rs_section(result: PipelineResult) -> str:
    """Section 9: Relative Strength vs SPY (Antonacci dual momentum).

    Research: Antonacci (2013) — RS > 1.0 = outperforming market on
    12-month basis. RS < 0.7 for 4+ weeks is sell criterion #5.
    """
    outperformers = []
    underperformers = []

    for ticker, r in result.scores.items():
        tech = r.get("technicals", {})
        rs = tech.get("rs_vs_spy", 0)
        if rs > 0:
            if rs > 1.0:
                outperformers.append((ticker, r, rs))
            elif rs < 0.7:
                underperformers.append((ticker, r, rs))

    lines = ["## 9. Relative Strength vs SPY", ""]

    if not outperformers and not underperformers:
        lines.append("No tickers with significant relative strength divergence.")
        lines.append("")
        return "\n".join(lines)

    if outperformers:
        outperformers.sort(key=lambda x: x[2], reverse=True)
        lines.append("### Outperformers (RS > 1.0)")
        lines.append("")
        lines.append("| Ticker | RS vs SPY | DCS | Momentum |")
        lines.append("|--------|-----------|-----|----------|")
        for ticker, r, rs in outperformers[:10]:
            dcs = r.get("dcs", 0)
            sa = r.get("sa_data", {})
            mom = sa.get("momentum", "-")
            lines.append(f"| **{ticker}** | {rs:.3f} | {dcs:.0f} | {mom} |")
        lines.append("")

    if underperformers:
        underperformers.sort(key=lambda x: x[2])
        lines.append("### Underperformers (RS < 0.7) — Sell Criterion #5")
        lines.append("")
        lines.append("| Ticker | RS vs SPY | DCS | Status |")
        lines.append("|--------|-----------|-----|--------|")
        for ticker, r, rs in underperformers[:10]:
            dcs = r.get("dcs", 0)
            held = "**Held**" if r.get("is_holding") else "Watchlist"
            lines.append(f"| {ticker} | {rs:.3f} | {dcs:.0f} | {held} |")
        lines.append("")
        lines.append("> RS < 0.7 for 4+ weeks triggers sell criterion #5. "
                     "Holdings require review; watchlist tickers deprioritized.")
        lines.append("")

    return "\n".join(lines)


def _build_revision_momentum_section(result: PipelineResult) -> str:
    """Section 10: EPS Revision Momentum.

    Research: Novy-Marx 2015 — revision momentum subsumes price momentum
    and eliminates crash risk. Mill Street Research: top-bottom decile
    spread 7.6% annualized. Fed WP 2024-049: revisions explain >10% of
    3-6 month return variation.
    """
    improving = []
    deteriorating = []

    for ticker, r in result.scores.items():
        rev_mom = r.get("revision_momentum", {})
        direction = rev_mom.get("direction", "")
        delta_4w = rev_mom.get("delta_4w", 0)
        if direction == "improving":
            improving.append((ticker, r, delta_4w))
        elif direction == "deteriorating":
            deteriorating.append((ticker, r, delta_4w))

    lines = ["## 10. EPS Revision Momentum", ""]

    if not improving and not deteriorating:
        lines.append("No revision momentum data available this run.")
        lines.append("")
        return "\n".join(lines)

    if improving:
        improving.sort(key=lambda x: x[2], reverse=True)
        lines.append("### Improving Revisions (bullish — Novy-Marx momentum)")
        lines.append("")
        lines.append("| Ticker | 4w Delta | DCS | SA Rev Grade | Status |")
        lines.append("|--------|----------|-----|-------------|--------|")
        for ticker, r, delta in improving[:8]:
            dcs = r.get("dcs", 0)
            sa = r.get("sa_data", {})
            rev_grade = sa.get("revisions", "-")
            held = "Held" if r.get("is_holding") else "WL"
            lines.append(f"| **{ticker}** | {delta:+.3f} | {dcs:.0f} | {rev_grade} | {held} |")
        lines.append("")

    if deteriorating:
        deteriorating.sort(key=lambda x: x[2])
        lines.append("### Deteriorating Revisions (bearish — sell criterion watch)")
        lines.append("")
        lines.append("| Ticker | 4w Delta | DCS | SA Rev Grade | Status |")
        lines.append("|--------|----------|-----|-------------|--------|")
        for ticker, r, delta in deteriorating[:8]:
            dcs = r.get("dcs", 0)
            sa = r.get("sa_data", {})
            rev_grade = sa.get("revisions", "-")
            held = "**Held**" if r.get("is_holding") else "WL"
            lines.append(f"| {ticker} | {delta:+.3f} | {dcs:.0f} | {rev_grade} | {held} |")
        lines.append("")
        lines.append("> 3+ sub-grade drop in 4 weeks triggers EPS Revision sell signal. "
                     "2 sub-grades triggers WARNING.")
        lines.append("")

    return "\n".join(lines)


def _build_obv_section(result: PipelineResult) -> str:
    """Section 11: OBV Divergence Analysis.

    Research: Granville 1963 — OBV divergences lead price by 2-6 weeks.
    Bullish divergence (price falling, OBV rising) = smart money accumulating.
    Bearish divergence (price rising, OBV falling) = smart money distributing.
    """
    bullish = []
    bearish = []

    for ticker, r in result.scores.items():
        tech = r.get("technicals", {})
        obv_div = tech.get("obv_divergence", "")
        obv_strength = tech.get("obv_divergence_strength", 0)
        if obv_div == "bullish":
            bullish.append((ticker, r, obv_strength))
        elif obv_div == "bearish":
            bearish.append((ticker, r, obv_strength))

    lines = ["## 11. OBV Divergence Analysis", ""]

    if not bullish and not bearish:
        lines.append("No OBV divergences detected this run.")
        lines.append("")
        return "\n".join(lines)

    if bullish:
        bullish.sort(key=lambda x: x[2], reverse=True)
        lines.append("### Bullish Divergences (accumulation — OBV rising, price falling)")
        lines.append("")
        lines.append("| Ticker | Strength | DCS | RSI | Status |")
        lines.append("|--------|----------|-----|-----|--------|")
        for ticker, r, strength in bullish[:8]:
            dcs = r.get("dcs", 0)
            tech = r.get("technicals", {})
            rsi = tech.get("rsi_14", 0)
            held = "Held" if r.get("is_holding") else "WL"
            lines.append(f"| **{ticker}** | {strength:.2f} | {dcs:.0f} | {rsi:.0f} | {held} |")
        lines.append("")

    if bearish:
        bearish.sort(key=lambda x: x[2], reverse=True)
        lines.append("### Bearish Divergences (distribution — OBV falling, price rising)")
        lines.append("")
        lines.append("| Ticker | Strength | DCS | RSI | Status |")
        lines.append("|--------|----------|-----|-----|--------|")
        for ticker, r, strength in bearish[:8]:
            dcs = r.get("dcs", 0)
            tech = r.get("technicals", {})
            rsi = tech.get("rsi_14", 0)
            held = "**Held**" if r.get("is_holding") else "WL"
            lines.append(f"| {ticker} | {strength:.2f} | {dcs:.0f} | {rsi:.0f} | {held} |")
        lines.append("")
        lines.append("> Bearish divergences on held positions warrant closer monitoring. "
                     "OBV leads price by 2-6 weeks (Granville 1963).")
        lines.append("")

    return "\n".join(lines)


def _build_sell_criteria_section(result: PipelineResult) -> str:
    """Section 12: Sell Criteria & Flags (holdings only)."""
    held = result.held_symbols or set()
    flagged = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("sell_flags") and ticker in held
    ]
    flagged.sort(key=lambda x: len(x[1].get("sell_flags", [])), reverse=True)

    lines = ["## 12. Sell Criteria & Flags", ""]

    if not flagged:
        lines.append("No sell flags triggered.")
        lines.append("")
        return "\n".join(lines)

    # Separate urgent (2+ flags) from monitor (1 flag)
    urgent = [(t, r) for t, r in flagged if len(r.get("sell_flags", [])) >= 2]
    monitor = [(t, r) for t, r in flagged if len(r.get("sell_flags", [])) == 1]

    if urgent:
        lines.append(f"### REVIEW REQUIRED ({len(urgent)} tickers with 2+ flags)")
        lines.append("")
        lines.append("| Ticker | DCS | Flags | RSI | SMA Status | Held? |")
        lines.append("|--------|-----|-------|-----|------------|-------|")
        for ticker, r in urgent:
            dcs = r.get("dcs", 0)
            flags = _format_sell_flags(r.get("sell_flags", []))
            tech = r.get("technicals", {})
            rsi = tech.get("rsi_14", 0)
            pct_200d = tech.get("pct_from_200d", 0)
            sma_status = f"{_pct(pct_200d)}" if pct_200d else "-"
            held = "**Yes**" if r.get("is_holding") else "No"
            lines.append(f"| **{ticker}** | {dcs:.0f} | {flags} | {rsi:.0f} | {sma_status} | {held} |")
        lines.append("")

    if monitor:
        lines.append(f"### Monitor ({len(monitor)} tickers with 1 flag)")
        lines.append("")
        lines.append("| Ticker | DCS | Flag | RSI | Held? |")
        lines.append("|--------|-----|------|-----|-------|")
        for ticker, r in monitor:
            dcs = r.get("dcs", 0)
            flags = _format_sell_flags(r.get("sell_flags", []))
            tech = r.get("technicals", {})
            rsi = tech.get("rsi_14", 0)
            held = "Yes" if r.get("is_holding") else "No"
            lines.append(f"| {ticker} | {dcs:.0f} | {flags} | {rsi:.0f} | {held} |")
        lines.append("")

    lines.append("> **Rule:** Any 2 sell criteria = review required. "
                 "Grace period: 180-day hold window for weakening positions.")
    lines.append("")
    return "\n".join(lines)


def _build_grace_period_section(
    active_grace_periods: list[dict[str, Any]] | None,
) -> str:
    """Section 13: Active Grace Periods."""
    lines = ["## 13. Active Grace Periods", ""]

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
    """Section 14: Exemption Status."""
    lines = ["## 14. Exemption Status", ""]

    if not exempt_tickers:
        lines.append("No tickers with exemptions.")
        lines.append("")
        return "\n".join(lines)

    crypto = []
    cash = []
    expired = []

    for ticker, exemption in exempt_tickers.items():
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


def _build_drawdown_section(
    classifications: dict[str, str] | None,
    ticker_values: dict[str, float] | None = None,
) -> str:
    """Section 15: Drawdown Defense (count + dollar-weighted).

    Research: 15-year backtest, 14 SPY drawdown episodes. Dollar-weighted
    composition matters more than count — a portfolio with many hedge
    tickers but tiny positions is not actually defensive.
    """
    lines = ["## 15. Drawdown Defense", ""]

    if not classifications:
        lines.append("No drawdown classification data available.")
        lines.append("")
        return "\n".join(lines)

    values = ticker_values or {}

    # Count and dollar-weighted by class
    class_order = ["HEDGE", "DEFENSIVE", "MODERATE", "CYCLICAL", "AMPLIFIER"]
    counts: dict[str, list[str]] = {c: [] for c in class_order}
    dollar_vals: dict[str, float] = {c: 0.0 for c in class_order}

    for ticker, cls in classifications.items():
        if cls in counts:
            counts[cls].append(ticker)
            dollar_vals[cls] += values.get(ticker, 0)

    total_count = len(classifications)
    total_dollars = sum(dollar_vals.values())
    offense_count = len(counts.get("CYCLICAL", [])) + len(counts.get("AMPLIFIER", []))
    defense_count = len(counts.get("HEDGE", [])) + len(counts.get("DEFENSIVE", []))
    offense_dollars = dollar_vals.get("CYCLICAL", 0) + dollar_vals.get("AMPLIFIER", 0)
    defense_dollars = dollar_vals.get("HEDGE", 0) + dollar_vals.get("DEFENSIVE", 0)

    lines.append(f"**Total:** {total_count} tickers classified | "
                 f"**Defense:** {defense_count} ({defense_count/total_count*100:.0f}%) | "
                 f"**Offense:** {offense_count} ({offense_count/total_count*100:.0f}%)")
    if total_dollars > 0:
        lines.append(f"**Dollar-Weighted:** Defense {_pct(defense_dollars / total_dollars)} | "
                     f"Offense {_pct(offense_dollars / total_dollars)}")
    lines.append("")

    # Build table with both count and dollar columns
    lines.append("| Class | Count | Count% | $ Value | $ % | Tickers |")
    lines.append("|-------|-------|--------|---------|-----|---------|")
    for cls in class_order:
        tickers = counts[cls]
        count_pct = len(tickers) / total_count * 100 if total_count > 0 else 0
        dval = dollar_vals[cls]
        dpct = dval / total_dollars * 100 if total_dollars > 0 else 0
        ticker_str = ", ".join(sorted(tickers)[:8])
        if len(tickers) > 8:
            ticker_str += f" +{len(tickers)-8}"
        lines.append(
            f"| {cls} | {len(tickers)} | {count_pct:.0f}% | "
            f"${dval:,.0f} | {dpct:.0f}% | {ticker_str or '-'} |"
        )

    lines.append("")

    if total_count > 0 and offense_count / total_count > 0.5:
        lines.append(f"> **Portfolio is {offense_count/total_count*100:.0f}% offense by count** "
                     f"(cyclical + amplifier).")
        if total_dollars > 0:
            offense_dpct = offense_dollars / total_dollars * 100
            lines.append(f"> Dollar-weighted offense: {offense_dpct:.0f}%. "
                         "Consider defensive rotation if VIX enters FEAR/PANIC regime.")
        lines.append("")

    return "\n".join(lines)


def _build_correlation_section(result: PipelineResult) -> str:
    """Section 16: Correlation & Diversification."""
    corr = result.correlation
    lines = ["## 16. Correlation & Diversification", ""]

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
    ticker_values: dict[str, float] | None = None,
) -> str:
    """Section 17: Sector Exposure."""
    lines = ["## 17. Sector Exposure", ""]

    if not ticker_sectors:
        lines.append("No sector mapping available.")
        lines.append("")
        return "\n".join(lines)

    values = ticker_values or {}

    # Group by sector with dollar values
    sector_stats: dict[str, list[tuple[str, float, float]]] = {}
    for ticker, r in result.scores.items():
        sector = ticker_sectors.get(ticker, "Other")
        dcs = r.get("dcs", 0)
        val = values.get(ticker, 0)
        sector_stats.setdefault(sector, []).append((ticker, dcs, val))

    total_val = sum(values.values()) if values else 0

    lines.append("| Sector | Count | $ Value | Weight | Avg DCS | Top Ticker |")
    lines.append("|--------|-------|---------|--------|---------|------------|")

    for sector in sorted(sector_stats.keys()):
        tickers = sector_stats[sector]
        count = len(tickers)
        avg_dcs = sum(d for _, d, _ in tickers) / count if count > 0 else 0
        sector_val = sum(v for _, _, v in tickers)
        weight = sector_val / total_val if total_val > 0 else 0
        top = max(tickers, key=lambda x: x[1])
        lines.append(
            f"| {sector} | {count} | ${sector_val:,.0f} | {_pct(weight)} | "
            f"{avg_dcs:.0f} | {top[0]} ({top[1]:.0f}) |"
        )

    lines.append("")

    # Concentration warning: any sector > 25%
    if total_val > 0:
        for sector, tickers in sector_stats.items():
            sector_val = sum(v for _, _, v in tickers)
            if sector_val / total_val > 0.25:
                lines.append(f"> **{sector} concentration: {sector_val/total_val*100:.0f}%.** "
                             "Consider diversifying — max sector target is 25%.")
                lines.append("")

    return "\n".join(lines)


def _build_war_chest_section(
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    vix_regime: str = "NORMAL",
    war_chest_value: float = 0.0,
    total_portfolio_value: float = 0.0,
) -> str:
    """Section 18: War Chest Status (VIX-regime dynamic targets)."""
    lines = ["## 18. War Chest Status", ""]

    surplus = war_chest_pct - war_chest_target
    status = "ADEQUATE" if surplus >= 0 else "**BELOW TARGET**"

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| VIX Regime | {_vix_emoji(vix_regime)} |")
    lines.append(f"| Target | {_pct(war_chest_target)} |")
    lines.append(f"| Actual | {_pct(war_chest_pct)} |")
    if total_portfolio_value > 0:
        lines.append(f"| War Chest Value | ${war_chest_value:,.0f} |")
        lines.append(f"| Total Portfolio | ${total_portfolio_value:,.0f} |")
    lines.append(f"| Surplus/Gap | {_pct(surplus)} |")
    lines.append(f"| Status | {status} |")
    lines.append("")

    if surplus < 0:
        lines.append(f"> **War chest below {_vix_emoji(vix_regime)} target.** "
                     f"Consider reducing position sizes or deferring new deployments "
                     f"until cash reserves reach {_pct(war_chest_target)}.")
        lines.append("")

    return "\n".join(lines)


def _build_per_account_section(
    positions: list[dict[str, Any]] | None,
    scores: dict[str, Any] | None = None,
) -> str:
    """Section 19: Per-Account Holdings Health.

    Purpose: Multi-account portfolio management — see the health of
    each account at a glance, with DCS scores and signals.
    """
    lines = ["## 19. Per-Account Holdings Health", ""]

    if not positions:
        lines.append("No position data available.")
        lines.append("")
        return "\n".join(lines)

    scores = scores or {}

    # Group by account
    accounts: dict[str, list[dict]] = {}
    for pos in positions:
        acct = pos.get("account_id", "Unknown")
        accounts.setdefault(acct, []).append(pos)

    for acct_id in sorted(accounts.keys()):
        acct_positions = accounts[acct_id]
        total_val = sum(float(p.get("market_value", 0)) for p in acct_positions)
        lines.append(f"### {acct_id} (${total_val:,.0f})")
        lines.append("")
        lines.append("| Ticker | Value | Weight | DCS | Signal | Flags |")
        lines.append("|--------|-------|--------|-----|--------|-------|")

        sorted_pos = sorted(
            acct_positions,
            key=lambda p: float(p.get("market_value", 0)),
            reverse=True,
        )

        for pos in sorted_pos:
            symbol = pos.get("symbol", "")
            value = float(pos.get("market_value", 0))
            if value <= 0:
                continue
            weight = value / total_val if total_val > 0 else 0
            r = scores.get(symbol, {})
            dcs = r.get("dcs", 0) if r else 0
            signal = r.get("dcs_signal", "-") if r else "-"
            flags = _format_sell_flags(r.get("sell_flags", [])) if r else "-"
            lines.append(
                f"| {symbol} | ${value:,.0f} | {_pct(weight)} | "
                f"{dcs:.0f} | {signal} | {flags} |"
            )

        lines.append("")

    return "\n".join(lines)


def _build_action_items(result: PipelineResult) -> str:
    """Section 20: Big Picture / Action Items."""
    lines = ["## 20. Action Items", ""]

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
    urgent = sum(1 for r in result.scores.values() if len(r.get("sell_flags", [])) >= 2)
    if urgent > 0:
        actions.append(f"- **{urgent} tickers need REVIEW** (2+ sell flags)")
    elif flagged > 0:
        actions.append(f"- {flagged} tickers with sell flags — monitor")

    # Reversal signals
    rev_count = sum(1 for r in result.scores.values() if r.get("reversal_confirmed"))
    btm_count = sum(1 for r in result.scores.values() if r.get("bottom_turning"))
    if rev_count > 0:
        actions.append(f"- {rev_count} reversal confirmed — high-confidence entry point(s)")
    if btm_count > 0:
        actions.append(f"- {btm_count} bottom turning — watchlist alert(s)")

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
    return "\n".join(lines)


def _build_quick_reference(
    result: PipelineResult,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
) -> str:
    """Section 21: Quick Reference Summary.

    Purpose: At-a-glance decision table — all key numbers in one place.
    """
    lines = [
        "## 21. Quick Reference",
        "",
        "---",
        "",
    ]

    # Count signals
    n_buy = sum(1 for r in result.scores.values() if r.get("dcs", 0) >= 65)
    n_watch = sum(1 for r in result.scores.values() if 50 <= r.get("dcs", 0) < 65)
    n_sell = sum(1 for r in result.scores.values() if r.get("sell_flags"))
    n_rev = sum(1 for r in result.scores.values() if r.get("reversal_confirmed"))
    n_held = len(result.held_symbols)

    top = result.top_scores[0] if result.top_scores else ("N/A", 0)

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| VIX | {result.vix_current:.1f} ({result.vix_regime}) |")
    lines.append(f"| SPY vs 200d | {_pct(result.spy_pct_from_200d)} |")
    lines.append(f"| Breadth | {_pct(result.breadth_pct, 0)} |")
    lines.append(f"| Regime Score | {result.market_regime_score:.2f} |")
    lines.append(f"| Holdings | {n_held} |")
    lines.append(f"| Buy Signals (>= 65) | {n_buy} |")
    lines.append(f"| Watch Zone (50-64) | {n_watch} |")
    lines.append(f"| Sell Flags | {n_sell} |")
    lines.append(f"| Reversals | {n_rev} |")
    lines.append(f"| Effective Bets | {result.correlation.effective_bets:.1f} |")
    lines.append(f"| War Chest | {_pct(war_chest_pct)} (target {_pct(war_chest_target)}) |")
    lines.append(f"| Top DCS | {top[0]} ({top[1]:.0f}) |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by Threshold v0.5.0 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_narrative(
    result: PipelineResult,
    *,
    ticker_sectors: dict[str, str] | None = None,
    ticker_values: dict[str, float] | None = None,
    drawdown_classifications: dict[str, str] | None = None,
    positions: list[dict[str, Any]] | None = None,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    war_chest_value: float = 0.0,
    total_portfolio_value: float = 0.0,
    output_dir: str | Path | None = None,
) -> str:
    """Generate the full Markdown narrative report.

    Parameters:
        result: PipelineResult from run_scoring_pipeline().
        ticker_sectors: {symbol: sector_name} mapping.
        ticker_values: {symbol: dollar_value} for dollar-weighted analysis.
        drawdown_classifications: {symbol: class_name} from backtest.
        positions: List of position dicts for per-account reporting.
        war_chest_pct: Current war chest % of portfolio.
        war_chest_target: VIX-regime target %.
        war_chest_value: Dollar value of war chest holdings.
        total_portfolio_value: Total portfolio dollar value.
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

    held = result.held_symbols or set()

    # Build sections in order (23 sections)
    parts: list[str] = [
        # Header
        _build_header(result),
        # 1. Macro
        _build_macro_section(result),
        # 2. Dip-Buy (split Holdings/Watchlist)
        _build_dipbuy_section(result, held_symbols=held),
        # 2.5. Gate 3
        _build_gate3_section(result),
        # 3. Falling Knives (non-defensive)
        _build_falling_knife_section(result, drawdown_classifications),
        # 4. Hedges/Defensives in Downtrend (D-7)
        _build_hedge_downtrend_section(result, drawdown_classifications),
        # 5. Watch Zone
        _build_watch_section(result),
        # 6. Bitcoin & Crypto
        _build_crypto_section(result),
        # 7. Reversal Signals
        _build_reversal_section(result),
        # 8. Sub-Score Drivers
        _build_subscore_drivers(result),
        # 9. Relative Strength vs SPY
        _build_rs_section(result),
        # 10. EPS Revision Momentum
        _build_revision_momentum_section(result),
        # 11. OBV Divergences
        _build_obv_section(result),
        # 12. Sell Criteria & Flags
        _build_sell_criteria_section(result),
        # 13. Grace Periods
        _build_grace_period_section(result.active_grace_periods),
        # 14. Exemptions
        _build_exemption_section(result.exempt_tickers),
        # 15. Drawdown Defense (count + dollar-weighted)
        _build_drawdown_section(drawdown_classifications, ticker_values),
        # 16. Correlation & Diversification
        _build_correlation_section(result),
        # 17. Sector Exposure
        _build_sector_section(result, ticker_sectors, ticker_values),
        # 18. War Chest
        _build_war_chest_section(
            war_chest_pct, war_chest_target, result.vix_regime,
            war_chest_value, total_portfolio_value,
        ),
        # 19. Per-Account Holdings
        _build_per_account_section(positions, result.scores),
        # 20. Action Items
        _build_action_items(result),
        # 21. Quick Reference
        _build_quick_reference(result, war_chest_pct, war_chest_target),
    ]

    content = "\n".join(parts)

    # Write file
    filepath = output_dir / f"narrative_{date_str}.md"
    with open(filepath, "w") as f:
        f.write(content)

    logger.info("Narrative generated: %s", filepath)
    return str(filepath)
