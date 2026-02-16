"""Decision Hierarchy Dashboard — HTML assembly.

Combines chart builder outputs from ``charts.py`` into a single self-contained
HTML file organized by investment decision hierarchy:

  Level 1   : Macro Regime
  Level 1.5 : Sector Rotation
  Level 2   : Allocation & War Chest
  Level 2.1 : Drawdown Defense
  Level 2.5 : Deployment Discipline
  Level 3   : Selection (DCS scores, signals)
  Level 3.1 : Sell Alerts & Warnings
  Level 3.2 : Holdings Health (per-account)
  Level 3.3 : Watchlist
  Level 4   : Behavioral (Housel framework)
  Ref       : Deep Dive (correlation, sector breakdown)
"""

from __future__ import annotations

import logging
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.io as pio

from threshold.engine.pipeline import PipelineResult
from threshold.output.charts import (
    COLORS,
    build_correlation_heatmap,
    build_dcs_scatter,
    build_drawdown_defense_bars,
    build_market_context_html,
    build_sector_rrg,
    build_sector_treemap,
    build_signal_cards_html,
    build_war_chest_gauge,
)

logger = logging.getLogger(__name__)

# Plotly CDN
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _html_header(title: str, date_str: str) -> str:
    """Generate HTML head with inline CSS and Plotly CDN."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root {{
    --bg: {COLORS['bg']};
    --card: {COLORS['card']};
    --text: {COLORS['text']};
    --muted: {COLORS['muted']};
    --green: {COLORS['green']};
    --red: {COLORS['red']};
    --yellow: {COLORS['yellow']};
    --blue: {COLORS['blue']};
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: Inter, -apple-system, sans-serif;
    line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(15, 17, 23, 0.95);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--card);
    padding: 8px 0;
  }}
  nav .inner {{
    max-width: 1200px; margin: 0 auto; padding: 0 20px;
    display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
  }}
  nav a {{
    color: var(--muted); text-decoration: none; font-size: 13px;
    padding: 4px 8px; border-radius: 4px; transition: all 0.2s;
  }}
  nav a:hover {{ color: var(--text); background: var(--card); }}
  nav .brand {{ color: var(--blue); font-weight: 700; font-size: 15px; }}
  .section {{
    margin: 32px 0; padding: 24px;
    background: var(--card); border-radius: 10px;
  }}
  .section h2 {{
    font-size: 20px; margin-bottom: 16px;
    border-bottom: 2px solid var(--blue); padding-bottom: 8px;
  }}
  .section h3 {{ font-size: 16px; margin: 16px 0 8px; color: var(--blue); }}
  .level-badge {{
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 3px; margin-right: 8px;
    background: var(--blue); color: var(--bg);
  }}
  .alert-card {{
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    border-left: 4px solid;
  }}
  .alert-urgent {{ border-color: var(--red); background: rgba(230,57,70,0.08); }}
  .alert-monitor {{ border-color: var(--yellow); background: rgba(255,209,102,0.08); }}
  .alert-ok {{ border-color: var(--green); background: rgba(0,217,126,0.08); }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin: 12px 0;
  }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2d36; }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
  .tab-container {{ margin: 16px 0; }}
  .tab-buttons {{ display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 12px; }}
  .tab-btn {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid #2a2d36;
    background: transparent; color: var(--muted); cursor: pointer;
    font-size: 12px; transition: all 0.2s;
  }}
  .tab-btn:hover {{ background: #2a2d36; color: var(--text); }}
  .tab-btn.active {{ background: var(--blue); color: var(--bg); border-color: var(--blue); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  .checklist {{ list-style: none; padding: 0; }}
  .checklist li {{
    padding: 8px 12px; border-bottom: 1px solid #2a2d36;
    font-size: 13px; color: var(--text);
  }}
  .checklist li::before {{ content: "\\2610 "; color: var(--muted); margin-right: 8px; }}
  .footer {{
    text-align: center; padding: 24px; font-size: 12px; color: var(--muted);
  }}
</style>
</head>
<body>
"""


def _navbar(date_str: str) -> str:
    """Build sticky navigation bar with section anchors."""
    links = [
        ("macro", "Macro"),
        ("allocation", "Allocation"),
        ("drawdown", "Drawdown"),
        ("deployment", "Deploy"),
        ("selection", "Selection"),
        ("sell-alerts", "Alerts"),
        ("holdings", "Holdings"),
        ("correlation", "Correlation"),
        ("behavioral", "Behavioral"),
    ]
    nav_links = " ".join(f'<a href="#{id}">{label}</a>' for id, label in links)
    return f"""
<nav>
  <div class="inner">
    <span class="brand">Threshold</span>
    <span style="color:{COLORS['muted']};font-size:12px;">{date_str}</span>
    {nav_links}
  </div>
</nav>
"""


def _embed_plotly(fig: Any, div_id: str) -> str:
    """Convert a Plotly figure to an embedded HTML div."""
    try:
        html = pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)
        return html
    except Exception as e:
        return f'<div style="color:{COLORS["red"]};padding:20px;">Chart error: {e}</div>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_macro_section(result: PipelineResult) -> str:
    """Level 1: Macro Regime section."""
    market_html = build_market_context_html(
        vix_current=result.vix_current,
        vix_regime=result.vix_regime,
        spy_pct=result.spy_pct_from_200d,
        spy_above_200d=result.spy_above_200d,
        breadth_pct=result.breadth_pct,
        effective_bets=result.correlation.effective_bets,
    )

    return f"""
<div class="section" id="macro">
  <h2><span class="level-badge">L1</span>Macro Regime</h2>
  {market_html}
</div>
"""


def _build_selection_section(
    result: PipelineResult,
    ticker_sectors: dict[str, str] | None = None,
) -> str:
    """Level 3: Selection section with DCS scatter and signal cards."""
    # Signal cards
    cards_html = build_signal_cards_html(result.scores)

    # DCS scatter (split by holdings/watchlist)
    scatter_fig = build_dcs_scatter(
        result.scores,
        ticker_sectors=ticker_sectors,
        held_symbols=result.held_symbols,
    )
    scatter_html = _embed_plotly(scatter_fig, "dcs-scatter")

    # Top scores table
    top_rows = ""
    for sym, dcs in result.top_scores[:10]:
        signal = result.scores[sym].get("dcs_signal", "")
        technicals = result.scores[sym].get("technicals", {})
        rsi = technicals.get("rsi_14", 0)
        sell_flags = result.scores[sym].get("sell_flags", [])
        flags_str = ", ".join(sell_flags) if sell_flags else "-"
        is_held = sym in (result.held_symbols or set())
        held_badge = f'<span style="color:{COLORS["blue"]}">H</span>' if is_held else f'<span style="color:{COLORS["teal"]}">W</span>'
        color = COLORS["green"] if dcs >= 65 else (COLORS["yellow"] if dcs >= 50 else COLORS["red"])
        top_rows += f"""
        <tr>
          <td><b>{sym}</b> {held_badge}</td>
          <td style="color:{color}">{dcs:.1f}</td>
          <td>{signal}</td>
          <td>{rsi:.0f}</td>
          <td style="font-size:11px;color:{COLORS['muted']}">{flags_str}</td>
        </tr>"""

    return f"""
<div class="section" id="selection">
  <h2><span class="level-badge">L3</span>Selection — DCS Signals</h2>
  {cards_html}
  <h3>DCS vs RSI (Blue=Holdings, Teal=Watchlist)</h3>
  {scatter_html}
  <h3>Top Scores</h3>
  <table>
    <tr><th>Ticker</th><th>DCS</th><th>Signal</th><th>RSI</th><th>Sell Flags</th></tr>
    {top_rows}
  </table>
</div>
"""


def _build_allocation_section(
    result: PipelineResult,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    war_chest_value: float = 0.0,
    total_portfolio_value: float = 0.0,
) -> str:
    """Level 2: Allocation & War Chest section."""
    gauge_fig = build_war_chest_gauge(
        actual_pct=war_chest_pct,
        target_pct=war_chest_target,
        vix_regime=result.vix_regime,
    )
    gauge_html = _embed_plotly(gauge_fig, "war-chest-gauge")

    # Additional details if dollar values available
    detail_html = ""
    if total_portfolio_value > 0:
        detail_html = f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;">
          <div style="background:{COLORS['bg']};padding:10px 14px;border-radius:6px;flex:1;">
            <div style="font-size:11px;color:{COLORS['muted']};">WAR CHEST</div>
            <div style="font-size:20px;font-weight:700;">${war_chest_value:,.0f}</div>
          </div>
          <div style="background:{COLORS['bg']};padding:10px 14px;border-radius:6px;flex:1;">
            <div style="font-size:11px;color:{COLORS['muted']};">PORTFOLIO</div>
            <div style="font-size:20px;font-weight:700;">${total_portfolio_value:,.0f}</div>
          </div>
        </div>"""

    return f"""
<div class="section" id="allocation">
  <h2><span class="level-badge">L2</span>Allocation & War Chest</h2>
  {gauge_html}
  {detail_html}
</div>
"""


def _build_drawdown_section(
    classifications: dict[str, str],
    ticker_values: dict[str, float] | None = None,
) -> str:
    """Level 2.1: Drawdown Defense section."""
    if not classifications:
        return ""

    bars_fig = build_drawdown_defense_bars(classifications, ticker_values=ticker_values)
    bars_html = _embed_plotly(bars_fig, "drawdown-bars")

    return f"""
<div class="section" id="drawdown">
  <h2><span class="level-badge">L2.1</span>Drawdown Defense</h2>
  {bars_html}
</div>
"""


def _build_deployment_section(
    result: PipelineResult,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
) -> str:
    """Level 2.5: Deployment Discipline (3 gates)."""
    # Gate 1: Buy candidates (DCS >= 65)
    candidates = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("dcs", 0) >= 65
    ]
    candidates.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    gate1_rows = ""
    for ticker, r in candidates[:10]:
        dcs = r.get("dcs", 0)
        signal = r.get("dcs_signal", "")
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        is_held = ticker in (result.held_symbols or set())
        held_str = "Held" if is_held else "New"
        color = COLORS["green"] if dcs >= 70 else COLORS["yellow"]
        gate1_rows += f"""<tr>
          <td><b>{ticker}</b></td>
          <td style="color:{color}">{dcs:.0f}</td>
          <td>{signal}</td>
          <td>{rsi:.0f}</td>
          <td>{held_str}</td>
        </tr>"""

    # Gate 2: VIX sizing
    regime = result.vix_regime
    sizing_map = {
        "COMPLACENT": ("Half size", COLORS["yellow"]),
        "NORMAL": ("Full size", COLORS["green"]),
        "FEAR": ("Full size + lean in", COLORS["blue"]),
        "PANIC": ("Aggressive — max lean in", COLORS["blue"]),
    }
    sizing_label, sizing_color = sizing_map.get(regime, ("Full size", COLORS["green"]))

    # Gate 3: Parabolic blocked
    blocked_rows = ""
    for ticker, r in result.scores.items():
        for sig in r.get("signal_board", []):
            if sig.get("legacy_prefix", "").startswith("GATE3:"):
                dcs = r.get("dcs", 0)
                sizing = sig.get("metadata", {}).get("sizing", "WAIT")
                tech = r.get("technicals", {})
                rsi = tech.get("rsi_14", 0)
                ret_8w = tech.get("ret_8w", 0)
                blocked_rows += f"""<tr>
                  <td style="color:{COLORS['red']}"><b>{ticker}</b></td>
                  <td>{dcs:.0f}</td>
                  <td>{rsi:.0f}</td>
                  <td>{ret_8w*100:.0f}%</td>
                  <td style="color:{COLORS['red']}">{sizing}</td>
                </tr>"""
                break

    # War chest check
    wc_ok = war_chest_pct >= war_chest_target
    wc_color = COLORS["green"] if wc_ok else COLORS["red"]
    wc_status = "PASS" if wc_ok else "BELOW TARGET"

    return f"""
<div class="section" id="deployment">
  <h2><span class="level-badge">L2.5</span>Deployment Discipline</h2>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
    <div style="background:{COLORS['bg']};padding:12px;border-radius:6px;flex:1;border-top:3px solid {COLORS['green']};">
      <div style="font-size:11px;color:{COLORS['muted']};">GATE 1: CANDIDATES</div>
      <div style="font-size:24px;font-weight:700;">{len(candidates)}</div>
      <div style="font-size:11px;color:{COLORS['muted']};">DCS >= 65</div>
    </div>
    <div style="background:{COLORS['bg']};padding:12px;border-radius:6px;flex:1;border-top:3px solid {sizing_color};">
      <div style="font-size:11px;color:{COLORS['muted']};">GATE 2: VIX SIZING</div>
      <div style="font-size:18px;font-weight:700;color:{sizing_color};">{sizing_label}</div>
      <div style="font-size:11px;color:{COLORS['muted']};">Regime: {regime}</div>
    </div>
    <div style="background:{COLORS['bg']};padding:12px;border-radius:6px;flex:1;border-top:3px solid {wc_color};">
      <div style="font-size:11px;color:{COLORS['muted']};">WAR CHEST</div>
      <div style="font-size:18px;font-weight:700;color:{wc_color};">{wc_status}</div>
      <div style="font-size:11px;color:{COLORS['muted']};">{war_chest_pct*100:.1f}% / {war_chest_target*100:.0f}%</div>
    </div>
  </div>
  {"<h3>Gate 1: Buy Candidates</h3><table><tr><th>Ticker</th><th>DCS</th><th>Signal</th><th>RSI</th><th>Type</th></tr>" + gate1_rows + "</table>" if gate1_rows else "<p style='color:" + COLORS['muted'] + "'>No buy candidates this run.</p>"}
  {"<h3 style='color:" + COLORS['red'] + "'>Gate 3: Parabolic Blocked</h3><table><tr><th>Ticker</th><th>DCS</th><th>RSI</th><th>8w Return</th><th>Sizing</th></tr>" + blocked_rows + "</table>" if blocked_rows else ""}
</div>
"""


def _build_sell_alerts_section(
    result: PipelineResult,
    drawdown_classifications: dict[str, str] | None = None,
) -> str:
    """Level 3.1: Sell Alerts & Warnings (holdings only)."""
    dd = drawdown_classifications or {}
    held = result.held_symbols or set()
    flagged = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if r.get("sell_flags") and ticker in held
    ]
    if not flagged:
        return f"""
<div class="section" id="sell-alerts">
  <h2><span class="level-badge">L3.1</span>Sell Alerts</h2>
  <p style="color:{COLORS['green']};">No sell flags triggered. All clear.</p>
</div>
"""

    urgent = [(t, r) for t, r in flagged if len(r.get("sell_flags", [])) >= 2]
    monitor = [(t, r) for t, r in flagged if len(r.get("sell_flags", [])) == 1]

    cards_html = ""
    for ticker, r in urgent:
        dcs = r.get("dcs", 0)
        flags = ", ".join(r.get("sell_flags", []))
        dd_class = dd.get(ticker, "")
        held = ticker in (result.held_symbols or set())
        held_badge = f'<span style="color:{COLORS["blue"]};">HELD</span>' if held else ""
        cards_html += f"""
        <div class="alert-card alert-urgent">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;font-size:15px;">{ticker}</span>
            <span>{held_badge} <span style="color:{COLORS['muted']}">{dd_class}</span></span>
          </div>
          <div style="font-size:12px;color:{COLORS['red']};margin-top:4px;">REVIEW REQUIRED — {len(r.get('sell_flags',[]))} flags</div>
          <div style="font-size:12px;color:{COLORS['muted']};margin-top:4px;">DCS {dcs:.0f} | {flags}</div>
        </div>"""

    for ticker, r in monitor:
        dcs = r.get("dcs", 0)
        flags = ", ".join(r.get("sell_flags", []))
        dd_class = dd.get(ticker, "")
        held = ticker in (result.held_symbols or set())
        held_badge = f'<span style="color:{COLORS["blue"]};">HELD</span>' if held else ""
        cards_html += f"""
        <div class="alert-card alert-monitor">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:700;">{ticker}</span>
            <span>{held_badge} <span style="color:{COLORS['muted']}">{dd_class}</span></span>
          </div>
          <div style="font-size:12px;color:{COLORS['yellow']};margin-top:4px;">MONITOR — 1 flag</div>
          <div style="font-size:12px;color:{COLORS['muted']};margin-top:4px;">DCS {dcs:.0f} | {flags}</div>
        </div>"""

    return f"""
<div class="section" id="sell-alerts">
  <h2><span class="level-badge">L3.1</span>Sell Alerts & Warnings</h2>
  <div style="display:flex;gap:12px;margin-bottom:16px;">
    <div style="background:{COLORS['bg']};padding:10px 14px;border-radius:6px;border-top:3px solid {COLORS['red']};">
      <div style="font-size:11px;color:{COLORS['muted']};">URGENT</div>
      <div style="font-size:24px;font-weight:700;color:{COLORS['red']};">{len(urgent)}</div>
    </div>
    <div style="background:{COLORS['bg']};padding:10px 14px;border-radius:6px;border-top:3px solid {COLORS['yellow']};">
      <div style="font-size:11px;color:{COLORS['muted']};">MONITOR</div>
      <div style="font-size:24px;font-weight:700;color:{COLORS['yellow']};">{len(monitor)}</div>
    </div>
  </div>
  {cards_html}
</div>
"""


def _build_holdings_section(
    result: PipelineResult,
    positions: list[dict[str, Any]] | None = None,
) -> str:
    """Level 3.2: Holdings Health by Account (tabbed)."""
    if not positions:
        return ""

    # Group by account
    accounts: dict[str, list[dict]] = {}
    for pos in positions:
        acct = pos.get("account_id", "Unknown")
        accounts.setdefault(acct, []).append(pos)

    if not accounts:
        return ""

    # Build tab buttons and content
    tab_buttons = ""
    tab_contents = ""
    for idx, (acct_id, acct_positions) in enumerate(sorted(accounts.items())):
        active = "active" if idx == 0 else ""
        total_val = sum(float(p.get("market_value", 0)) for p in acct_positions)

        tab_buttons += f'<button class="tab-btn {active}" onclick="showTab(\'tab-{idx}\')">{acct_id} (${total_val:,.0f})</button>'

        rows = ""
        sorted_pos = sorted(acct_positions, key=lambda p: float(p.get("market_value", 0)), reverse=True)
        for pos in sorted_pos:
            symbol = pos.get("symbol", "")
            value = float(pos.get("market_value", 0))
            if value <= 0:
                continue
            weight = value / total_val if total_val > 0 else 0
            r = result.scores.get(symbol, {})
            dcs = r.get("dcs", 0) if r else 0
            signal = r.get("dcs_signal", "-") if r else "-"
            flags = r.get("sell_flags", []) if r else []
            flags_str = ", ".join(flags) if flags else "-"
            dcs_color = COLORS["green"] if dcs >= 65 else (COLORS["yellow"] if dcs >= 50 else COLORS["red"])
            flag_color = COLORS["red"] if len(flags) >= 2 else (COLORS["yellow"] if flags else COLORS["muted"])
            rows += f"""<tr>
              <td><b>{symbol}</b></td>
              <td>${value:,.0f}</td>
              <td>{weight*100:.1f}%</td>
              <td style="color:{dcs_color}">{dcs:.0f}</td>
              <td>{signal}</td>
              <td style="color:{flag_color};font-size:11px;">{flags_str}</td>
            </tr>"""

        tab_contents += f"""
        <div class="tab-content {active}" id="tab-{idx}">
          <table>
            <tr><th>Ticker</th><th>Value</th><th>Weight</th><th>DCS</th><th>Signal</th><th>Flags</th></tr>
            {rows}
          </table>
        </div>"""

    return f"""
<div class="section" id="holdings">
  <h2><span class="level-badge">L3.2</span>Holdings Health by Account</h2>
  <div class="tab-container">
    <div class="tab-buttons">{tab_buttons}</div>
    {tab_contents}
  </div>
</div>
<script>
function showTab(tabId) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}}
</script>
"""


def _build_watchlist_section(
    result: PipelineResult,
) -> str:
    """Level 3.3: Watchlist tickers ranked by DCS."""
    held = result.held_symbols or set()
    watchlist = [
        (ticker, r)
        for ticker, r in result.scores.items()
        if ticker not in held
    ]
    if not watchlist:
        return ""

    watchlist.sort(key=lambda x: x[1].get("dcs", 0), reverse=True)

    rows = ""
    for ticker, r in watchlist[:20]:
        dcs = r.get("dcs", 0)
        signal = r.get("dcs_signal", "")
        tech = r.get("technicals", {})
        rsi = tech.get("rsi_14", 0)
        pct_200d = tech.get("pct_from_200d", 0)
        color = COLORS["green"] if dcs >= 65 else (COLORS["yellow"] if dcs >= 50 else COLORS["muted"])
        deploy = "Ready" if dcs >= 65 else ("-" if dcs < 50 else "Watch")
        deploy_color = COLORS["green"] if dcs >= 65 else COLORS["muted"]
        rows += f"""<tr>
          <td><b>{ticker}</b></td>
          <td style="color:{color}">{dcs:.0f}</td>
          <td>{signal}</td>
          <td>{rsi:.0f}</td>
          <td>{pct_200d*100:+.1f}%</td>
          <td style="color:{deploy_color}">{deploy}</td>
        </tr>"""

    n_ready = sum(1 for _, r in watchlist if r.get("dcs", 0) >= 65)

    return f"""
<div class="section">
  <h2><span class="level-badge">L3.3</span>Watchlist ({n_ready} ready to deploy)</h2>
  <table>
    <tr><th>Ticker</th><th>DCS</th><th>Signal</th><th>RSI</th><th>vs 200d</th><th>Deploy?</th></tr>
    {rows}
  </table>
</div>
"""


def _build_behavioral_section() -> str:
    """Level 4: Behavioral Checklists (Housel framework)."""
    return f"""
<div class="section" id="behavioral">
  <h2><span class="level-badge">L4</span>Behavioral Guardrails</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;">
    <div style="background:{COLORS['bg']};padding:16px;border-radius:8px;">
      <h3 style="color:{COLORS['green']};margin-top:0;">Pre-Buy Checklist</h3>
      <ul class="checklist">
        <li>Do I have a plan or is this FOMO?</li>
        <li>Can I articulate why this fits my strategy?</li>
        <li>Does it pass 3+ of 5 buy criteria?</li>
        <li>Have I passed all 3 deployment gates?</li>
        <li>Is RSI < 80 and not parabolic?</li>
        <li>Will my war chest stay above target?</li>
      </ul>
    </div>
    <div style="background:{COLORS['bg']};padding:16px;border-radius:8px;">
      <h3 style="color:{COLORS['red']};margin-top:0;">Pre-Sell Checklist</h3>
      <ul class="checklist">
        <li>Am I reacting to price or fundamentals?</li>
        <li>Would I be comfortable if market closed 5 years?</li>
        <li>Have I stuck to my original thesis?</li>
        <li>Is this panic or process?</li>
        <li>Does this meet 2+ sell criteria?</li>
      </ul>
    </div>
    <div style="background:{COLORS['bg']};padding:16px;border-radius:8px;">
      <h3 style="color:{COLORS['yellow']};margin-top:0;">"Never Enough" Check</h3>
      <ul class="checklist">
        <li>Am I moving goalposts?</li>
        <li>Taking excessive risk for arbitrary numbers?</li>
        <li>Would achieving this actually change my life?</li>
        <li>Is my risk budget still appropriate?</li>
      </ul>
    </div>
  </div>
</div>
"""


def _build_correlation_section(
    result: PipelineResult,
) -> str:
    """Reference: Correlation section."""
    heatmap_fig = build_correlation_heatmap(
        result.correlation.correlation_matrix,
    )
    heatmap_html = _embed_plotly(heatmap_fig, "corr-heatmap")

    # High correlation pairs table
    pairs_html = ""
    if result.correlation.high_corr_pairs:
        pairs_rows = ""
        for a, b, corr in result.correlation.high_corr_pairs[:10]:
            color = COLORS["red"] if corr > 0.85 else COLORS["yellow"]
            pairs_rows += f"<tr><td>{a}</td><td>{b}</td><td style='color:{color}'>{corr:.3f}</td></tr>"
        pairs_html = f"""
        <h3>High-Correlation Pairs (>{0.80})</h3>
        <table>
          <tr><th>Ticker A</th><th>Ticker B</th><th>Correlation</th></tr>
          {pairs_rows}
        </table>"""

    # Concentration warnings
    warn_html = ""
    if result.concentration_warnings:
        warn_rows = ""
        for w in result.concentration_warnings:
            warn_rows += (
                f"<tr><td style='color:{COLORS['red']}'>{w['ticker']}</td>"
                f"<td>{w['correlated_with']}</td>"
                f"<td>{w['correlation']:.3f}</td></tr>"
            )
        warn_html = f"""
        <h3 style="color:{COLORS['red']}">Concentration Warnings</h3>
        <table>
          <tr><th>Buy Candidate</th><th>Correlated With</th><th>Correlation</th></tr>
          {warn_rows}
        </table>"""

    return f"""
<div class="section" id="correlation">
  <h2><span class="level-badge">REF</span>Correlation & Diversification</h2>
  <p>Effective Bets: <b style="color:{COLORS['blue']}">{result.correlation.effective_bets:.1f}</b>
  &nbsp;|&nbsp; Tickers: {result.correlation.n_tickers}</p>
  {heatmap_html}
  {pairs_html}
  {warn_html}
</div>
"""


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def generate_dashboard(
    result: PipelineResult,
    *,
    ticker_sectors: dict[str, str] | None = None,
    ticker_values: dict[str, float] | None = None,
    drawdown_classifications: dict[str, str] | None = None,
    positions: list[dict[str, Any]] | None = None,
    sector_rankings: list[dict[str, Any]] | None = None,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    war_chest_value: float = 0.0,
    total_portfolio_value: float = 0.0,
    output_dir: str | Path | None = None,
    auto_open: bool = True,
) -> str:
    """Generate the full Decision Hierarchy Dashboard HTML file.

    Parameters:
        result: PipelineResult from run_scoring_pipeline().
        ticker_sectors: {symbol: sector_name} mapping.
        ticker_values: {symbol: dollar_value} for treemap sizing.
        drawdown_classifications: {symbol: class_name} for drawdown bars.
        positions: List of position dicts for per-account health.
        sector_rankings: Sector rotation data for RRG chart.
        war_chest_pct: Current war chest % of portfolio.
        war_chest_target: VIX-regime target %.
        war_chest_value: Dollar value of war chest holdings.
        total_portfolio_value: Total portfolio dollar value.
        output_dir: Directory for HTML output.
        auto_open: Open in browser after generation.

    Returns:
        Path to generated HTML file.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    title = f"Threshold Dashboard — {date_str}"

    if output_dir is None:
        output_dir = Path("~/.threshold/dashboards").expanduser()
    else:
        output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sections
    parts: list[str] = []
    parts.append(_html_header(title, date_str))
    parts.append(_navbar(date_str))
    parts.append('<div class="container">')

    # Level 1: Macro
    parts.append(_build_macro_section(result))

    # Level 1.5: Sector Rotation (if data available)
    if sector_rankings:
        rrg_fig = build_sector_rrg(sector_rankings)
        rrg_html = _embed_plotly(rrg_fig, "sector-rrg")
        parts.append(f"""
<div class="section">
  <h2><span class="level-badge">L1.5</span>Sector Rotation</h2>
  {rrg_html}
</div>""")

    # Level 2: Allocation & War Chest
    parts.append(_build_allocation_section(
        result, war_chest_pct, war_chest_target,
        war_chest_value, total_portfolio_value,
    ))

    # Level 2.1: Drawdown Defense (with dollar-weighted bars)
    if drawdown_classifications:
        parts.append(_build_drawdown_section(drawdown_classifications, ticker_values))

    # Level 2.5: Deployment Discipline
    parts.append(_build_deployment_section(result, war_chest_pct, war_chest_target))

    # Level 3: Selection (split DCS scatter)
    parts.append(_build_selection_section(result, ticker_sectors))

    # Level 3.1: Sell Alerts
    parts.append(_build_sell_alerts_section(result, drawdown_classifications))

    # Level 3.2: Holdings Health (tabbed by account)
    if positions:
        parts.append(_build_holdings_section(result, positions))

    # Level 3.3: Watchlist
    parts.append(_build_watchlist_section(result))

    # Sector Treemap
    if ticker_sectors and result.scores:
        treemap_fig = build_sector_treemap(
            result.scores,
            ticker_sectors=ticker_sectors,
            ticker_values=ticker_values,
        )
        treemap_html = _embed_plotly(treemap_fig, "sector-treemap")
        parts.append(f"""
<div class="section">
  <h2><span class="level-badge">REF</span>Holdings by Sector</h2>
  {treemap_html}
</div>""")

    # Correlation
    if result.correlation.n_tickers >= 2:
        parts.append(_build_correlation_section(result))

    # Level 4: Behavioral
    parts.append(_build_behavioral_section())

    # Footer
    parts.append(f"""
<div class="footer">
  Generated by Threshold v0.5.0 &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""")
    parts.append('</div></body></html>')

    # Write file
    filepath = output_dir / f"dashboard_{date_str}.html"
    with open(filepath, "w") as f:
        f.write("\n".join(parts))

    logger.info("Dashboard generated: %s", filepath)

    if auto_open:
        webbrowser.open(f"file://{filepath}")

    return str(filepath)
