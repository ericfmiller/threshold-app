"""Decision Hierarchy Dashboard — HTML assembly.

Combines chart builder outputs from ``charts.py`` into a single self-contained
HTML file organized by investment decision hierarchy:

  Level 1   : Macro Regime
  Level 1.5 : Sector Rotation
  Level 2   : Allocation & War Chest
  Level 2.1 : Drawdown Defense
  Level 2.5 : Deployment Discipline
  Level 3   : Selection (DCS scores, signals)
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
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
    margin: 12px 0;
  }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2d36; }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
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
        ("selection", "Selection"),
        ("correlation", "Correlation"),
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

    # DCS scatter
    scatter_fig = build_dcs_scatter(
        result.scores,
        ticker_sectors=ticker_sectors,
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
        color = COLORS["green"] if dcs >= 65 else (COLORS["yellow"] if dcs >= 50 else COLORS["red"])
        top_rows += f"""
        <tr>
          <td><b>{sym}</b></td>
          <td style="color:{color}">{dcs:.1f}</td>
          <td>{signal}</td>
          <td>{rsi:.0f}</td>
          <td style="font-size:11px;color:{COLORS['muted']}">{flags_str}</td>
        </tr>"""

    return f"""
<div class="section" id="selection">
  <h2><span class="level-badge">L3</span>Selection — DCS Signals</h2>
  {cards_html}
  <h3>DCS vs RSI</h3>
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
) -> str:
    """Level 2: Allocation & War Chest section."""
    gauge_fig = build_war_chest_gauge(
        actual_pct=war_chest_pct,
        target_pct=war_chest_target,
        vix_regime=result.vix_regime,
    )
    gauge_html = _embed_plotly(gauge_fig, "war-chest-gauge")

    return f"""
<div class="section" id="allocation">
  <h2><span class="level-badge">L2</span>Allocation & War Chest</h2>
  {gauge_html}
</div>
"""


def _build_drawdown_section(
    classifications: dict[str, str],
) -> str:
    """Level 2.1: Drawdown Defense section."""
    if not classifications:
        return ""

    bars_fig = build_drawdown_defense_bars(classifications)
    bars_html = _embed_plotly(bars_fig, "drawdown-bars")

    return f"""
<div class="section" id="drawdown">
  <h2><span class="level-badge">L2.1</span>Drawdown Defense</h2>
  {bars_html}
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
    sector_rankings: list[dict[str, Any]] | None = None,
    war_chest_pct: float = 0.0,
    war_chest_target: float = 0.10,
    output_dir: str | Path | None = None,
    auto_open: bool = True,
) -> str:
    """Generate the full Decision Hierarchy Dashboard HTML file.

    Parameters:
        result: PipelineResult from run_scoring_pipeline().
        ticker_sectors: {symbol: sector_name} mapping.
        ticker_values: {symbol: dollar_value} for treemap sizing.
        drawdown_classifications: {symbol: class_name} for drawdown bars.
        sector_rankings: Sector rotation data for RRG chart.
        war_chest_pct: Current war chest % of portfolio.
        war_chest_target: VIX-regime target %.
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
    parts.append(_build_allocation_section(result, war_chest_pct, war_chest_target))

    # Level 2.1: Drawdown Defense
    if drawdown_classifications:
        parts.append(_build_drawdown_section(drawdown_classifications))

    # Level 3: Selection
    parts.append(_build_selection_section(result, ticker_sectors))

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

    # Footer
    parts.append(f"""
<div class="footer">
  Generated by Threshold v0.4.0 &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
