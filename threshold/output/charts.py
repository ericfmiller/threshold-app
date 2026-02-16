"""Individual chart builder functions for the Decision Hierarchy Dashboard.

Each function returns a Plotly ``go.Figure`` (or HTML string for non-chart
components).  The dashboard assembler in ``dashboard.py`` arranges these
into a single HTML file.

Charts are organized by decision hierarchy level:
  Level 1  : Macro regime
  Level 1.5: Sector rotation
  Level 2  : Allocation & war chest
  Level 2.1: Drawdown defense
  Level 3  : Selection (DCS scatter, signals)
  Ref      : Correlation, sector breakdown
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from threshold.engine.scorer import ScoringResult

# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d26",
    "text": "#e0e0e0",
    "muted": "#888888",
    "green": "#00d97e",
    "red": "#e63946",
    "yellow": "#ffd166",
    "blue": "#4ea8de",
    "purple": "#b388ff",
    "teal": "#2ec4b6",
    "orange": "#f77f00",
    "grid": "#2a2d36",
}

SECTOR_COLORS = {
    "Technology": "#4ea8de",
    "Information Technology": "#4ea8de",
    "Energy": "#f77f00",
    "Materials": "#b388ff",
    "Financials": "#00d97e",
    "Health Care": "#e63946",
    "Consumer Discretionary": "#ffd166",
    "Consumer Staples": "#2ec4b6",
    "Industrials": "#888888",
    "Real Estate": "#d4a574",
    "Communication Services": "#ff6b9d",
    "Utilities": "#a8d8ea",
    "Broad Market": "#e0e0e0",
    "Other": "#555555",
}


def _dark_layout(**overrides: Any) -> dict[str, Any]:
    """Standard dark-theme Plotly layout."""
    layout = {
        "paper_bgcolor": COLORS["bg"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "Inter, sans-serif"},
        "margin": {"l": 50, "r": 30, "t": 50, "b": 50},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
    }
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Level 3: DCS Scatter (primary chart)
# ---------------------------------------------------------------------------

def build_dcs_scatter(
    scores: dict[str, ScoringResult],
    ticker_sectors: dict[str, str] | None = None,
    title: str = "DCS vs RSI — Portfolio Holdings",
) -> go.Figure:
    """Build DCS vs RSI scatter plot with signal zones and reversal markers.

    X-axis: RSI-14, Y-axis: DCS score.
    Green zone: oversold (RSI < 30) + high DCS (>= 65).
    Reversal markers: star=REVERSAL CONFIRMED, diamond=BOTTOM TURNING,
    triangle=RSI DIVERGENCE.
    """
    sectors = ticker_sectors or {}
    fig = go.Figure()

    # Collect data by sector for legend grouping
    sector_data: dict[str, list[dict]] = {}
    for ticker, result in scores.items():
        dcs = result.get("dcs", 0)
        technicals = result.get("technicals", {})
        rsi = technicals.get("rsi_14", 50)
        sector = sectors.get(ticker, "Other")

        point = {
            "ticker": ticker,
            "dcs": dcs,
            "rsi": rsi,
            "signal": result.get("dcs_signal", ""),
            "reversal": result.get("reversal_confirmed", False),
            "bottom": result.get("bottom_turning", False),
            "divergence": result.get("rsi_bullish_divergence", False),
        }
        sector_data.setdefault(sector, []).append(point)

    # Plot each sector
    for sector, points in sorted(sector_data.items()):
        color = SECTOR_COLORS.get(sector, COLORS["muted"])

        # Separate regular and reversal points
        regular = [p for p in points if not p["reversal"] and not p["bottom"]]
        reversals = [p for p in points if p["reversal"]]
        bottoms = [p for p in points if p["bottom"] and not p["reversal"]]
        divergences = [p for p in points if p["divergence"] and not p["reversal"] and not p["bottom"]]

        for group, marker_sym, name_suffix in [
            (regular, "circle", ""),
            (reversals, "star", " [REV]"),
            (bottoms, "diamond", " [BTM]"),
            (divergences, "triangle-up", " [DIV]"),
        ]:
            if not group:
                continue
            fig.add_trace(go.Scatter(
                x=[p["rsi"] for p in group],
                y=[p["dcs"] for p in group],
                mode="markers+text",
                text=[p["ticker"] for p in group],
                textposition="top center",
                textfont={"size": 9, "color": COLORS["text"]},
                marker={
                    "size": 10,
                    "color": color,
                    "symbol": marker_sym,
                    "line": {"width": 1, "color": COLORS["text"]},
                },
                name=f"{sector}{name_suffix}",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "DCS: %{y:.1f}<br>"
                    "RSI: %{x:.1f}<br>"
                    f"Sector: {sector}<br>"
                    "<extra></extra>"
                ),
                legendgroup=sector,
                showlegend=bool(not name_suffix),
            ))

    # Add signal zones
    # Green zone: oversold + high DCS
    fig.add_shape(
        type="rect", x0=0, x1=30, y0=65, y1=100,
        fillcolor="rgba(0, 217, 126, 0.08)", line={"width": 0},
        layer="below",
    )
    # Red zone: overbought + low DCS
    fig.add_shape(
        type="rect", x0=70, x1=100, y0=0, y1=50,
        fillcolor="rgba(230, 57, 70, 0.08)", line={"width": 0},
        layer="below",
    )

    # Threshold lines
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS["green"],
                  annotation_text="BUY DIP (65)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["yellow"],
                  annotation_text="WATCH (50)", annotation_position="right")
    fig.add_vline(x=30, line_dash="dot", line_color=COLORS["blue"],
                  annotation_text="Oversold", annotation_position="top")
    fig.add_vline(x=70, line_dash="dot", line_color=COLORS["red"],
                  annotation_text="Overbought", annotation_position="top")

    fig.update_layout(**_dark_layout(
        title=title,
        xaxis_title="RSI-14",
        yaxis_title="DCS Score",
        xaxis_range=[0, 100],
        yaxis_range=[0, 100],
        height=550,
    ))

    return fig


# ---------------------------------------------------------------------------
# Level 2: War Chest Gauge
# ---------------------------------------------------------------------------

def build_war_chest_gauge(
    actual_pct: float,
    target_pct: float,
    vix_regime: str = "NORMAL",
) -> go.Figure:
    """Build a gauge chart showing war chest % vs VIX-regime target."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=actual_pct * 100,
        delta={"reference": target_pct * 100, "relative": False,
               "increasing": {"color": COLORS["green"]},
               "decreasing": {"color": COLORS["red"]}},
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"War Chest ({vix_regime})",
               "font": {"size": 16, "color": COLORS["text"]}},
        gauge={
            "axis": {"range": [0, 30], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["blue"]},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["grid"],
            "steps": [
                {"range": [0, target_pct * 100], "color": "rgba(230,57,70,0.15)"},
                {"range": [target_pct * 100, 30], "color": "rgba(0,217,126,0.1)"},
            ],
            "threshold": {
                "line": {"color": COLORS["yellow"], "width": 3},
                "thickness": 0.8,
                "value": target_pct * 100,
            },
        },
    ))

    fig.update_layout(**_dark_layout(height=300, margin={"t": 80, "b": 30}))
    return fig


# ---------------------------------------------------------------------------
# Level 2.1: Drawdown Defense Composition
# ---------------------------------------------------------------------------

def build_drawdown_defense_bars(
    classifications: dict[str, str],
    title: str = "Portfolio Defensive Composition",
) -> go.Figure:
    """Build grouped bars showing count of tickers per defense class."""
    class_order = ["HEDGE", "DEFENSIVE", "MODERATE", "CYCLICAL", "AMPLIFIER"]
    class_colors = {
        "HEDGE": COLORS["blue"],
        "DEFENSIVE": COLORS["teal"],
        "MODERATE": COLORS["yellow"],
        "CYCLICAL": COLORS["orange"],
        "AMPLIFIER": COLORS["red"],
    }

    counts = {c: 0 for c in class_order}
    for _, cls in classifications.items():
        if cls in counts:
            counts[cls] += 1

    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=[class_colors.get(c, COLORS["muted"]) for c in counts],
        text=list(counts.values()),
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))

    total = sum(counts.values())
    offense_pct = (counts.get("CYCLICAL", 0) + counts.get("AMPLIFIER", 0)) / total * 100 if total > 0 else 0

    fig.update_layout(**_dark_layout(
        title=f"{title} ({offense_pct:.0f}% Offense)",
        yaxis_title="Ticker Count",
        height=350,
        showlegend=False,
    ))

    return fig


# ---------------------------------------------------------------------------
# Ref: Correlation Heatmap
# ---------------------------------------------------------------------------

def build_correlation_heatmap(
    corr_matrix: dict[str, dict[str, float]],
    title: str = "90-Day Rolling Correlation",
) -> go.Figure:
    """Build a correlation heatmap from the correlation matrix dict."""
    if not corr_matrix:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for correlation analysis",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"size": 16, "color": COLORS["muted"]})
        fig.update_layout(**_dark_layout(height=400, title=title))
        return fig

    tickers = list(corr_matrix.keys())
    z_data = [[corr_matrix[t1].get(t2, 0) for t2 in tickers] for t1 in tickers]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=tickers,
        y=tickers,
        colorscale=[
            [0.0, COLORS["red"]],
            [0.5, COLORS["card"]],
            [1.0, COLORS["green"]],
        ],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(**_dark_layout(
        title=title,
        height=max(350, 50 * len(tickers)),
        xaxis={"side": "bottom", "gridcolor": COLORS["grid"]},
        yaxis={"autorange": "reversed", "gridcolor": COLORS["grid"]},
    ))

    return fig


# ---------------------------------------------------------------------------
# Level 1.5: Sector Rotation RRG
# ---------------------------------------------------------------------------

def build_sector_rrg(
    sector_rankings: list[dict[str, Any]],
    title: str = "Relative Rotation Graph — Sector ETFs",
) -> go.Figure:
    """Build an RRG-style scatter plot of sectors.

    X-axis: Relative strength vs SPY, Y-axis: Momentum.
    Quadrants: LEADING (top-right), WEAKENING (bottom-right),
    LAGGING (bottom-left), IMPROVING (top-left).
    """
    fig = go.Figure()

    if not sector_rankings:
        fig.add_annotation(text="No sector data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"size": 16, "color": COLORS["muted"]})
        fig.update_layout(**_dark_layout(height=450, title=title))
        return fig

    quadrant_colors = {
        "LEADING": COLORS["green"],
        "WEAKENING": COLORS["yellow"],
        "LAGGING": COLORS["red"],
        "IMPROVING": COLORS["blue"],
    }

    for sr in sector_rankings:
        sector = sr.get("sector", "")
        rs = sr.get("rs_vs_spy", 1.0)
        mom = sr.get("momentum", 0.0)
        quadrant = sr.get("quadrant", "LAGGING")
        color = quadrant_colors.get(quadrant, COLORS["muted"])

        fig.add_trace(go.Scatter(
            x=[rs],
            y=[mom],
            mode="markers+text",
            text=[sector],
            textposition="top center",
            textfont={"size": 10, "color": COLORS["text"]},
            marker={"size": 14, "color": color, "line": {"width": 1, "color": COLORS["text"]}},
            name=f"{sector} ({quadrant})",
            hovertemplate=(
                f"<b>{sector}</b><br>"
                f"RS vs SPY: {rs:.3f}<br>"
                f"Momentum: {mom:+.1%}<br>"
                f"Quadrant: {quadrant}<br>"
                "<extra></extra>"
            ),
        ))

    # Quadrant lines
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["muted"])
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])

    # Quadrant labels
    for label, x, y in [
        ("LEADING", 0.85, 0.95), ("WEAKENING", 0.85, 0.05),
        ("LAGGING", 0.15, 0.05), ("IMPROVING", 0.15, 0.95),
    ]:
        fig.add_annotation(
            text=label, xref="paper", yref="paper", x=x, y=y,
            showarrow=False, font={"size": 12, "color": COLORS["muted"]},
            opacity=0.5,
        )

    fig.update_layout(**_dark_layout(
        title=title,
        xaxis_title="Relative Strength vs SPY",
        yaxis_title="Momentum (12-week return)",
        height=500,
    ))

    return fig


# ---------------------------------------------------------------------------
# Ref: Sector Treemap
# ---------------------------------------------------------------------------

def build_sector_treemap(
    scores: dict[str, ScoringResult],
    ticker_sectors: dict[str, str] | None = None,
    ticker_values: dict[str, float] | None = None,
    title: str = "Holdings by Sector",
) -> go.Figure:
    """Build a treemap of holdings grouped by sector.

    Size = position weight (or equal if not provided).
    Color = DCS score (green = high, red = low).
    """
    sectors = ticker_sectors or {}
    values = ticker_values or {}

    labels: list[str] = []
    parents: list[str] = []
    sizes: list[float] = []
    colors: list[float] = []
    hover_text: list[str] = []

    # Root
    all_sectors = sorted(set(sectors.get(t, "Other") for t in scores))

    for sector in all_sectors:
        labels.append(sector)
        parents.append("")
        sizes.append(0)
        colors.append(50)
        hover_text.append(sector)

    for ticker, result in scores.items():
        dcs = result.get("dcs", 0)
        sector = sectors.get(ticker, "Other")
        val = values.get(ticker, 1.0)

        labels.append(ticker)
        parents.append(sector)
        sizes.append(max(val, 0.01))
        colors.append(dcs)
        hover_text.append(f"{ticker}<br>DCS: {dcs:.0f}<br>Sector: {sector}")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=sizes,
        marker={
            "colors": colors,
            "colorscale": [
                [0.0, COLORS["red"]],
                [0.5, COLORS["yellow"]],
                [1.0, COLORS["green"]],
            ],
            "cmin": 0,
            "cmax": 100,
            "colorbar": {"title": "DCS"},
        },
        hovertext=hover_text,
        hoverinfo="text",
        textinfo="label",
        textfont={"size": 11},
    ))

    fig.update_layout(**_dark_layout(title=title, height=450, margin={"t": 50, "b": 10, "l": 10, "r": 10}))
    return fig


# ---------------------------------------------------------------------------
# Signal cards (HTML, not Plotly)
# ---------------------------------------------------------------------------

def build_signal_cards_html(
    scores: dict[str, ScoringResult],
) -> str:
    """Build HTML signal summary cards (STRONG BUY / BUY DIP / WATCH / AVOID)."""
    categories = {
        "STRONG BUY": {"color": "#006400", "tickers": []},
        "HIGH CONVICTION": {"color": "#228B22", "tickers": []},
        "BUY DIP": {"color": "#00d97e", "tickers": []},
        "WATCH": {"color": "#ffd166", "tickers": []},
        "WEAK": {"color": "#e63946", "tickers": []},
    }

    for ticker, result in scores.items():
        signal = result.get("dcs_signal", "")
        dcs = result.get("dcs", 0)
        for cat_name in categories:
            if cat_name in signal.upper():
                categories[cat_name]["tickers"].append((ticker, dcs))
                break
        else:
            if dcs < 50:
                categories["WEAK"]["tickers"].append((ticker, dcs))

    html_parts = ['<div style="display:flex;gap:12px;flex-wrap:wrap;">']
    for name, data in categories.items():
        count = len(data["tickers"])
        if count == 0:
            continue
        ticker_list = ", ".join(f"{t} ({d:.0f})" for t, d in sorted(data["tickers"], key=lambda x: -x[1]))
        html_parts.append(f'''
        <div style="background:{COLORS['card']};border-left:4px solid {data['color']};
                    padding:12px 16px;border-radius:6px;min-width:200px;flex:1;">
            <div style="font-size:13px;color:{data['color']};font-weight:600;">{name}</div>
            <div style="font-size:28px;font-weight:700;color:{COLORS['text']};">{count}</div>
            <div style="font-size:11px;color:{COLORS['muted']};margin-top:4px;">{ticker_list}</div>
        </div>''')
    html_parts.append('</div>')

    return "\n".join(html_parts)


# ---------------------------------------------------------------------------
# Market Context Panel (HTML)
# ---------------------------------------------------------------------------

def build_market_context_html(
    vix_current: float,
    vix_regime: str,
    spy_pct: float,
    spy_above_200d: bool,
    breadth_pct: float,
    effective_bets: float = 0.0,
) -> str:
    """Build HTML KPI cards for market context."""
    vix_color = {
        "COMPLACENT": COLORS["yellow"],
        "NORMAL": COLORS["green"],
        "FEAR": COLORS["orange"],
        "PANIC": COLORS["red"],
    }.get(vix_regime, COLORS["muted"])

    spy_color = COLORS["green"] if spy_above_200d else COLORS["red"]
    breadth_color = COLORS["green"] if breadth_pct > 0.5 else COLORS["red"]
    bets_color = COLORS["green"] if effective_bets >= 20 else (
        COLORS["yellow"] if effective_bets >= 15 else COLORS["red"]
    )

    cards = [
        ("VIX", f"{vix_current:.1f}", vix_regime, vix_color),
        ("SPY vs 200d", f"{spy_pct:+.1%}", "Above" if spy_above_200d else "BELOW", spy_color),
        ("Breadth", f"{breadth_pct:.0%}", "above 200d", breadth_color),
    ]
    if effective_bets > 0:
        cards.append(("Eff. Bets", f"{effective_bets:.1f}", "diversification", bets_color))

    html = ['<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">']
    for label, value, sub, color in cards:
        html.append(f'''
        <div style="background:{COLORS['card']};padding:14px 18px;border-radius:8px;
                    min-width:130px;flex:1;border-top:3px solid {color};">
            <div style="font-size:11px;color:{COLORS['muted']};text-transform:uppercase;">{label}</div>
            <div style="font-size:28px;font-weight:700;color:{color};">{value}</div>
            <div style="font-size:12px;color:{COLORS['muted']};">{sub}</div>
        </div>''')
    html.append('</div>')

    return "\n".join(html)
