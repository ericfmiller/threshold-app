"""Pydantic models for config.yaml validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from threshold.config.defaults import (
    ADVANCED_FACTOR_MOMENTUM,
    ADVANCED_SENTIMENT,
    ADVANCED_TREND_FOLLOWING,
    AGGREGATOR,
    ALDEN_CATEGORIES,
    ALERT_THRESHOLDS,
    ALLOCATION_TARGETS,
    CROSS_ASSET_ETFS,
    DATA_VALIDATION,
    DCS_WEIGHTS,
    DEFENSIVE_FLOOR_BY_REGIME,
    DEFENSIVE_TARGET_BY_BUBBLE,
    DEPLOYMENT,
    DRAWDOWN_DCS_MODIFIERS,
    FALLING_KNIFE_CAPS,
    FQ_WEIGHTS,
    FRED_SERIES,
    GRADE_TO_NUM,
    INTERNATIONAL_RANGE,
    MODIFIERS,
    MQ_WEIGHTS,
    MR_WEIGHTS,
    PORTFOLIO_HRP,
    PORTFOLIO_INVERSE_VOL,
    PORTFOLIO_TAX,
    PROFITABILITY_BLEND,
    REVISION_MOMENTUM,
    RISK_CDAR,
    RISK_CVAR,
    RISK_EBP,
    RISK_MOMENTUM_CRASH,
    RISK_TURBULENCE,
    SA_DEFAULTS,
    SECTOR_CONCENTRATION_LIMIT,
    SECTOR_DCS_THRESHOLDS,
    SECTOR_ETFS,
    SELL_CRITERIA,
    SIGNAL_THRESHOLDS,
    TIINGO_DEFAULTS,
    TO_WEIGHTS,
    VC_WEIGHTS,
    VIX_REGIMES,
    WAR_CHEST_VIX_TARGETS,
    YFINANCE_DEFAULTS,
)

# ---------------------------------------------------------------------------
# Data Source Configs
# ---------------------------------------------------------------------------

class YFinanceConfig(BaseModel):
    enabled: bool = True
    price_period: str = YFINANCE_DEFAULTS["price_period"]
    fundamentals_delay: float = YFINANCE_DEFAULTS["fundamentals_delay"]


class TiingoConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    base_url: str = TIINGO_DEFAULTS["base_url"]
    rate_delay: float = TIINGO_DEFAULTS["rate_delay"]


class SeekingAlphaConfig(BaseModel):
    enabled: bool = True
    api_url: str = SA_DEFAULTS["api_url"]
    export_dir: str = ""
    z_file_dir: str = ""
    stale_threshold_days: int = SA_DEFAULTS["stale_threshold_days"]
    watch_interval_seconds: int = 600
    api_rate_delay: float = 1.0


class FredConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    series: dict[str, str] = Field(default_factory=lambda: dict(FRED_SERIES))


class DataSourcesConfig(BaseModel):
    yfinance: YFinanceConfig = Field(default_factory=YFinanceConfig)
    tiingo: TiingoConfig = Field(default_factory=TiingoConfig)
    seeking_alpha: SeekingAlphaConfig = Field(default_factory=SeekingAlphaConfig)
    fred: FredConfig = Field(default_factory=FredConfig)


# ---------------------------------------------------------------------------
# Scoring Configs
# ---------------------------------------------------------------------------

class ScoringWeightsConfig(BaseModel):
    MQ: int = DCS_WEIGHTS["MQ"]
    FQ: int = DCS_WEIGHTS["FQ"]
    TO: int = DCS_WEIGHTS["TO"]
    MR: int = DCS_WEIGHTS["MR"]
    VC: int = DCS_WEIGHTS["VC"]

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> ScoringWeightsConfig:
        total = self.MQ + self.FQ + self.TO + self.MR + self.VC
        if total != 100:
            raise ValueError(f"DCS weights must sum to 100, got {total}")
        return self


class MQWeightsConfig(BaseModel):
    trend: float = MQ_WEIGHTS["trend"]
    vol_adj_momentum: float = MQ_WEIGHTS["vol_adj_momentum"]
    sa_momentum: float = MQ_WEIGHTS["sa_momentum"]
    relative_strength: float = MQ_WEIGHTS["relative_strength"]


class FQWeightsConfig(BaseModel):
    with_yf_and_revmom: dict[str, float] = Field(
        default_factory=lambda: dict(FQ_WEIGHTS["with_yf_and_revmom"])
    )
    with_yf_only: dict[str, float] = Field(
        default_factory=lambda: dict(FQ_WEIGHTS["with_yf_only"])
    )
    with_revmom_only: dict[str, float] = Field(
        default_factory=lambda: dict(FQ_WEIGHTS["with_revmom_only"])
    )
    base: dict[str, float] = Field(
        default_factory=lambda: dict(FQ_WEIGHTS["base"])
    )


class TOWeightsConfig(BaseModel):
    rsi: float = TO_WEIGHTS["rsi"]
    sma_distance: float = TO_WEIGHTS["sma_distance"]
    bollinger: float = TO_WEIGHTS["bollinger"]
    macd: float = TO_WEIGHTS["macd"]


class MRWeightsConfig(BaseModel):
    vix_contrarian: float = MR_WEIGHTS["vix_contrarian"]
    spy_trend: float = MR_WEIGHTS["spy_trend"]
    breadth: float = MR_WEIGHTS["breadth"]


class VCWeightsConfig(BaseModel):
    sa_value: float = VC_WEIGHTS["sa_value"]
    ev_ebitda_sector: float = VC_WEIGHTS["ev_ebitda_sector"]


class ProfitabilityBlendConfig(BaseModel):
    sa_weight: float = PROFITABILITY_BLEND["sa_weight"]
    novy_marx_weight: float = PROFITABILITY_BLEND["novy_marx_weight"]


class SignalThresholdsConfig(BaseModel):
    strong_buy_dip: int = SIGNAL_THRESHOLDS["strong_buy_dip"]
    high_conviction: int = SIGNAL_THRESHOLDS["high_conviction"]
    buy_dip: int = SIGNAL_THRESHOLDS["buy_dip"]
    watch: int = SIGNAL_THRESHOLDS["watch"]
    weak: int = SIGNAL_THRESHOLDS["weak"]


class ModifiersConfig(BaseModel):
    obv_bullish_max: int = MODIFIERS["obv_bullish_max"]
    rsi_divergence_boost: int = MODIFIERS["rsi_divergence_boost"]
    rsi_divergence_min_dcs: int = MODIFIERS["rsi_divergence_min_dcs"]


class FallingKnifeConfig(BaseModel):
    freefall: dict[str, int] = Field(
        default_factory=lambda: dict(FALLING_KNIFE_CAPS["freefall"])
    )
    downtrend: dict[str, int] = Field(
        default_factory=lambda: dict(FALLING_KNIFE_CAPS["downtrend"])
    )


class DrawdownModifiersConfig(BaseModel):
    HEDGE: int = DRAWDOWN_DCS_MODIFIERS["HEDGE"]
    DEFENSIVE: int = DRAWDOWN_DCS_MODIFIERS["DEFENSIVE"]
    MODERATE: int = DRAWDOWN_DCS_MODIFIERS["MODERATE"]
    CYCLICAL: int = DRAWDOWN_DCS_MODIFIERS["CYCLICAL"]
    AMPLIFIER: int = DRAWDOWN_DCS_MODIFIERS["AMPLIFIER"]


class VIXRegimesConfig(BaseModel):
    COMPLACENT: list[int] = Field(default_factory=lambda: list(VIX_REGIMES["COMPLACENT"]))
    NORMAL: list[int] = Field(default_factory=lambda: list(VIX_REGIMES["NORMAL"]))
    FEAR: list[int] = Field(default_factory=lambda: list(VIX_REGIMES["FEAR"]))
    PANIC: list[int] = Field(default_factory=lambda: list(VIX_REGIMES["PANIC"]))


class RevisionMomentumConfig(BaseModel):
    min_history_weeks: int = REVISION_MOMENTUM["min_history_weeks"]
    min_calendar_days: int = REVISION_MOMENTUM["min_calendar_days"]
    sell_threshold_subgrades: int = REVISION_MOMENTUM["sell_threshold_subgrades"]
    warning_threshold_subgrades: int = REVISION_MOMENTUM["warning_threshold_subgrades"]


class DataValidationConfig(BaseModel):
    min_data_points: int = DATA_VALIDATION["min_data_points"]
    preferred_data_points: int = DATA_VALIDATION["preferred_data_points"]
    stale_gap_days: int = DATA_VALIDATION["stale_gap_days"]
    extreme_move_threshold: float = DATA_VALIDATION["extreme_move_threshold"]


class ScoringConfig(BaseModel):
    weights: ScoringWeightsConfig = Field(default_factory=ScoringWeightsConfig)
    mq_weights: MQWeightsConfig = Field(default_factory=MQWeightsConfig)
    fq_weights: FQWeightsConfig = Field(default_factory=FQWeightsConfig)
    to_weights: TOWeightsConfig = Field(default_factory=TOWeightsConfig)
    mr_weights: MRWeightsConfig = Field(default_factory=MRWeightsConfig)
    vc_weights: VCWeightsConfig = Field(default_factory=VCWeightsConfig)
    profitability_blend: ProfitabilityBlendConfig = Field(
        default_factory=ProfitabilityBlendConfig
    )
    thresholds: SignalThresholdsConfig = Field(default_factory=SignalThresholdsConfig)
    modifiers: ModifiersConfig = Field(default_factory=ModifiersConfig)
    falling_knife: FallingKnifeConfig = Field(default_factory=FallingKnifeConfig)
    drawdown_modifiers: DrawdownModifiersConfig = Field(
        default_factory=DrawdownModifiersConfig
    )
    vix_regimes: VIXRegimesConfig = Field(default_factory=VIXRegimesConfig)
    revision_momentum: RevisionMomentumConfig = Field(
        default_factory=RevisionMomentumConfig
    )
    validation: DataValidationConfig = Field(default_factory=DataValidationConfig)
    crypto_exempt_expiry: str = ""  # ISO date e.g. "2026-11-15"


# ---------------------------------------------------------------------------
# Risk Framework Config (Phase 2B — all disabled by default)
# ---------------------------------------------------------------------------

class EBPConfig(BaseModel):
    enabled: bool = RISK_EBP["enabled"]
    high_risk_threshold: float = RISK_EBP["high_risk_threshold"]
    elevated_threshold: float = RISK_EBP["elevated_threshold"]
    normal_threshold: float = RISK_EBP["normal_threshold"]
    lookback_months: int = RISK_EBP["lookback_months"]


class TurbulenceConfig(BaseModel):
    enabled: bool = RISK_TURBULENCE["enabled"]
    window: int = RISK_TURBULENCE["window"]
    threshold_pctl: float = RISK_TURBULENCE["threshold_pctl"]
    min_assets: int = RISK_TURBULENCE["min_assets"]


class MomentumCrashConfig(BaseModel):
    enabled: bool = RISK_MOMENTUM_CRASH["enabled"]
    lookback_months: int = RISK_MOMENTUM_CRASH["lookback_months"]
    crash_threshold: float = RISK_MOMENTUM_CRASH["crash_threshold"]
    min_weight: float = RISK_MOMENTUM_CRASH["min_weight"]


class CVaRConfig(BaseModel):
    enabled: bool = RISK_CVAR["enabled"]
    alpha: float = RISK_CVAR["alpha"]
    method: str = RISK_CVAR["method"]


class CDaRConfig(BaseModel):
    enabled: bool = RISK_CDAR["enabled"]
    alpha: float = RISK_CDAR["alpha"]


class RiskConfig(BaseModel):
    ebp: EBPConfig = Field(default_factory=EBPConfig)
    turbulence: TurbulenceConfig = Field(default_factory=TurbulenceConfig)
    momentum_crash: MomentumCrashConfig = Field(default_factory=MomentumCrashConfig)
    cvar: CVaRConfig = Field(default_factory=CVaRConfig)
    cdar: CDaRConfig = Field(default_factory=CDaRConfig)


# ---------------------------------------------------------------------------
# Advanced Signals Config (Phase 2C — all disabled by default)
# ---------------------------------------------------------------------------

class TrendFollowingConfig(BaseModel):
    enabled: bool = ADVANCED_TREND_FOLLOWING["enabled"]
    window: int = ADVANCED_TREND_FOLLOWING["window"]
    vol_window: int = ADVANCED_TREND_FOLLOWING["vol_window"]
    mq_blend_weight: float = ADVANCED_TREND_FOLLOWING["mq_blend_weight"]


class FactorMomentumConfig(BaseModel):
    enabled: bool = ADVANCED_FACTOR_MOMENTUM["enabled"]
    lookback_months: int = ADVANCED_FACTOR_MOMENTUM["lookback_months"]
    breadth_threshold_high: float = ADVANCED_FACTOR_MOMENTUM["breadth_threshold_high"]
    breadth_threshold_low: float = ADVANCED_FACTOR_MOMENTUM["breadth_threshold_low"]


class SentimentConfig(BaseModel):
    enabled: bool = ADVANCED_SENTIMENT["enabled"]
    n_components: int = ADVANCED_SENTIMENT["n_components"]
    mr_reduction: float = ADVANCED_SENTIMENT["mr_reduction"]
    overheated_pctl: float = ADVANCED_SENTIMENT["overheated_pctl"]
    depressed_pctl: float = ADVANCED_SENTIMENT["depressed_pctl"]
    min_observations: int = ADVANCED_SENTIMENT["min_observations"]


class AdvancedConfig(BaseModel):
    trend_following: TrendFollowingConfig = Field(default_factory=TrendFollowingConfig)
    factor_momentum: FactorMomentumConfig = Field(default_factory=FactorMomentumConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)


# ---------------------------------------------------------------------------
# Portfolio Construction Config (Phase 2D — all disabled by default)
# ---------------------------------------------------------------------------

class InverseVolConfig(BaseModel):
    enabled: bool = PORTFOLIO_INVERSE_VOL["enabled"]
    eta: float = PORTFOLIO_INVERSE_VOL["eta"]
    window: int = PORTFOLIO_INVERSE_VOL["window"]
    min_periods: int = PORTFOLIO_INVERSE_VOL["min_periods"]


class HRPConfig(BaseModel):
    enabled: bool = PORTFOLIO_HRP["enabled"]
    linkage_method: str = PORTFOLIO_HRP["linkage_method"]
    min_periods: int = PORTFOLIO_HRP["min_periods"]


class TaxConfig(BaseModel):
    enabled: bool = PORTFOLIO_TAX["enabled"]
    lot_method: str = PORTFOLIO_TAX["lot_method"]
    loss_threshold_pct: float = PORTFOLIO_TAX["loss_threshold_pct"]
    wash_sale_window_days: int = PORTFOLIO_TAX["wash_sale_window_days"]
    long_term_days: int = PORTFOLIO_TAX["long_term_days"]


class PortfolioConstructionConfig(BaseModel):
    inverse_vol: InverseVolConfig = Field(default_factory=InverseVolConfig)
    hrp: HRPConfig = Field(default_factory=HRPConfig)
    tax: TaxConfig = Field(default_factory=TaxConfig)


# ---------------------------------------------------------------------------
# Signal Aggregation Config (Phase 2E — disabled by default)
# ---------------------------------------------------------------------------

class AggregatorConfig(BaseModel):
    enabled: bool = AGGREGATOR["enabled"]
    ebp_weight: float = AGGREGATOR["ebp_weight"]
    turbulence_weight: float = AGGREGATOR["turbulence_weight"]
    crash_weight: float = AGGREGATOR["crash_weight"]
    high_risk_threshold: float = AGGREGATOR["high_risk_threshold"]
    elevated_threshold: float = AGGREGATOR["elevated_threshold"]
    high_risk_penalty: int = AGGREGATOR["high_risk_penalty"]
    elevated_penalty: int = AGGREGATOR["elevated_penalty"]


# ---------------------------------------------------------------------------
# Sell Criteria Config
# ---------------------------------------------------------------------------

class SellCriteriaConfig(BaseModel):
    sma_breach_days: int = SELL_CRITERIA["sma_breach_days"]
    sma_breach_warning_days: int = SELL_CRITERIA["sma_breach_warning_days"]
    sma_breach_threshold: float = SELL_CRITERIA["sma_breach_threshold"]
    quant_drop_threshold: float = SELL_CRITERIA["quant_drop_threshold"]
    quant_drop_lookback_days: int = SELL_CRITERIA["quant_drop_lookback_days"]


# ---------------------------------------------------------------------------
# Allocation Config
# ---------------------------------------------------------------------------

class AllocationConfig(BaseModel):
    targets: dict[str, float] = Field(
        default_factory=lambda: dict(ALLOCATION_TARGETS)
    )
    rebalance_trigger: float = 0.05
    international_range: list[float] = Field(
        default_factory=lambda: list(INTERNATIONAL_RANGE)
    )
    sector_concentration_limit: float = SECTOR_CONCENTRATION_LIMIT
    defensive_by_bubble: dict[str, float] = Field(
        default_factory=lambda: dict(DEFENSIVE_TARGET_BY_BUBBLE)
    )
    defensive_by_regime: dict[str, float] = Field(
        default_factory=lambda: dict(DEFENSIVE_FLOOR_BY_REGIME)
    )
    war_chest_vix: dict[str, float] = Field(
        default_factory=lambda: dict(WAR_CHEST_VIX_TARGETS)
    )


# ---------------------------------------------------------------------------
# Deployment Config
# ---------------------------------------------------------------------------

class DeploymentConfig(BaseModel):
    gate3_rsi_max: int = DEPLOYMENT["gate3_rsi_max"]
    gate3_ret_8w_max: float = DEPLOYMENT["gate3_ret_8w_max"]
    gold_rsi_max_sizing: float = DEPLOYMENT["gold_rsi_max_sizing"]


# ---------------------------------------------------------------------------
# Account Config
# ---------------------------------------------------------------------------

class TSPFundConfig(BaseModel):
    name: str
    allocation: float
    etf_proxy: str


class AccountConfig(BaseModel):
    id: str
    name: str
    type: str  # taxable, roth, traditional_ira, inherited_roth, inherited_ira, tsp
    institution: str = "Fidelity"
    tax_treatment: str = "taxable"
    sa_export_prefix: str = ""
    sa_export_prefix_old: str = ""
    is_active: bool = True
    funds: list[TSPFundConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Separate Holdings Config
# ---------------------------------------------------------------------------

class SeparateHoldingConfig(BaseModel):
    symbol: str
    quantity: float
    description: str = ""


# ---------------------------------------------------------------------------
# Alert Config
# ---------------------------------------------------------------------------

class EmailConfig(BaseModel):
    to: str = ""
    from_addr: str = Field("", alias="from")
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    app_password: str = ""

    model_config = {"populate_by_name": True}


class AlertsConfig(BaseModel):
    enabled: bool = True
    email: EmailConfig = Field(default_factory=EmailConfig)
    thresholds: dict[str, int] = Field(
        default_factory=lambda: dict(ALERT_THRESHOLDS)
    )


# ---------------------------------------------------------------------------
# Output Config
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    score_history_dir: str = "~/.threshold/history"
    dashboard_dir: str = "~/.threshold/dashboards"
    narrative_dir: str = "~/.threshold/narratives"
    auto_open_browser: bool = True


# ---------------------------------------------------------------------------
# Database Config
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    path: str = "~/.threshold/threshold.db"


# ---------------------------------------------------------------------------
# Sector & Cross-Asset ETFs
# ---------------------------------------------------------------------------

class ETFsConfig(BaseModel):
    sector_etfs: dict[str, str] = Field(
        default_factory=lambda: dict(SECTOR_ETFS)
    )
    cross_asset_etfs: dict[str, str] = Field(
        default_factory=lambda: dict(CROSS_ASSET_ETFS)
    )
    sector_dcs_thresholds: dict[str, int] = Field(
        default_factory=lambda: dict(SECTOR_DCS_THRESHOLDS)
    )


# ---------------------------------------------------------------------------
# Grade Mappings Config
# ---------------------------------------------------------------------------

class GradeMappingsConfig(BaseModel):
    grade_to_num: dict[str, int] = Field(
        default_factory=lambda: dict(GRADE_TO_NUM)
    )


# ---------------------------------------------------------------------------
# Alden Categories Config
# ---------------------------------------------------------------------------

class AldenCategoryConfig(BaseModel):
    target: list[float]
    tsp_pct: float = 0.0
    cross_cutting: bool = False
    is_catchall: bool = False


# ---------------------------------------------------------------------------
# Top-Level Config
# ---------------------------------------------------------------------------

class ThresholdConfig(BaseModel):
    """Root configuration model for the Threshold application."""

    version: int = 1
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    portfolio_construction: PortfolioConstructionConfig = Field(
        default_factory=PortfolioConstructionConfig
    )
    aggregator: AggregatorConfig = Field(default_factory=AggregatorConfig)
    sell_criteria: SellCriteriaConfig = Field(default_factory=SellCriteriaConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    accounts: list[AccountConfig] = Field(default_factory=list)
    separate_holdings: list[SeparateHoldingConfig] = Field(default_factory=list)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    etfs: ETFsConfig = Field(default_factory=ETFsConfig)
    grade_mappings: GradeMappingsConfig = Field(default_factory=GradeMappingsConfig)
    alden_categories: dict[str, AldenCategoryConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_to_defaults(cls, data: Any) -> Any:
        """YAML parses empty keys as None. Coerce to proper defaults."""
        if isinstance(data, dict):
            for key in ("accounts", "separate_holdings"):
                if key in data and data[key] is None:
                    data[key] = []
            for key in ("alden_categories",):
                if key in data and data[key] is None:
                    data[key] = {}
        return data

    @model_validator(mode="after")
    def set_alden_defaults(self) -> ThresholdConfig:
        if not self.alden_categories:
            self.alden_categories = {
                name: AldenCategoryConfig(
                    target=data["target"],
                    tsp_pct=data.get("tsp_pct", 0.0),
                    cross_cutting=data.get("cross_cutting", False),
                    is_catchall=data.get("is_catchall", False),
                )
                for name, data in ALDEN_CATEGORIES.items()
            }
        return self
