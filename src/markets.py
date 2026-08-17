"""Market profiles and runtime adaptations for the EWS pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketProfile:
    key: str
    display_name: str
    series_name: str
    ticker: str
    signal_file: Path
    signal_metadata_file: Path
    investable_ticker: str
    investable_instrument: str
    investable_file: Path
    index_start: str
    investable_start: str


MARKET_DIR = Path("Data") / "MARKET"

MARKET_PROFILES = {
    "kospi200": MarketProfile(
        key="kospi200",
        display_name="KOSPI200",
        series_name="kospi200",
        ticker="^KS200",
        signal_file=MARKET_DIR / "KOSPI.csv",
        signal_metadata_file=MARKET_DIR / "KOSPI.metadata.json",
        investable_ticker="069500.KS",
        investable_instrument="KODEX 200 ETF",
        investable_file=MARKET_DIR / "KODEX200_adjusted.csv",
        index_start="1996-01-01",
        investable_start="2002-01-01",
    ),
    "sp500": MarketProfile(
        key="sp500",
        display_name="S&P 500",
        series_name="sp500",
        ticker="^GSPC",
        signal_file=MARKET_DIR / "SP500.csv",
        signal_metadata_file=MARKET_DIR / "SP500.metadata.json",
        investable_ticker="SPY",
        investable_instrument="SPDR S&P 500 ETF Trust",
        investable_file=MARKET_DIR / "SPY_adjusted.csv",
        index_start="1988-01-01",
        investable_start="1993-01-01",
    ),
    "nasdaq100": MarketProfile(
        key="nasdaq100",
        display_name="NASDAQ-100",
        series_name="nasdaq100",
        ticker="^NDX",
        signal_file=MARKET_DIR / "NASDAQ100.csv",
        signal_metadata_file=MARKET_DIR / "NASDAQ100.metadata.json",
        investable_ticker="QQQ",
        investable_instrument="Invesco QQQ Trust",
        investable_file=MARKET_DIR / "QQQ_adjusted.csv",
        index_start="1988-01-01",
        investable_start="1999-01-01",
    ),
}


def get_market_profile(key: str) -> MarketProfile:
    try:
        return MARKET_PROFILES[str(key).strip().lower()]
    except KeyError as exc:
        choices = ", ".join(MARKET_PROFILES)
        raise ValueError(f"Unknown market {key!r}; choose one of: {choices}") from exc


def marketize_raw_catalog(catalog, profile: MarketProfile):
    """Rename only index-derived KOSPI200 sensors for the selected market.

    The Korean constituent-panel sensors remain explicitly Korean cross-market
    explanatory variables; no S&P 500 or NASDAQ-100 constituent data are
    silently claimed.
    """
    result = catalog.copy()
    mask = result["source"].eq("derived_market") & result["name"].str.startswith(
        "kospi200_"
    )
    result.loc[mask, "name"] = result.loc[mask, "name"].str.replace(
        r"^kospi200_", f"{profile.series_name}_", regex=True
    )
    return result


def required_core_families(profile: MarketProfile):
    """Return target-appropriate structural families for model selection."""
    if profile.key == "kospi200":
        return {
            "turnover_trend": (
                "korea_stock_universe_trading_value_to_market_cap",
            ),
            "term_spread": (
                "term_spread_10y2y",
                "term_spread_10y3m",
            ),
            "pairwise_correlation": (
                "korea_stock_universe_pairwise_correlation_1m",
            ),
            "return_skew_1m": (
                "korea_stock_universe_return_skew_1m",
            ),
        }
    prefix = profile.series_name
    return {
        "turnover_trend": (f"{prefix}_trading_volume_ratio_12m",),
        "term_spread": (
            "term_spread_10y2y",
            "term_spread_10y3m",
        ),
        "realized_volatility": (f"{prefix}_realized_volatility_1m",),
        "downside_risk": (f"{prefix}_downside_volatility_1m",),
    }


def optional_market_families(profile: MarketProfile):
    """Return market families that are screened but never forced into a model."""
    if profile.key == "kospi200":
        return {}
    prefix = profile.series_name
    return {
        "absolute_trend": (
            f"{prefix}_momentum_6m",
            f"{prefix}_momentum_12m",
            f"{prefix}_trend_10m",
        ),
    }


def mlp_params_for_market(profile: MarketProfile, default_params):
    """Return the pre-holdout-researched small-sample MLP specification."""
    if profile.key == "kospi200":
        return {
            "hidden_layer_sizes": (4,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.10,
            "max_iter": 1000,
            "tol": 1e-4,
            "shuffle": False,
            "hybrid_mode": "risk_veto",
            "risk_on_threshold": 0.65,
            "mlp_veto_threshold": 0.50,
            "neutral_probability": 0.50,
        }
    return dict(default_params)


def predeclared_mlp_features(profile: MarketProfile):
    """Return a locked MLP set when market-specific prior evidence exists.

    NASDAQ-100 uses the complete US-equity MLP specification first locked on
    the S&P 500 pre-2020 development sample.  Cross-index transfer fixes the
    feature membership before NASDAQ holdout scoring and reduces target-market
    search degrees of freedom.

    S&P 500 uses the candidate locked from the pre-2020 development search.  It
    is deliberately not re-selected from an outer-fold result or from the
    2020+ historical holdout.  The latter remains diagnostic-only evidence.
    """
    if profile.key == "kospi200":
        return (
            "korea_stock_universe_trading_value_to_market_cap__ma_9m_chg_9m",
            "term_spread_10y2y__ma_48m_chg_3m",
            "korea_stock_universe_pairwise_correlation_1m__ewma_12m_chg_24m",
            "korea_stock_universe_return_skew_1m__ma_60m_chg_3m",
        )
    if profile.key == "sp500":
        return (
            "us_corporate_equity_value__dist_ma_3m",
            "term_spread_10y3m__ma_60m_chg_2m",
            "usd_per_aud__ma_12m_chg_6m",
            "us_nonfinancial_profits_after_tax__vol_6m",
        )
    if profile.key == "nasdaq100":
        return (
            "us_corporate_equity_value__dist_ma_3m",
            "term_spread_10y3m__ma_60m_chg_2m",
            "usd_per_aud__ma_12m_chg_6m",
            "us_nonfinancial_profits_after_tax__vol_6m",
        )
    return None


def predeclared_mlp_feature_provenance(profile: MarketProfile):
    """Describe the chronology of a locked market-specific MLP feature set."""
    if profile.key == "kospi200":
        return "locked_pre2020_required_core_candidate_v1"
    if profile.key == "sp500":
        return "locked_pre2020_development_candidate_v1"
    if profile.key == "nasdaq100":
        return "locked_sp500_pre2020_full_spec_transfer_drawdown_target_v1"
    return None


def predeclared_primary_features(profile: MarketProfile):
    """Return an externally fixed Logistic feature set when one is available."""
    if profile.key == "kospi200":
        return (
            "korea_stock_universe_trading_value_to_market_cap__ma_9m_chg_9m",
            "term_spread_10y2y__ma_48m_chg_3m",
            "korea_stock_universe_pairwise_correlation_1m__ewma_12m_chg_24m",
            "korea_stock_universe_return_skew_1m__ma_60m_chg_3m",
        )
    return None


def predeclared_primary_feature_provenance(profile: MarketProfile):
    if profile.key == "kospi200":
        return "original_ews_reference_kospi_structure_cash_excess_v1"
    return None


def mlp_target_spec_for_market(profile: MarketProfile):
    """Return the MLP label definition fixed before historical holdout use."""
    if profile.key == "kospi200":
        return {
            "mode": "cash_excess",
            "provenance": (
                "pre2020_kospi_linear_backbone_mlp_risk_veto_v1;"
                "cash_hurdle_observable_at_signal_date;holdout_safety_veto_only"
            ),
        }
    if profile.key == "nasdaq100":
        return {
            "mode": "future_drawdown",
            "drawdown_threshold": -0.05,
            "provenance": (
                "pre2020_us_equity_tail_risk_target;"
                "sp500_locked_spec_transfer;holdout_safety_veto_only"
            ),
        }
    return {
        "mode": "absolute_positive",
        "return_threshold": 0.0,
        "provenance": "baseline_three_month_positive_return_target",
    }


def primary_target_spec_for_market(profile: MarketProfile):
    """Return the Logistic/SVM target fixed by the market research protocol."""
    if profile.key == "kospi200":
        return {
            "mode": "cash_excess",
            "provenance": (
                "original_ews_kospi_structure_pre2020_fixed_candidate;"
                "cash_hurdle_observable_at_signal_date;holdout_safety_veto_only"
            ),
        }
    return {
        "mode": "absolute_positive",
        "return_threshold": 0.0,
        "provenance": "baseline_three_month_positive_return_target",
    }


def required_core_transform_tokens(profile: MarketProfile):
    del profile
    return {
        "turnover_trend": (
            "chg_",
            "slope_",
            "dist_ma_",
            "dist_ewma_",
        ),
    }
