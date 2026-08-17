from pathlib import Path


# ============================================================
# PATH
# ============================================================

ROOT_DIR = Path(".")

FRED_DIR = ROOT_DIR / "Data" / "FRED"
ALFRED_DIR = ROOT_DIR / "Data" / "ALFRED"
RAW_SERIES_CATALOG_FILE = ROOT_DIR / "raw_series_catalog.csv"
EXACT_INDICATOR_GAP_FILE = ROOT_DIR / "exact_indicator_gap_registry.csv"
KOSPI200_MARKET_FILE = ROOT_DIR / "Data" / "MARKET" / "KOSPI.csv"
KOSPI200_MARKET_METADATA_FILE = ROOT_DIR / "Data" / "MARKET" / "KOSPI.metadata.json"
KOREA_STOCK_PANEL_FILE = ROOT_DIR / "Data" / "MARKET" / "KOREA_STOCK_PRICE.csv"
MARKET_BREADTH_FILE = ROOT_DIR / "Data" / "DERIVED" / "korea_market_breadth.parquet"
MARKET_BREADTH_METADATA_FILE = (
    ROOT_DIR / "Data" / "DERIVED" / "korea_market_breadth.metadata.json"
)

# Backward-compatible alias.  The source file contains KOSPI200 (^KS200),
# not the broad KOSPI composite index.
MARKET_FILE = KOSPI200_MARKET_FILE
MARKET_NAME = "KOSPI200"
MARKET_SERIES_NAME = "kospi200"
# Distribution-adjusted ETF history is used only for implementable portfolio
# returns.  The prediction target remains the KOSPI200 index to avoid silently
# changing the model's economic question.
INVESTABLE_MARKET_FILE = (
    ROOT_DIR / "Data" / "MARKET" / "KODEX200_adjusted.csv"
)

RESULT_DIR = ROOT_DIR / "results"
RUNS_DIR = ROOT_DIR / "runs"
ECONOMIC_REVIEW_FILE = ROOT_DIR / "economic_review_registry.csv"
EXTERNAL_REFERENCE_FILE = ROOT_DIR / "external_reference_registry.csv"
OPERATIONAL_RISK_ACCEPTANCE_FILE = ROOT_DIR / "operational_risk_acceptance.csv"
OPERATIONAL_GATE_PROFILE = "research_waiver"


# ============================================================
# FRED SERIES CONFIG
#
# availability_lag:
#   해당 observation이 투자자가 사용 가능해지는 시점을
#   보수적으로 몇 개월 뒤로 미룰 것인가.
#
# 중요:
# 이 값들은 '보수적 모델링 기본값'이지
# 실제 ALFRED vintage/release calendar를 완벽히 재현한 것은 아님.
# ============================================================

FRED_CONFIG = {

    "DCOILWTICO.csv": {
        "name": "wti",
        "freq": "daily",
        "agg": "mean",
        "kind": "price",
        "availability_lag": 0,
        "group": "inflation",
    },

    "EMVOVERALLEMV.csv": {
        "name": "equity_market_volatility",
        "freq": "monthly",
        "agg": "last",
        "kind": "index",
        "availability_lag": 1,
        "group": "risk",
    },

    "EXPINF10YR.csv": {
        "name": "expected_inflation_10y",
        "freq": "monthly",
        "agg": "last",
        "kind": "rate",
        "availability_lag": 1,
        "group": "inflation",
    },

    "HIGHYIELD_SPREAD.csv": {
        "name": "high_yield_spread",
        "freq": "daily",
        "agg": "mean",
        "kind": "spread",
        "availability_lag": 0,
        "group": "credit",
    },

    "KCPRU.csv": {
        "name": "policy_rate_uncertainty",
        "freq": "daily",
        "agg": "mean",
        "kind": "index",
        "availability_lag": 0,
        "group": "risk",
    },

    "M2SL.csv": {
        "name": "m2",
        "freq": "monthly",
        "agg": "last",
        "kind": "quantity",
        "availability_lag": 1,
        "group": "liquidity",
    },

    "RECPROUSM156N.csv": {
        "name": "recession_probability",
        "freq": "monthly",
        "agg": "last",
        "kind": "rate",
        "availability_lag": 1,
        "group": "cycle",
    },

    "T10Y2Y.csv": {
        "name": "term_spread_10y2y",
        "freq": "daily",
        "agg": "mean",
        "kind": "spread",
        "availability_lag": 0,
        "group": "cycle",
    },

    "UMCSENT.csv": {
        "name": "consumer_sentiment",
        "freq": "monthly",
        "agg": "last",
        "kind": "index",
        "availability_lag": 1,
        "group": "cycle",
    },

    "VIXCLS.csv": {
        "name": "vix",
        "freq": "daily",
        "agg": "mean",
        "kind": "index",
        "availability_lag": 0,
        "group": "risk",
    },

    "USEPUINDXD.csv": {
        "name": "us_economic_policy_uncertainty",
        "freq": "daily",
        "agg": "mean",
        "kind": "index",
        "availability_lag": 0,
        "group": "risk",
    },

    "WLEMUINDXD.csv": {
        "name": "equity_market_related_economic_uncertainty",
        "freq": "daily",
        "agg": "mean",
        "kind": "index",
        "availability_lag": 0,
        "group": "risk",
    },

    "DGS10.csv": {
        "name": "treasury_yield_10y",
        "freq": "daily",
        "agg": "mean",
        "kind": "rate",
        "availability_lag": 0,
        "group": "liquidity",
    },

    "CPIAUCSL.csv": {
        "name": "cpi",
        "freq": "monthly",
        "agg": "last",
        "kind": "price_index",
        "availability_lag": 1,
        "group": "inflation",
    },
    "DGS3MO.csv": {
        "name": "cash_yield_3m",
        "freq": "daily",
        "agg": "last",
        "kind": "rate",
        "availability_lag": 0,
        "group": "liquidity",
    },
}

# The CSV catalog is the authoritative registry.  This assignment retains the
# historical import name for downstream notebooks without duplicating config.
from .raw_catalog import fred_config_from_catalog, load_raw_series_catalog

FRED_CONFIG = fred_config_from_catalog(
    load_raw_series_catalog(RAW_SERIES_CATALOG_FILE)
)


# ============================================================
# FACTOR FACTORY
# ============================================================

CHANGE_HORIZONS = [
    1,
    2,
    3,
    6,
    9,
    12,
    18,
    24,
]

MA_WINDOWS = [
    2,
    3,
    4,
    6,
    9,
    12,
    18,
    24,
    36,
    48,
    60,
]

Z_WINDOWS = [
    12,
    24,
    36,
    60,
]

EWM_SPANS = [
    3,
    6,
    12,
    24,
]

VOL_WINDOWS = [
    6,
    12,
    24,
    36,
]

SLOPE_WINDOWS = [
    6,
    12,
    24,
    36,
]


# ============================================================
# TARGET
# ============================================================

# 현재 t월 → 앞으로 3개월 KOSPI
FORECAST_HORIZON = 3

# 3개월 뒤 수익률이 0%보다 높으면 Risk-On = 1
TARGET_RETURN_THRESHOLD = 0.00


# ============================================================
# WALK FORWARD
# ============================================================

# 사진은 24개월도 가능하다고 했지만
# 60개월 정도는 있어야 그나마 덜 흔들림
MIN_TRAIN_MONTHS = 60

# 단일 Factor 대량 검사에서는 속도 때문에
# 3개월마다 재학습
SINGLE_FACTOR_REFIT_EVERY = 3
COMBINATION_SELECTION_REFIT_EVERY = 3

# 최종 모델은 매월 재학습
FINAL_REFIT_EVERY = 1


# ============================================================
# FEATURE SELECTION
# ============================================================

MIN_OOS_PREDICTIONS = 36

# 단일지표 상위권 진단 파일에 남길 최대 개수
TOP_FEATURE_POOL = 400

# 1만 개 변환을 곧바로 조합 탐색에 넣지 않는다. 원천별 상위 3개,
# 경제 그룹별 상위 15개를 순서대로 남겨 winner's curse와 원천 쏠림을
# 완화한다. 최종 모델에서는 동일 원천의 서로 다른 변환을 허용한다.
RAW_TOP_FEATURES_PER_BASE = 3
GROUP_CANDIDATES_PER_GROUP = 15

# 상관관계 중복 제거 후 실제 조합검색 후보
COMBO_CANDIDATE_POOL = 60
NESTED_COMBO_CANDIDATE_POOL = 40
EXHAUSTIVE_COMBO_CANDIDATE_POOL = 8
# 8개 후보에 대한 bounded exhaustive search이므로 7개까지 탐색해도
# 계산량은 제한적이다.
EXHAUSTIVE_MAX_COMBO_SIZE = 7

CORRELATION_THRESHOLD = 0.90

MIN_MODEL_FEATURES = 4
MAX_MODEL_FEATURES = 7
MIN_DISTINCT_GROUPS = 3

# 새 Factor 하나 넣어서 Validation AUC가
# 최소 이만큼 개선되지 않으면 중단
MIN_VALIDATION_IMPROVEMENT = 0.002

# 한 원자료에서 변형 Factor 여러 개가
# 최종모델을 독식하는 것 방지
# Raw-series labels are diagnostic only; multiple transforms may coexist.
MAX_FEATURES_PER_BASE = None

# Category/source concentration is audited, but not hard-capped.  Multiple
# distinct signals from one category remain eligible; correlation pruning
# still removes near-duplicates.
MAX_FEATURES_PER_GROUP = None

# A normal production/research run must actually use an expanded raw-data
# universe.  Partial mode remains available only for smoke tests.
MIN_EXPANDED_RAW_SERIES = 50


# ============================================================
# EWS
# ============================================================

EWS_RISK_OFF = 35
EWS_RISK_ON = 65

MIN_STOCK_WEIGHT = 0.20
MAX_STOCK_WEIGHT = 0.80

TRANSACTION_COST_BPS = 10

POSITION_SIZING_POLICIES = (
    "linear",
    "smoothed_linear",
    "fixed_bin",
    "expanding_percentile",
    "static_50_50",
)
SMOOTHED_LINEAR_SPAN = 3
STATIC_FALLBACK_WEIGHT = 0.50

FIXED_BIN_THRESHOLDS = (
    35.0,
    50.0,
    65.0,
)

FIXED_BIN_WEIGHTS = (
    0.20,
    0.40,
    0.60,
    0.80,
)

PERCENTILE_BREAKS = (
    0.20,
    0.40,
    0.60,
    0.80,
)

PERCENTILE_WEIGHTS = (
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
)

PERCENTILE_MIN_HISTORY = 36

TRANSACTION_COST_SCENARIOS_BPS = (
    0,
    10,
    25,
    50,
)

RESEARCH_HOLDOUT_START = "2020-04-30"
RESEARCH_VALIDATION_MONTHS = 72
# The target is a 3-month forward return.  A 24-month fold contains only
# about eight non-overlapping outcomes, which made the joint AUC/IC direction
# check needlessly noisy.  Twelve non-overlapping outcomes is the minimum
# predeclared evidence per full outer fold.
MIN_NONOVERLAPPING_OUTCOMES_PER_OUTER_FOLD = 12
OUTER_VALIDATION_MONTHS = (
    FORECAST_HORIZON * MIN_NONOVERLAPPING_OUTCOMES_PER_OUTER_FOLD
)
INNER_VALIDATION_MONTHS = 24
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_BLOCK_MONTHS = 3


# ============================================================
# FINAL MODEL
# ============================================================

FINAL_MODEL_TYPE = "logistic"
SINGLE_FACTOR_MODEL_TYPE = "fast_logistic"
SELECTION_MODEL_TYPE = "logistic"

# Structural core of the deployable Logistic champion.  These families come
# from the original EWS materials and the user's requested market-internal
# sensors.  Selection may choose a causal transform within each family but
# may not drop a family.
REQUIRED_CORE_FAMILIES = {
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

# Turnover must describe whether trading value is becoming large relative to
# market capitalization, not merely its level or volatility.
REQUIRED_CORE_TRANSFORM_TOKENS = {
    "turnover_trend": (
        "chg_",
        "slope_",
        "dist_ma_",
        "dist_ewma_",
    ),
}

SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
}

SVM_CALIBRATION_SPLITS = 3

# SVM은 Logistic보다 데이터 좀 더 먹게 하자
SVM_MIN_TRAIN_MONTHS = 84

# 작은 표본의 월간 금융 데이터에 맞춘 저복잡도 MLP challenger.
# 이 사양은 historical holdout을 보며 튜닝하지 않고 pre-holdout에서만
# feature combination을 고른다.
MLP_PARAMS = {
    "hidden_layer_sizes": (8, 4),
    "activation": "tanh",
    "solver": "adam",
    "alpha": 0.05,
    "max_iter": 500,
    "tol": 1e-3,
    "learning_rate_init": 0.001,
    "batch_size": 32,
    "shuffle": False,
    "n_iter_no_change": 30,
}
MLP_MIN_TRAIN_MONTHS = 84
# Validation and frozen production scoring use the same monthly expanding
# refit protocol.  A slower cadence may look better in research, but using it
# only in validation would create a different live model.
MLP_REFIT_EVERY = 1


# ============================================================
# ANALYTICS
# ============================================================

IC_ROLLING_WINDOW = 36
SHARPE_ROLLING_WINDOW = 36
