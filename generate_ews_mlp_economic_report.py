"""Generate a self-contained Korean HTML/SVG explainer for the three-market MLP EWS."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def pct(value, digits=1) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def num(value, digits=2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def one_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one row: {path}")
    return frame.iloc[0]


@dataclass
class Market:
    run_dir: Path
    config: dict
    key: str
    name: str
    instrument: str
    target_mode: str
    model_params: dict
    features: list[str]
    policy: str
    gates: pd.Series
    holdout: pd.Series
    dynamic: pd.Series
    same: pd.Series
    latest: dict
    curve: pd.DataFrame

    @classmethod
    def load(cls, run_dir: str | Path):
        run_dir = Path(run_dir).resolve()
        manifest = json.loads(
            (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        config = manifest["config"]
        gates = one_row(run_dir / "mlp_validation_gates.csv")
        holdout = one_row(run_dir / "mlp_historical_holdout_confirmation.csv")
        performance = pd.read_csv(run_dir / "performance_comparison.csv")
        dynamic = performance.loc[
            performance["strategy"].eq("MLP Dynamic")
        ].iloc[0]
        same = performance.loc[
            performance["strategy"].str.startswith("MLP Same Exposure")
        ].iloc[0]
        latest = json.loads(
            (run_dir / "latest_mlp_ews.json").read_text(encoding="utf-8")
        )
        model_params = json.loads(gates["model_params_json"])
        return cls(
            run_dir=run_dir,
            config=config,
            key=config["market_key"],
            name=config["market_name"],
            instrument=config["investable_instrument"],
            target_mode=gates["target_mode"],
            model_params=model_params,
            features=list(latest["features"]),
            policy=gates["allocation_policy"],
            gates=gates,
            holdout=holdout,
            dynamic=dynamic,
            same=same,
            latest=latest,
            curve=pd.read_csv(run_dir / "mlp_backtest.csv"),
        )

    @property
    def validated(self):
        return (
            as_bool(self.gates["signal_gate"])
            and as_bool(self.gates["portfolio_gate"])
            and as_bool(self.gates["historical_holdout_safety_gate"])
        )

    @property
    def architecture(self):
        return (
            "선형 백본 + MLP 위험 거부권"
            if self.model_params.get("hybrid_mode") == "risk_veto"
            else "독립 소형 MLP"
        )


FEATURE_LABELS = {
    "korea_stock_universe_trading_value_to_market_cap__ma_9m_chg_9m": "KOSPI 거래대금/시가총액 변화",
    "term_spread_10y2y__ma_48m_chg_3m": "미국 10년-2년 금리차 변화",
    "korea_stock_universe_pairwise_correlation_1m__ewma_12m_chg_24m": "KOSPI 종목 간 상관관계 변화",
    "korea_stock_universe_return_skew_1m__ma_60m_chg_3m": "KOSPI 수익률 왜도 변화",
    "us_corporate_equity_value__dist_ma_3m": "미국 기업 주식가치의 단기 추세 이격",
    "term_spread_10y3m__ma_60m_chg_2m": "미국 10년-3개월 금리차 변화",
    "usd_per_aud__ma_12m_chg_6m": "AUD/USD 위험선호·달러 유동성 변화",
    "us_nonfinancial_profits_after_tax__vol_6m": "미국 비금융기업 세후이익 변동성",
}

TARGET_LABELS = {
    "cash_excess": "향후 3개월 지수수익률이 신호일에 알 수 있던 현금수익률을 초과하는가",
    "absolute_positive": "향후 3개월 지수수익률이 0%보다 높은가",
    "future_drawdown": "향후 3개월 안에 -5% 이하 낙폭이 발생하지 않는가",
}

FAILURE_LABELS = {
    "auc_not_above_0.5": "AUC가 0.5 이하",
    "rank_ic_not_above_0": "Rank IC가 0 이하",
    "insufficient_predictions": "예측 coverage 부족",
    "median_fold_sharpe_gain_below_0.10": "fold 중앙 Sharpe 개선이 0.10 미만",
    "positive_fold_ratio_below_two_thirds": "양의 능동수익 fold가 2/3 미만",
    "active_return_not_positive_at_10_and_25_bps": "10·25bp 모두에서 능동수익이 양수가 아님",
    "drawdown_difference_below_minus_0.03": "동일노출 대비 MDD 열위가 3%p 초과",
    "passed": "통과",
}


def badge(ok, yes="통과", no="실패"):
    return f'<span class="badge {"ok" if ok else "bad"}">{yes if ok else no}</span>'


def line_points(values, width=620, height=190, pad=22, value_range=None):
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        return ""
    if len(values) > 100:
        indices = np.linspace(0, len(values) - 1, 100).astype(int)
        values = values[indices]
    lo, hi = (
        value_range
        if value_range is not None
        else (float(values.min()), float(values.max()))
    )
    if hi <= lo:
        hi = lo + 1.0
    xs = np.linspace(pad, width - pad, len(values))
    ys = height - pad - (values - lo) / (hi - lo) * (height - 2 * pad)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def svg_curves(markets):
    panels = []
    for i, market in enumerate(markets):
        x = 16 + i * 344
        combined = pd.concat(
            [market.curve["strategy_curve"], market.curve["same_exposure_curve"]]
        )
        combined = pd.to_numeric(combined, errors="coerce").dropna()
        shared_range = (float(combined.min()), float(combined.max()))
        strategy = line_points(
            market.curve["strategy_curve"], 312, 150, 18, shared_range
        )
        same = line_points(
            market.curve["same_exposure_curve"], 312, 150, 18, shared_range
        )
        panels.append(
            f"""<g transform="translate({x},50)">
              <rect width="320" height="190" rx="16" class="svg-panel"/>
              <text x="18" y="26" class="svg-head">{esc(market.name)}</text>
              <polyline points="{same}" transform="translate(4,27)" class="line same"/>
              <polyline points="{strategy}" transform="translate(4,27)" class="line mlp"/>
              <text x="18" y="178" class="svg-note">Sharpe {num(market.dynamic['Sharpe'],3)} · 동일노출 {num(market.same['Sharpe'],3)}</text>
            </g>"""
        )
    return f"""<svg viewBox="0 0 1060 260" role="img" aria-labelledby="curve-title curve-desc">
      <title id="curve-title">세 시장 MLP와 동일 주식노출 누적곡선</title>
      <desc id="curve-desc">2020년 4월 이후 각 시장에서 동적 MLP 전략과 평균 주식비중이 같은 비교 전략의 누적곡선을 보여준다.</desc>
      <text x="16" y="25" class="svg-title">역사적 확인 구간: MLP 동적배분 vs 동일 평균 주식노출</text>
      <line x1="740" y1="20" x2="775" y2="20" class="line mlp"/><text x="782" y="24" class="svg-note">MLP</text>
      <line x1="870" y1="20" x2="905" y2="20" class="line same"/><text x="912" y="24" class="svg-note">동일노출</text>
      {''.join(panels)}
    </svg>"""


def svg_sharpe(markets):
    width, height, base = 920, 305, 240
    maximum = max(float(m.dynamic["Sharpe"]) for m in markets) * 1.2
    marks = []
    for i, market in enumerate(markets):
        center = 175 + i * 285
        for j, row in enumerate((market.dynamic, market.same)):
            value = float(row["Sharpe"])
            bar_h = value / maximum * 170
            x = center - 46 + j * 62
            cls = "bar-mlp" if j == 0 else "bar-same"
            marks.append(
                f'<rect x="{x}" y="{base-bar_h:.1f}" width="46" height="{bar_h:.1f}" rx="7" class="{cls}"/>'
                f'<text x="{x+23}" y="{base-bar_h-8:.1f}" text-anchor="middle" class="svg-value">{value:.3f}</text>'
            )
        marks.append(
            f'<text x="{center+8}" y="270" text-anchor="middle" class="svg-label">{esc(market.name)}</text>'
        )
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="sharpe-title sharpe-desc">
      <title id="sharpe-title">시장별 MLP 전략 샤프지수 비교</title>
      <desc id="sharpe-desc">각 시장의 MLP 동적 전략과 동일 평균 주식노출 전략의 샤프지수를 비교한다.</desc>
      <text x="20" y="28" class="svg-title">Sharpe: 타이밍 효과와 단순 주식비중 효과 분리</text>
      <rect x="650" y="17" width="12" height="12" rx="3" class="bar-mlp"/><text x="670" y="28" class="svg-note">MLP 동적</text>
      <rect x="770" y="17" width="12" height="12" rx="3" class="bar-same"/><text x="790" y="28" class="svg-note">동일노출</text>
      <line x1="40" y1="{base}" x2="880" y2="{base}" class="axis"/>{''.join(marks)}
    </svg>"""


def svg_economic_chain():
    labels = [
        ("현금흐름", "기업이익·경기"),
        ("할인율", "장단기 금리차"),
        ("유동성", "통화·달러·거래대금"),
        ("위험선호", "변동성·왜도·상관"),
        ("3개월 EWS", "수익/안전 확률"),
        ("자산배분", "ETF + 현금"),
    ]
    nodes = []
    for i, (title, sub) in enumerate(labels):
        x = 14 + i * 166
        cls = "economic-node output" if i >= 4 else "economic-node"
        nodes.append(
            f'<g transform="translate({x},48)"><rect width="142" height="78" rx="14" class="{cls}"/>'
            f'<text x="71" y="31" text-anchor="middle" class="svg-head">{title}</text>'
            f'<text x="71" y="53" text-anchor="middle" class="svg-note">{sub}</text></g>'
        )
        if i < len(labels) - 1:
            nodes.append(
                f'<path d="M {x+143} 87 L {x+162} 87" class="arrow" marker-end="url(#arrowhead)"/>'
            )
    return f"""<svg viewBox="0 0 1020 155" role="img" aria-labelledby="econ-title econ-desc">
      <title id="econ-title">EWS의 경제적 전달경로</title>
      <desc id="econ-desc">현금흐름, 할인율, 유동성, 위험선호 정보가 3개월 시장 안전확률을 거쳐 ETF와 현금 비중으로 변환된다.</desc>
      <defs><marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" class="arrow-fill"/></marker></defs>
      <text x="14" y="24" class="svg-title">경제 변수 → 기대수익과 위험의 가격 → 실행 가능한 비중</text>{''.join(nodes)}
    </svg>"""


def svg_architecture():
    return """<svg viewBox="0 0 1020 320" role="img" aria-labelledby="arch-title arch-desc">
      <title id="arch-title">시장별 MLP 구조</title>
      <desc id="arch-desc">KOSPI는 선형 백본의 공격적 신호를 MLP가 방어적으로 거부하며, 미국 시장은 검증된 독립 MLP 확률을 백분위 비중으로 변환한다.</desc>
      <defs><marker id="arch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" class="arrow-fill"/></marker></defs>
      <text x="16" y="28" class="svg-title">작은 표본에는 신경망의 역할을 제한하고, 충분히 안정적인 시장에는 자율성을 부여</text>
      <text x="18" y="70" class="lane-label">KOSPI200</text>
      <g transform="translate(150,48)"><rect width="150" height="58" rx="13" class="economic-node"/><text x="75" y="25" text-anchor="middle" class="svg-head">표준화 4개 Factor</text><text x="75" y="43" text-anchor="middle" class="svg-note">현금초과 목표</text></g>
      <path d="M302 77 L355 77" class="arrow" marker-end="url(#arch-arrow)"/>
      <g transform="translate(360,40)"><rect width="170" height="74" rx="13" class="economic-node"/><text x="85" y="27" text-anchor="middle" class="svg-head">Logistic 백본</text><text x="85" y="49" text-anchor="middle" class="svg-note">안정적 방향성</text></g>
      <g transform="translate(360,130)"><rect width="170" height="74" rx="13" class="economic-node neural"/><text x="85" y="27" text-anchor="middle" class="svg-head">4-unit tanh MLP</text><text x="85" y="49" text-anchor="middle" class="svg-note">비선형 위험 확인</text></g>
      <path d="M302 77 C330 77 330 167 355 167" class="arrow" marker-end="url(#arch-arrow)"/>
      <path d="M532 77 L610 77" class="arrow" marker-end="url(#arch-arrow)"/><path d="M532 167 L610 100" class="arrow" marker-end="url(#arch-arrow)"/>
      <g transform="translate(615,58)"><rect width="202" height="88" rx="14" class="economic-node output"/><text x="101" y="27" text-anchor="middle" class="svg-head">MLP 위험 거부권</text><text x="101" y="49" text-anchor="middle" class="svg-note">백본 ≥65%, MLP &lt;50%</text><text x="101" y="68" text-anchor="middle" class="svg-note">→ 50% 중립으로 하향</text></g>
      <path d="M819 102 L875 102" class="arrow" marker-end="url(#arch-arrow)"/>
      <g transform="translate(880,72)"><rect width="120" height="60" rx="13" class="economic-node output"/><text x="60" y="25" text-anchor="middle" class="svg-head">고정 구간</text><text x="60" y="44" text-anchor="middle" class="svg-note">20/40/60/80%</text></g>
      <line x1="18" y1="225" x2="1000" y2="225" class="axis"/>
      <text x="18" y="269" class="lane-label">S&amp;P 500 · NASDAQ-100</text>
      <g transform="translate(250,244)"><rect width="165" height="58" rx="13" class="economic-node"/><text x="82" y="25" text-anchor="middle" class="svg-head">표준화 4개 Factor</text><text x="82" y="43" text-anchor="middle" class="svg-note">시장별 목표</text></g>
      <path d="M417 273 L500 273" class="arrow" marker-end="url(#arch-arrow)"/>
      <g transform="translate(505,244)"><rect width="176" height="58" rx="13" class="economic-node neural"/><text x="88" y="25" text-anchor="middle" class="svg-head">8×4 tanh MLP</text><text x="88" y="43" text-anchor="middle" class="svg-note">독립 확률</text></g>
      <path d="M683 273 L765 273" class="arrow" marker-end="url(#arch-arrow)"/>
      <g transform="translate(770,244)"><rect width="190" height="58" rx="13" class="economic-node output"/><text x="95" y="25" text-anchor="middle" class="svg-head">확장 백분위 비중</text><text x="95" y="43" text-anchor="middle" class="svg-note">과거 분포만 사용</text></g>
    </svg>"""


def svg_code_flow():
    steps = [
        ("1", "src/data.py", "FRED·시장·ETF 월말 통합"),
        ("2", "src/features.py", "가용시점 지연 후 Factor 생성"),
        ("3", "src/modeling.py", "3개월 목표·MLP 확률 산출"),
        ("4", "src/validation.py", "purge된 walk-forward·gate"),
        ("5", "src/position_sizing.py", "확률을 20~80% 비중으로"),
        ("6", "src/backtest.py", "ETF+현금·비용·동일노출 비교"),
    ]
    items = []
    for i, (n, file, label) in enumerate(steps):
        x = 12 + i * 166
        items.append(
            f'<g transform="translate({x},48)"><circle cx="68" cy="20" r="18" class="step-circle"/>'
            f'<text x="68" y="25" text-anchor="middle" class="step-number">{n}</text>'
            f'<text x="68" y="59" text-anchor="middle" class="svg-head">{file}</text>'
            f'<text x="68" y="81" text-anchor="middle" class="svg-note">{label}</text></g>'
        )
        if i < len(steps) - 1:
            items.append(
                f'<path d="M {x+142} 68 L {x+159} 68" class="arrow" marker-end="url(#code-arrow)"/>'
            )
    return f"""<svg viewBox="0 0 1020 150" role="img" aria-labelledby="code-title code-desc">
      <title id="code-title">EWS 코드 실행 흐름</title><desc id="code-desc">데이터 통합부터 특징 생성, 모델링, 검증, 비중 결정, 백테스트까지의 실제 코드 모듈 순서다.</desc>
      <defs><marker id="code-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" class="arrow-fill"/></marker></defs>
      <text x="12" y="24" class="svg-title">run_pipeline.py가 아래 모듈을 순서대로 호출한다</text>{''.join(items)}
    </svg>"""


def market_cards(markets):
    cards = []
    for market in markets:
        lift = float(market.dynamic["Sharpe"]) - float(market.same["Sharpe"])
        cards.append(
            f"""<article class="market-card">
              <div class="card-top"><div><p class="kicker">시장별 특화 · {esc(market.architecture)}</p><h2>{esc(market.name)}</h2><p>{esc(market.instrument)}</p></div>{badge(market.validated, '특화모델 통과', '검증 실패')}</div>
              <div class="score-grid"><div><span>MLP Sharpe</span><strong>{num(market.dynamic['Sharpe'],3)}</strong></div><div><span>동일노출</span><strong>{num(market.same['Sharpe'],3)}</strong></div><div><span>Sharpe 차이</span><strong class="positive">{lift:+.3f}</strong></div><div><span>최대낙폭</span><strong>{pct(market.dynamic['MaxDrawdown'])}</strong></div></div>
              <dl><div><dt>2020+ CAGR</dt><dd>{pct(market.dynamic['CAGR'])} vs {pct(market.same['CAGR'])}</dd></div><div><dt>사전정책</dt><dd>{esc(market.policy)}</dd></div><div><dt>목표</dt><dd>{esc(market.target_mode)}</dd></div><div><dt>최신 EWS / 비중</dt><dd>{float(market.latest['ews']):.1f} / {pct(market.latest['target_stock_weight'],0)}</dd></div></dl>
            </article>"""
        )
    return "".join(cards)


def gate_rows(markets):
    rows = []
    for m in markets:
        rows.append(
            f"<tr><td>{esc(m.name)}</td><td>{badge(as_bool(m.gates['signal_gate']))}</td>"
            f"<td>{badge(as_bool(m.gates['portfolio_gate']))}</td>"
            f"<td>{badge(as_bool(m.gates['historical_holdout_safety_gate']))}</td>"
            f"<td>{badge(as_bool(m.gates['strict_operational_gate']), '가능', '차단')}</td>"
            f"<td>{esc(m.gates['failed_conditions'])}</td></tr>"
        )
    return "".join(rows)


def feature_rows(markets):
    rows = []
    for market in markets:
        for feature in market.features:
            rows.append(
                f"<tr><td>{esc(market.name)}</td><td>{esc(FEATURE_LABELS.get(feature, feature))}</td><td><code>{esc(feature)}</code></td></tr>"
            )
    return "".join(rows)


def diagnosis_rows(research_dir: Path):
    summary = pd.read_csv(research_dir / "cross_market_guarded_mlp_summary.csv")
    keep = summary.loc[
        summary["market_key"].eq("kospi200")
        & summary["candidate"].isin(["pure_mlp", "risk_veto_0.50"])
    ]
    rows = []
    labels = {"pure_mlp": "기존 독립 MLP", "risk_veto_0.50": "개선: MLP 위험 거부권"}
    for row in keep.itertuples():
        rows.append(
            f"<tr><td>{labels[row.candidate]}</td><td>{num(row.aggregate_auc,3)}</td>"
            f"<td>{num(row.aggregate_rank_ic,3)}</td><td>{pct(row.fold_joint_direction_pass_ratio,0)}</td>"
            f"<td>{esc(row.selected_policy)}</td><td>{num(row.median_fold_sharpe_difference,3)}</td>"
            f"<td>{pct(row.annualized_active_return_25bps,2)}</td><td>{badge(as_bool(row.joint_gate))}</td></tr>"
        )
    return "".join(rows)


def fairness_rows(frame):
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"<tr><td>{esc(row.market_name)}</td>"
            f"<td>{num(row.pre2020_auc,3)}</td><td>{num(row.pre2020_rank_ic,3)}</td>"
            f"<td>{pct(row.pre2020_fold_joint_direction_ratio,0)}</td>"
            f"<td>{badge(as_bool(row.pre2020_signal_gate))}</td>"
            f"<td>{badge(as_bool(row.pre2020_portfolio_gate))}</td>"
            f"<td>{badge(as_bool(row.holdout_safety_gate))}</td>"
            f"<td>{badge(as_bool(row.fully_validated_historical), '범용성 통과', '미통과')}</td></tr>"
        )
    return "".join(rows)


def translated_failures(value):
    keys = str(value).split("|")
    return " · ".join(FAILURE_LABELS.get(key, key) for key in keys)


def fairness_pre2020_rows(frame):
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"<tr><td>{esc(row.market_name)}</td>"
            f"<td>{num(row.pre2020_auc,3)}</td><td>{num(row.pre2020_rank_ic,3)}</td>"
            f"<td>{pct(row.pre2020_fold_joint_direction_ratio,0)}</td>"
            f"<td>{num(row.pre2020_median_fold_sharpe_difference,3)}</td>"
            f"<td>{pct(row.pre2020_positive_fold_ratio,0)}</td>"
            f"<td>{pct(row.pre2020_active_return_25bps,2)}</td>"
            f"<td>{pct(row.pre2020_drawdown_difference_25bps,2)}</td>"
            f"<td>{badge(as_bool(row.pre2020_signal_gate))} {badge(as_bool(row.pre2020_portfolio_gate))}</td>"
            f"<td>{badge(as_bool(row.pre2020_joint_gate))}</td></tr>"
        )
    return "".join(rows)


def fairness_holdout_rows(frame):
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"<tr><td>{esc(row.market_name)}</td>"
            f"<td>{num(row.holdout_auc,3)}</td><td>{num(row.holdout_rank_ic,3)}</td>"
            f"<td>{num(row.holdout_dynamic_sharpe,3)} / {num(row.holdout_same_exposure_sharpe,3)}</td>"
            f"<td>{float(row.holdout_sharpe_difference):+.3f}</td>"
            f"<td>{pct(row.holdout_dynamic_cagr,2)} / {pct(row.holdout_same_exposure_cagr,2)}</td>"
            f"<td>{float(row.holdout_cagr_difference) * 100:+.2f}%p</td>"
            f"<td>{pct(row.holdout_dynamic_max_drawdown,2)} / {pct(row.holdout_same_exposure_max_drawdown,2)}</td>"
            f"<td>{badge(as_bool(row.holdout_safety_gate), '안전 veto 통과', '안전 veto 실패')}</td></tr>"
        )
    return "".join(rows)


def fold_evidence(fairness_dir: Path, track: str, frame):
    blocks = []
    for summary in frame.itertuples():
        stem = f"{track}_{summary.market_key}"
        folds = pd.read_csv(fairness_dir / f"{stem}_pre2020_signal_folds.csv")
        gate = pd.read_csv(fairness_dir / f"{stem}_pre2020_policy_gate.csv")
        tactical = gate.loc[gate["policy"].eq("expanding_percentile")].iloc[0]
        rows = []
        for row in folds.itertuples():
            rows.append(
                f"<tr><td>{int(row.fold)}</td><td>{esc(row.start)} ~ {esc(row.end)}</td>"
                f"<td>{int(row.observations)}/{int(row.expected_observations)}</td>"
                f"<td>{num(row.auc,3)}</td><td>{num(row.rank_ic,3)}</td>"
                f"<td>{badge(as_bool(row.joint_direction_passed))}</td>"
                f"<td>{esc(translated_failures(row.failure_reason))}</td></tr>"
            )
        blocks.append(
            f"""<details><summary>{esc(summary.market_name)} fold 근거 · Signal {('통과' if as_bool(summary.pre2020_signal_gate) else '실패')} · Portfolio {('통과' if as_bool(summary.pre2020_portfolio_gate) else '실패')}</summary>
            <div class="details-body"><div class="table-wrap"><table><thead><tr><th>Fold</th><th>평가기간</th><th>관측/예정</th><th>AUC</th><th>Rank IC</th><th>공동 방향</th><th>판정 사유</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
            <p class="note"><strong>포트폴리오 판정:</strong> 중앙 fold Sharpe Δ {num(tactical['median_fold_Sharpe_difference'],3)}, 양의 fold {pct(tactical['positive_fold_ratio'],0)}, 10bp 능동수익 {pct(tactical['annualized_active_return_10bps'],2)}, 25bp 능동수익 {pct(tactical['annualized_active_return_25bps'],2)}, 25bp MDD 차이 {pct(tactical['drawdown_difference_25bps'],2)}. {esc(translated_failures(tactical['failed_conditions']))}</p></div></details>"""
        )
    return "".join(blocks)


def specialized_evidence_rows(markets):
    rows = []
    for market in markets:
        rows.append(
            f"<tr><td>{esc(market.name)}</td><td>{esc(market.architecture)}</td>"
            f"<td>{esc(TARGET_LABELS.get(market.target_mode, market.target_mode))}</td>"
            f"<td>{num(market.gates['aggregate_auc'],3)}</td>"
            f"<td>{num(market.gates['aggregate_rank_ic'],3)}</td>"
            f"<td>{pct(market.gates['fold_joint_direction_pass_ratio'],0)}</td>"
            f"<td>{num(market.holdout['dynamic_sharpe'],3)} / {num(market.holdout['same_exposure_sharpe'],3)}</td>"
            f"<td>{pct(market.holdout['dynamic_cagr'],2)} / {pct(market.holdout['same_exposure_cagr'],2)}</td>"
            f"<td>{pct(market.holdout['dynamic_max_drawdown'],2)} / {pct(market.holdout['same_exposure_max_drawdown'],2)}</td>"
            f"<td>{badge(market.validated, '역사 게이트 통과', '역사 게이트 실패')}</td></tr>"
        )
    return "".join(rows)


def operational_rows(markets):
    rows = []
    for market in markets:
        gates = market.gates
        rows.append(
            f"<tr><td>{esc(market.name)}</td>"
            f"<td>{badge(as_bool(gates['point_in_time_vintage_gate']))}</td>"
            f"<td>{badge(as_bool(gates['release_timing_gate']))}</td>"
            f"<td>{badge(as_bool(gates['investable_return_source_gate']))}</td>"
            f"<td>{badge(as_bool(gates['economic_review_approved_gate']))}</td>"
            f"<td>{badge(as_bool(gates['publication_lag_review_gate']))}</td>"
            f"<td>{badge(as_bool(gates['duplicate_information_review_gate']))}</td>"
            f"<td>{badge(as_bool(gates['capital_use_allowed']), '허용', '차단')}</td></tr>"
        )
    return "".join(rows)


def baseline_model_rows(baseline_dir: Path):
    metrics = pd.read_csv(baseline_dir / "model_comparison.csv")
    ic = pd.read_csv(baseline_dir / "model_ic_comparison.csv")
    merged = metrics.merge(ic, on="model", how="left")
    rows = []
    for row in merged.itertuples():
        rows.append(
            f"<tr><td>{esc(row.model)}</td><td>{int(row.n)}</td><td>{num(row.auc,3)}</td>"
            f"<td>{num(row.brier,3)}</td><td>{pct(row.accuracy,1)}</td>"
            f"<td>{pct(row.naive_accuracy,1)}</td><td>{float(row.accuracy_lift_vs_naive) * 100:+.1f}%p</td>"
            f"<td>{num(row.rank_ic,3)}</td><td>{pct(row.rolling_ic_positive_ratio,1)}</td></tr>"
        )
    return "".join(rows)


def baseline_performance_rows(baseline_dir: Path):
    performance = pd.read_csv(baseline_dir / "performance_comparison.csv")
    keep = performance.loc[
        performance["strategy"].str.contains("Dynamic|Same Exposure", regex=True)
    ]
    rows = []
    for row in keep.itertuples():
        rows.append(
            f"<tr><td>{esc(row.model)}</td><td>{esc(row.strategy)}</td>"
            f"<td>{esc(row.Start)} ~ {esc(row.End)}</td><td>{int(row.Months)}</td>"
            f"<td>{pct(row.CAGR,2)}</td><td>{num(row.Sharpe,3)}</td>"
            f"<td>{pct(row.MaxDrawdown,2)}</td><td>{num(row.Calmar,3)}</td></tr>"
        )
    return "".join(rows)


def factor_funnel_rows(baseline_dir: Path):
    summary = pd.read_csv(baseline_dir / "factor_universe_summary.csv")
    labels = {
        "configured_raw_series": "카탈로그 원천",
        "available_raw_series": "실제 확보 원천",
        "research_eligible_raw_series": "연구 적격 원천",
        "factor_factory_output": "Factory 변환",
        "all_factor_matrix": "전체 후보 Factor",
        "robust_univariate_screen": "단일 Factor 검정",
        "top_ranked_pool_limit": "상위 후보 pool",
        "correlation_pruned_limit": "상관 중복 제거 후",
        "bounded_exhaustive_limit": "조합 탐색 핵심 후보",
        "final_model_feature_minimum": "최종 최소 변수 수",
        "final_model_feature_limit": "최종 최대 변수 수",
    }
    return "".join(
        f"<tr><td>{esc(labels.get(row.stage, row.stage))}</td><td>{int(row.count):,}</td><td><code>{esc(row.stage)}</code></td></tr>"
        for row in summary.itertuples()
    )


def protocol_feature_rows(protocol):
    return "".join(
        f"<tr><td>{i}</td><td>{esc(FEATURE_LABELS.get(feature, feature))}</td><td><code>{esc(feature)}</code></td></tr>"
        for i, feature in enumerate(protocol["features"], 1)
    )


def protocol_fold_rows(protocol):
    return "".join(
        f"<tr><td>{int(fold['fold'])}</td><td>{esc(fold['outer_start'])}</td><td>{esc(fold['outer_end'])}</td></tr>"
        for fold in protocol["common_outer_folds"]
    )


def svg_validation_timeline(protocol):
    folds = protocol["common_outer_folds"]
    marks = []
    for i, fold in enumerate(folds):
        x = 28 + i * 190
        marks.append(
            f'<g transform="translate({x},65)"><rect width="170" height="76" rx="13" class="economic-node"/>'
            f'<text x="85" y="25" text-anchor="middle" class="svg-head">Outer fold {fold["fold"]}</text>'
            f'<text x="85" y="48" text-anchor="middle" class="svg-note">{fold["outer_start"][:7]} ~ {fold["outer_end"][:7]}</text>'
            f'<text x="85" y="64" text-anchor="middle" class="svg-note">선택·게이트 근거</text></g>'
        )
    marks.append(
        f'<g transform="translate(800,65)"><rect width="190" height="76" rx="13" class="economic-node output"/>'
        f'<text x="95" y="25" text-anchor="middle" class="svg-head">역사적 holdout</text>'
        f'<text x="95" y="48" text-anchor="middle" class="svg-note">{protocol["historical_holdout_start"][:7]} ~ {protocol["historical_holdout_end"][:7]}</text>'
        f'<text x="95" y="64" text-anchor="middle" class="svg-note">실패 시 veto만 허용</text></g>'
    )
    return f"""<svg viewBox="0 0 1020 175" role="img" aria-labelledby="timeline-title timeline-desc">
      <title id="timeline-title">공정 비교 검증 구간</title>
      <desc id="timeline-desc">2008년 11월부터 2020년 3월까지 네 개 outer fold가 모델 선택과 게이트 근거이며, 2020년 4월 이후는 실패 시 중단시키는 역사적 안전성 확인에만 사용된다.</desc>
      <text x="28" y="27" class="svg-title">2020-03 이전에 사양을 잠그고, 이후 구간은 좋은 결과를 골라내는 데 쓰지 않았다</text>
      <line x1="28" y1="103" x2="990" y2="103" class="axis"/>{''.join(marks)}</svg>"""


def svg_claim_ladder(markets, same_spec, lomo):
    specific_count = sum(m.validated for m in markets)
    same_count = int(same_spec["fully_validated_historical"].sum())
    lomo_count = int(lomo["fully_validated_historical"].sum())
    rows = [
        ("공통 검증 프레임", 3, "동일 gate·비용·누수 방지", True),
        ("동일 사양, 시장별 재학습", same_count, "입력·목표·MLP·정책 동일", same_count == 3),
        ("Leave-one-market-out", lomo_count, "대상 시장 label 완전 제외", lomo_count == 3),
        ("시장별 특화 전략", specific_count, "서로 다른 목표·모델·정책", specific_count == 3),
    ]
    marks = []
    for i, (label, count, note, passed) in enumerate(rows):
        y = 58 + i * 67
        cls = "claim-pass" if passed else "claim-fail"
        marks.append(
            f'<text x="20" y="{y+23}" class="svg-head">{label}</text>'
            f'<text x="290" y="{y+23}" class="svg-note">{note}</text>'
            f'<rect x="650" y="{y}" width="230" height="38" rx="10" class="{cls}"/>'
            f'<text x="765" y="{y+25}" text-anchor="middle" class="svg-head">{count}/3 통과</text>'
        )
    return f"""<svg viewBox="0 0 920 340" role="img" aria-labelledby="claim-title claim-desc">
      <title id="claim-title">EWS 모델 주장 수준별 검증 결과</title>
      <desc id="claim-desc">공통 검증 프레임과 시장별 특화 전략은 세 시장에서 확인됐지만, 동일 사양과 leave-one-market-out 범용성은 세 시장 모두에서 확인되지 않았다.</desc>
      <text x="20" y="27" class="svg-title">같은 검증을 썼다는 것과 같은 모델이 통한다는 것은 다른 주장이다</text>
      {''.join(marks)}
      <text x="20" y="326" class="svg-note">결론: 범용 모델 주장은 미입증 · 현재 성과는 시장별 특화 전략의 역사적 결과로만 해석</text>
    </svg>"""


def build(
    markets,
    baseline_dir: Path,
    research_dir: Path,
    fairness_dir: Path,
    output: Path,
):
    baseline_dir = baseline_dir.resolve()
    research_dir = research_dir.resolve()
    fairness_dir = fairness_dir.resolve()
    same_spec = pd.read_csv(fairness_dir / "same_spec_summary.csv")
    lomo = pd.read_csv(fairness_dir / "leave_one_market_out_summary.csv")
    protocol = json.loads((fairness_dir / "protocol.json").read_text(encoding="utf-8"))
    fairness_conclusion = json.loads(
        (fairness_dir / "fairness_conclusion.json").read_text(encoding="utf-8")
    )
    baseline_latest = json.loads(
        (baseline_dir / "latest_ews.json").read_text(encoding="utf-8")
    )
    baseline_gates = one_row(baseline_dir / "deployment_gates.csv")
    universal_supported = bool(fairness_conclusion["universal_model_claim_supported"])
    same_count = int(same_spec["fully_validated_historical"].sum())
    lomo_count = int(lomo["fully_validated_historical"].sum())
    specialized_count = sum(m.validated for m in markets)
    model_params = protocol["model_params"]
    commands = "\n".join(
        f"python run_pipeline.py --market {m.key} --run-id <새로운_{m.key}_run_id>"
        for m in markets
    )
    commands += (
        "\n\npython run_universal_mlp_fairness.py "
        "--kospi-run runs\\<kospi_run_id> --sp500-run runs\\<sp500_run_id> "
        "--nasdaq-run runs\\<nasdaq_run_id> "
        "--output-dir runs\\<fairness_run_id>"
    )
    report_command = (
        "python generate_ews_mlp_economic_report.py "
        "--kospi-run runs\\<kospi_run_id> --sp500-run runs\\<sp500_run_id> "
        "--nasdaq-run runs\\<nasdaq_run_id> --baseline-run runs\\<baseline_run_id> "
        "--research-dir runs\\<research_run_id> --fairness-dir runs\\<fairness_run_id> "
        "--output reports\\<report_name>.html"
    )
    sources = "<br>".join(esc(m.run_dir) for m in markets)
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    artifact_paths = [
        ("기준 실행 manifest", baseline_dir / "experiment_manifest.json"),
        ("기준 Logistic/SVM/MLP 성과", baseline_dir / "performance_comparison.csv"),
        ("시장별 동일 사양 요약", fairness_dir / "same_spec_summary.csv"),
        ("Leave-one-market-out 요약", fairness_dir / "leave_one_market_out_summary.csv"),
        ("공정 비교 프로토콜", fairness_dir / "protocol.json"),
        ("범용성 최종 판정", fairness_dir / "fairness_conclusion.json"),
    ]
    artifact_rows = "".join(
        f'<tr><td>{esc(label)}</td><td><a href="{esc(path.resolve().as_uri())}"><code>{esc(path.name)}</code></a></td><td>{path.stat().st_size:,} bytes</td></tr>'
        for label, path in artifact_paths
    )
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EWS MLP 상세 기술보고서 · 경제적 의미 · 검증 · 범용성</title>
<style>
:root{{--bg:#08111f;--panel:#101d30;--panel2:#0b1728;--line:#24344c;--text:#e8f1ff;--muted:#9db0c8;--blue:#43b9ff;--teal:#3ed8b0;--amber:#ffbd59;--red:#ff7186;--purple:#9f8cff}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:radial-gradient(circle at 10% 0,#102945 0,transparent 32%),var(--bg);color:var(--text);font:15px/1.72 Inter,"Noto Sans KR",system-ui,sans-serif}}main{{width:min(1220px,calc(100% - 32px));margin:auto;padding:48px 0 72px}}h1,h2,h3,p{{margin-top:0}}h1{{font-size:clamp(34px,5vw,62px);line-height:1.08;letter-spacing:-.045em;max-width:980px}}h2{{font-size:28px;letter-spacing:-.025em;margin-bottom:10px}}h3{{font-size:18px;margin-bottom:8px}}p,li{{color:var(--muted)}}p:last-child{{margin-bottom:0}}a{{color:#89dcff;text-decoration:none}}a:hover{{text-decoration:underline}}.hero{{display:grid;grid-template-columns:1fr auto;gap:28px;align-items:end;margin-bottom:28px}}.lede{{max-width:920px;font-size:17px}}.eyebrow,.kicker{{color:#79d3ff;text-transform:uppercase;letter-spacing:.13em;font-size:11px;font-weight:700;margin-bottom:8px}}.stamp{{font-size:12px;color:var(--muted);text-align:right}}.summary{{background:linear-gradient(120deg,rgba(255,189,89,.14),rgba(255,113,134,.08));border:1px solid rgba(255,189,89,.45);padding:20px 24px;border-radius:16px;margin-bottom:18px;color:#ffe8bd}}.summary strong{{font-size:18px}}.toc{{display:flex;flex-wrap:wrap;gap:8px;background:#091625;border:1px solid var(--line);padding:14px;border-radius:14px;margin-bottom:22px}}.toc a{{background:#102139;border:1px solid #29405d;border-radius:999px;padding:5px 10px;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.market-card,section{{background:linear-gradient(150deg,rgba(16,29,48,.98),rgba(10,22,38,.98));border:1px solid var(--line);border-radius:18px;padding:24px}}section{{margin-top:18px}}.card-top{{display:flex;justify-content:space-between;gap:10px;align-items:start}}.card-top h2{{margin:0 0 2px}}.card-top p{{margin:0;font-size:12px}}.score-grid,.metric-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin:18px 0}}.metric-grid{{grid-template-columns:repeat(4,1fr)}}.score-grid div,.metric-grid div{{background:var(--panel2);padding:12px;border-radius:11px;border:1px solid rgba(44,64,90,.55)}}.score-grid span,.metric-grid span{{display:block;color:var(--muted);font-size:11px}}.score-grid strong,.metric-grid strong{{font-size:22px}}.positive{{color:#64e5ba}}.negative{{color:#ff9cac}}dl{{margin:0}}dl div{{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line);padding:8px 0}}dt{{color:var(--muted)}}dd{{margin:0;text-align:right}}.badge{{display:inline-block;border-radius:999px;padding:3px 9px;font-size:11px;font-weight:700;white-space:nowrap}}.ok{{background:#0d4a3d;color:#8af1ce}}.bad{{background:#52202b;color:#ffadbb}}.grid2,.grid3{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}.grid3{{grid-template-columns:repeat(3,1fr)}}.mini-card{{background:#0a1728;border:1px solid var(--line);border-radius:14px;padding:16px}}.callout,.warning{{border-left:4px solid var(--amber);background:#211b12;padding:16px 18px;border-radius:10px;color:#ffe1a6}}.warning{{border-left-color:var(--red);background:#25151c;color:#ffc6cf}}.note{{font-size:12px;color:var(--muted)}}.table-caption{{color:#d7e7fb;font-weight:700;margin:18px 0 8px}}.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;min-width:800px}}th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#0a1626;color:#a9bad0;font-size:11px;text-transform:uppercase;letter-spacing:.05em}}tbody tr:hover{{background:rgba(67,185,255,.035)}}code,pre{{font-family:Consolas,monospace}}code{{font-size:12px;color:#b7e7ff}}pre{{white-space:pre-wrap;background:#071321;border:1px solid var(--line);padding:16px;border-radius:12px;color:#b7e7ff;overflow:auto}}details{{margin-top:10px;background:#091625;border:1px solid var(--line);border-radius:12px}}summary{{cursor:pointer;padding:13px 15px;color:#dcebff;font-weight:700}}.details-body{{padding:0 14px 14px}}.formula{{background:#071321;border:1px solid var(--line);border-radius:10px;padding:12px 14px;color:#c8eaff;font-family:Consolas,monospace}}svg{{width:100%;height:auto}}.svg-panel,.economic-node{{fill:#0c1a2c;stroke:#2c405a;stroke-width:1}}.economic-node.output{{fill:#10322f;stroke:#2f806e}}.economic-node.neural{{fill:#241d3f;stroke:#6653a6}}.svg-title{{fill:#e7f2ff;font-size:15px;font-weight:700}}.svg-head{{fill:#e7f2ff;font-size:12px;font-weight:700}}.svg-note,.svg-label{{fill:#9db0c8;font-size:10px}}.svg-value{{fill:#e7f2ff;font-size:12px;font-weight:700}}.line{{fill:none;stroke-width:2}}.line.mlp{{stroke:var(--blue);stroke-width:3}}.line.same{{stroke:#6b7e98}}.axis{{stroke:#34465e;stroke-width:1}}.bar-mlp{{fill:var(--blue)}}.bar-same{{fill:#647792}}.claim-pass{{fill:#0d4a3d;stroke:#2f806e}}.claim-fail{{fill:#52202b;stroke:#a44558}}.arrow{{stroke:#6e819a;stroke-width:1.5;fill:none}}.arrow-fill{{fill:#6e819a}}.lane-label{{fill:#79d3ff;font-size:12px;font-weight:700}}.step-circle{{fill:#173a52;stroke:#43b9ff}}.step-number{{fill:#dff5ff;font-size:12px;font-weight:700}}.footer{{margin-top:26px;color:#768aa5;font-size:11px;word-break:break-all}}@media(max-width:960px){{.cards,.grid3,.metric-grid{{grid-template-columns:1fr 1fr}}.grid2{{grid-template-columns:1fr}}.hero{{grid-template-columns:1fr}}.stamp{{text-align:left}}}}@media(max-width:600px){{main{{width:min(100% - 20px,1220px);padding-top:28px}}.market-card,section{{padding:16px}}.cards,.grid3,.metric-grid{{grid-template-columns:1fr}}h1{{font-size:36px}}}}
</style></head><body><main>
<header class="hero"><div><p class="eyebrow">EARLY WARNING SYSTEM · DETAILED TECHNICAL REPORT</p><h1>시장별 성과, 공정 비교, 운영 한계를 하나의 증거 체계로 읽는다</h1><p class="lede">이 보고서는 기준 KOSPI 실행, 시장별 특화 MLP, 완전 동일 사양 비교, leave-one-market-out 이전 검증을 서로 섞지 않고 설명한다. 좋은 Sharpe가 무엇을 입증하며 무엇을 입증하지 못하는지까지 수치와 코드 산출물로 추적한다.</p></div><div class="stamp">생성 {esc(generated)}<br>데이터 스냅샷 2026-08-14<br>공정 비교 cutoff 2020-03</div></header>
<div class="summary"><strong>{'범용 MLP 주장이 역사적으로 지지됐다.' if universal_supported else '핵심 결론: 범용 MLP 주장은 아직 입증되지 않았다.'}</strong><br>시장별 특화 모델은 역사적 게이트 {specialized_count}/3 통과, 동일 사양 개별 재학습은 {same_count}/3, leave-one-market-out은 {lomo_count}/3 통과했다. 특화 결과가 좋다는 사실과 하나의 모델이 세 시장으로 이전된다는 주장은 별개다. 현재 세 특화 모델 모두 <code>strict_operational_gate=False</code>이므로 연구·shadow 기록은 가능하지만 실자금 사용은 차단된다.</div>
<nav class="toc" aria-label="보고서 목차"><a href="#executive">판정 요약</a><a href="#baseline">기준 실행</a><a href="#protocol">공정 비교 설계</a><a href="#same-spec">동일 사양</a><a href="#lomo">시장 제외</a><a href="#specialized">특화 모델</a><a href="#korea">KOSPI 개선</a><a href="#economics">경제적 의미</a><a href="#gates">게이트</a><a href="#operations">운영 준비</a><a href="#reproduce">재현</a><a href="#limits">한계</a></nav>

<section id="executive"><p class="eyebrow">EXECUTIVE DECISION</p><h2>이 보고서에서 확정할 수 있는 것</h2><div class="grid3"><div class="mini-card"><h3>① 비교 버그는 수정됨</h3><p>기준 실행의 Logistic·SVM·MLP와 각 benchmark는 모두 2020-04~2026-05의 같은 74개월로 잘렸다. 모델별 prediction과 backtest도 분리됐다.</p></div><div class="mini-card"><h3>② 특화 전략은 역사적으로 통과</h3><p>KOSPI는 선형 백본+MLP 위험 거부권, 미국은 독립 MLP를 사용한다. 각각의 사전 게이트와 역사적 safety veto는 통과했다.</p></div><div class="mini-card"><h3>③ 범용성은 미입증</h3><p>동일 입력·목표·MLP·비중정책을 고정하면 0/3, 대상 시장 label을 제외한 이전 검증은 1/3만 전체 통과했다.</p></div></div>
<div class="metric-grid"><div><span>원자료 카탈로그</span><strong>70</strong></div><div><span>연구 적격 원천</span><strong>64</strong></div><div><span>전체 후보 Factor</span><strong>10,961</strong></div><div><span>공정 비교 공통 Factor</span><strong>4</strong></div></div>
<p class="warning"><strong>배포 판정:</strong> 이 문서의 통과는 역사적 연구 게이트 통과다. 사람의 경제적 검토, 발표시차 검토, 중복정보 검토가 끝나지 않아 <code>capital_use_allowed=False</code>다. “좋은 백테스트”와 “실전 배포 가능”을 같은 뜻으로 읽으면 안 된다.</p>{svg_claim_ladder(markets, same_spec, lomo)}</section>

<section id="baseline"><p class="eyebrow">BASELINE REPRODUCTION</p><h2><code>ews_reproduce_20260814_v2</code>에서 무엇이 확인됐는가</h2><p>기준 실행은 51개 FRED 계열과 시장 파생 계열을 월말로 통합해 1,291개월 패널을 만들었다. KOSPI 원자료는 밀린 yfinance 헤더를 감지해 <code>Price→close, Close→high, High→low, Low→open, Open→volume</code>으로 복구했다. 개발 1996-03~2014-03, 검증 2014-04~2020-03, 연구 holdout 2020-04 이후로 분리했으며 최초 rolling 예측에는 60개월을 요구했다.</p>
<p class="table-caption">후보군이 최종 모델 입력으로 좁혀지는 과정</p><div class="table-wrap"><table><thead><tr><th>단계</th><th>개수</th><th>산출물 필드</th></tr></thead><tbody>{factor_funnel_rows(baseline_dir)}</tbody></table></div>
<p class="note">원천별 최대 1개와 카테고리별 최대 1개 hard cap은 사용하지 않는다(<code>max_features_per_base=null</code>, <code>max_features_per_group=null</code>). 대신 미래정보 방지, 연구 적격성, 상관 중복 제거와 사전 선언 핵심 family 조건을 유지한다. 제한 해제는 같은 경제정보의 변형을 무제한 허용한다는 뜻이 아니다.</p>
<p class="table-caption">동일 74개월에서의 모델 예측력과 class balance</p><div class="table-wrap"><table><thead><tr><th>모델</th><th>N</th><th>AUC</th><th>Brier</th><th>Accuracy</th><th>Naive</th><th>Accuracy lift</th><th>Rank IC</th><th>Rolling IC +</th></tr></thead><tbody>{baseline_model_rows(baseline_dir)}</tbody></table></div>
<p class="callout">기준 실행에서 MLP AUC 0.795와 Rank IC 0.378은 강했지만 Accuracy 64.4%는 다수 class 65.8%보다 높지 않았다. 따라서 Accuracy가 아니라 확률 순위(AUC), 미래수익 정렬(Rank IC), 동기간 포트폴리오 성과와 fold 안정성을 함께 본다. 이 결과 하나만으로 MLP를 배포하지 않았다.</p>
<p class="table-caption">Dynamic과 동일 평균 주식노출 benchmark</p><div class="table-wrap"><table><thead><tr><th>모델</th><th>전략</th><th>공통기간</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Calmar</th></tr></thead><tbody>{baseline_performance_rows(baseline_dir)}</tbody></table></div>
<p class="note">표의 모든 행은 74개월이다. Same Exposure는 Dynamic의 평균 주식비중만큼 주식을 계속 보유한 전략이므로, 강세장에서 평균 주식노출이 높아 생기는 착시를 분리한다. 기준 실행의 최신 출력은 {esc(baseline_latest['date'])}, Logistic EWS {float(baseline_latest['ews']):.1f}, 목표 주식비중 {pct(baseline_latest['target_stock_weight'],0)}였지만 <code>deployment_eligible={str(baseline_latest['deployment_eligible']).lower()}</code>였다.</p></section>

<section id="protocol"><p class="eyebrow">FAIRNESS PROTOCOL</p><h2>“동일 모델”을 어떻게 정의했는가</h2><p>시장별 특화 결과는 목표와 구조가 달라 범용성 근거가 될 수 없다. 그래서 아래 항목을 한 글자도 시장별로 바꾸지 않은 별도 공정 비교 track을 만들었다.</p>{svg_validation_timeline(protocol)}
<div class="grid2"><div><h3>고정된 학습 사양</h3><ul><li>목표: <code>{esc(protocol['target_mode'])}</code>, 향후 {int(protocol['forecast_horizon_months'])}개월 현금초과 수익</li><li>MLP: {model_params['hidden_layer_sizes'][0]}×{model_params['hidden_layer_sizes'][1]} hidden, {esc(model_params['activation'])}, {esc(model_params['solver'])}, alpha {model_params['alpha']}</li><li>최소 학습: 시장별 {int(protocol['minimum_training_months_per_source_market'])}개월</li><li>재학습: 매월, 목표 horizon만큼 purge</li><li>비중: <code>{esc(protocol['allocation_policy'])}</code>, 10·25bp 비용</li></ul></div><div><h3>누수 방지 규칙</h3><ul><li>세 input run의 공통 Factor matrix가 값과 index까지 완전히 같아야 실행</li><li>공통 outer fold 달력을 세 시장에 동일 적용</li><li>2020-04 이후는 모델·정책 선택에 사용하지 않음</li><li>LOMO에서는 대상 시장 label을 학습에서 완전 제외</li><li>historical holdout 통과는 승격 근거가 아니며 실패만 veto</li></ul></div></div>
<p class="table-caption">범용 공정 비교에 사용한 정확히 같은 네 Factor</p><div class="table-wrap"><table><thead><tr><th>#</th><th>경제적 의미</th><th>정확한 컬럼명</th></tr></thead><tbody>{protocol_feature_rows(protocol)}</tbody></table></div>
<p class="table-caption">공통 outer fold</p><div class="table-wrap"><table><thead><tr><th>Fold</th><th>시작</th><th>종료</th></tr></thead><tbody>{protocol_fold_rows(protocol)}</tbody></table></div></section>

<section id="same-spec"><p class="eyebrow">APPLES TO APPLES</p><h2>동일 사양을 각 시장에서 따로 재학습한 결과: 0/3</h2><p>같은 레시피를 쓰되 각 시장의 과거 label로 따로 학습했다. 이는 “한 코드와 한 hyperparameter 사양이 시장별 재학습을 허용할 때도 안정적인가”를 묻는다.</p>
<p class="table-caption">2020-03 이전: 선택과 게이트에 사용한 근거</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>AUC</th><th>Rank IC</th><th>방향 fold</th><th>중앙 Sharpe Δ</th><th>양의 fold</th><th>25bp 능동수익</th><th>25bp MDD Δ</th><th>Signal / Port.</th><th>사전 공동</th></tr></thead><tbody>{fairness_pre2020_rows(same_spec)}</tbody></table></div>
<p class="table-caption">2020-04 이후: 선택에 쓰지 않은 역사적 safety veto</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>AUC</th><th>Rank IC</th><th>Sharpe 전략/동일노출</th><th>Sharpe Δ</th><th>CAGR 전략/동일노출</th><th>CAGR Δ</th><th>MDD 전략/동일노출</th><th>Safety</th></tr></thead><tbody>{fairness_holdout_rows(same_spec)}</tbody></table></div>
<div class="grid3"><div class="mini-card"><h3>KOSPI200</h3><p>사전 집계 AUC 0.533·IC 0.023이지만 공동 방향 fold는 50%로 기준 66.7%에 못 미쳤다. 포트폴리오도 중앙 Sharpe Δ -0.002, 25bp 능동수익 -0.23%, MDD Δ -4.06%p로 실패했다. holdout AUC 0.444·IC -0.072와 CAGR 열위로 safety도 실패했다.</p></div><div class="mini-card"><h3>S&amp;P 500</h3><p>사전 AUC 0.611·IC 0.044와 portfolio gate는 좋았지만, 두 지표가 동시에 양수인 fold가 25%뿐이었다. 2020+ Sharpe 1.486 대 1.068은 강했어도 사전 signal 실패를 뒤집지 않는다.</p></div><div class="mini-card"><h3>NASDAQ-100</h3><p>사전 AUC 0.560이지만 IC -0.030으로 신호 정렬이 반대였다. portfolio와 2020+ safety는 통과했지만 signal gate가 실패해 전체 미통과다.</p></div></div>
<p class="callout"><strong>중요:</strong> “방향 fold”는 계수 부호 안정성이 아니다. 평가 가능한 outer fold 중 AUC&gt;0.5와 Rank IC&gt;0을 동시에 만족한 비율이다. 전체 기준은 2/3 이상이다.</p>{fold_evidence(fairness_dir, 'same_spec', same_spec)}</section>

<section id="lomo"><p class="eyebrow">CROSS-MARKET TRANSFER</p><h2>Leave-one-market-out 결과: 1/3</h2><p>각 대상 시장의 label을 한 건도 쓰지 않고 나머지 두 시장의 과거 label을 합쳐 하나의 MLP를 학습했다. 대상 시장의 공통 Factor 값만 넣어 parameter 변경 없이 예측했으므로, 시장 간 이전 가능성을 더 직접적으로 검증한다.</p>
<p class="table-caption">2020-03 이전 이전성 검증</p><div class="table-wrap"><table><thead><tr><th>대상 시장</th><th>AUC</th><th>Rank IC</th><th>방향 fold</th><th>중앙 Sharpe Δ</th><th>양의 fold</th><th>25bp 능동수익</th><th>25bp MDD Δ</th><th>Signal / Port.</th><th>사전 공동</th></tr></thead><tbody>{fairness_pre2020_rows(lomo)}</tbody></table></div>
<p class="table-caption">2020-04 이후 safety veto</p><div class="table-wrap"><table><thead><tr><th>대상 시장</th><th>AUC</th><th>Rank IC</th><th>Sharpe 전략/동일노출</th><th>Sharpe Δ</th><th>CAGR 전략/동일노출</th><th>CAGR Δ</th><th>MDD 전략/동일노출</th><th>Safety</th></tr></thead><tbody>{fairness_holdout_rows(lomo)}</tbody></table></div>
<div class="grid3"><div class="mini-card"><h3>KOSPI200 ← 미국</h3><p>S&amp;P 500·NASDAQ-100으로 학습했다. 사전 Rank IC -0.078과 방향 fold 25% 때문에 signal이 실패했다. portfolio와 holdout은 통과했으나 이전성을 확정할 수 없다.</p></div><div class="mini-card"><h3>S&amp;P 500 ← KOSPI·Nasdaq</h3><p>사전 AUC 0.572, IC 0.095, 방향 fold 75%; portfolio와 holdout도 모두 통과했다. 선언된 역사 게이트 아래 이전 가능성이 확인된 유일한 시장이다.</p></div><div class="mini-card"><h3>NASDAQ-100 ← KOSPI·S&amp;P</h3><p>방향 fold 50%로 signal 실패. 중앙 fold Sharpe Δ 0.095가 최소 0.10에 약간 못 미쳐 portfolio도 실패했다. holdout 통과만으로 승격하지 않았다.</p></div></div>
<p class="warning">S&amp;P 500 한 시장의 성공은 “세 시장 범용 모델”을 입증하지 않는다. 범용성 주장은 세 시장이 모두 동일 protocol을 통과해야 한다. 최종 JSON 판정은 <code>{esc(fairness_conclusion['decision'])}</code>이다.</p>{fold_evidence(fairness_dir, 'leave_one_market_out', lomo)}</section>

<section id="specialized"><p class="eyebrow">MARKET-SPECIFIC RESULTS</p><h2>시장별 특화 모델은 왜 별도 주장인가</h2><p>아래 세 모델은 각 시장에 맞춰 목표, MLP 역할, 비중 정책을 달리 고정했다. 역사적 성과는 유효한 연구 결과지만 서로 같은 모델이 아니다. KOSPI는 현금초과 목표와 위험 거부권, S&amp;P 500은 절대 양(+) 수익 목표, NASDAQ-100은 향후 -5% 낙폭 회피 목표를 쓴다.</p><div class="cards">{market_cards(markets)}</div>
<p class="table-caption">사전 신호와 2020+ 역사적 확인을 한 표에서 보기</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>구조</th><th>예측 질문</th><th>사전 AUC</th><th>사전 IC</th><th>방향 fold</th><th>Holdout Sharpe 전략/동일</th><th>Holdout CAGR 전략/동일</th><th>Holdout MDD 전략/동일</th><th>역사 판정</th></tr></thead><tbody>{specialized_evidence_rows(markets)}</tbody></table></div></section>
<section>{svg_sharpe(markets)}<p class="note">이 차트는 시장별 특화 모델의 역사적 holdout이다. 모델 구조와 목표가 달라 범용성 비교표가 아니다.</p></section><section>{svg_curves(markets)}</section>

<section id="korea"><p class="eyebrow">WHY KOREA WAS DIFFERENT</p><h2>KOSPI에서 독립 MLP를 그대로 쓰지 않은 이유</h2><p>기준 실행의 KOSPI MLP는 2020+ AUC와 Rank IC가 강하고 Dynamic Sharpe도 높았지만, 이 구간은 이미 진단에 노출된 역사적 holdout이었다. 사전 2020 후보 검증에서 독립 MLP의 확률 진폭과 비중 임계값 교차가 시기별로 흔들리며 거래비용 후 능동수익과 fold Sharpe 기준을 안정적으로 넘지 못했다. 따라서 후반 성과를 보고 독립 MLP를 채택하지 않고, 사전에 평가한 방어적 역할로 제한했다.</p>
<div class="table-wrap"><table><thead><tr><th>사전 2020 후보</th><th>AUC</th><th>Rank IC</th><th>방향 fold</th><th>정책</th><th>중앙 Sharpe Δ</th><th>25bp 능동수익</th><th>공동 gate</th></tr></thead><tbody>{diagnosis_rows(research_dir)}</tbody></table></div>
<p class="callout"><strong>최종 위험 거부권:</strong> Logistic 백본이 65% 이상 risk-on을 말해도 MLP가 50% 미만이면 50% 중립으로 낮춘다. MLP는 공격 신호나 단독 비중을 만들 수 없고 위험만 거부한다. 사전 구간 개입률은 약 7.3%였으며, 이 구조가 KOSPI 특화 MLP의 의미다.</p>{svg_architecture()}</section>

<section id="economics"><p class="eyebrow">ECONOMIC MEANING</p><h2>Factor가 자산배분으로 이어지는 경제적 경로</h2>{svg_economic_chain()}<div class="grid2"><div class="mini-card"><h3>현금흐름</h3><p>기업이익과 경기 변수는 미래 배당·이익의 기반을 근사한다. 기업가치 추세 이격과 이익 변동성은 현금흐름 기대의 수준과 불확실성을 함께 표현한다.</p></div><div class="mini-card"><h3>할인율과 금융주기</h3><p>10년-2년 또는 10년-3개월 금리차는 통화정책과 경기 기대가 결합된 할인율 상태다. 단순 금리 수준보다 곡선의 변화가 전환점을 포착하는 데 쓰인다.</p></div><div class="mini-card"><h3>글로벌 유동성과 위험선호</h3><p>AUD/USD는 경기민감 통화와 달러 유동성의 상대 움직임을 대리한다. KOSPI 내부 거래대금·상관·왜도는 거시지표보다 빠르게 쏠림과 급락 취약성을 드러낸다.</p></div><div class="mini-card"><h3>실행 가능한 비교</h3><p>신호는 지수로 만들되 수익은 KODEX 200·SPY·QQQ 조정가격으로 측정한다. 현금과 회전율 비용을 포함하고 같은 평균 주식노출과 비교해 timing alpha를 분리한다.</p></div></div>
<p class="table-caption">시장별 특화 모델의 최종 입력</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>경제적 해석</th><th>코드 Factor</th></tr></thead><tbody>{feature_rows(markets)}</tbody></table></div></section>

<section id="code"><p class="eyebrow">CODE AND DATA PATH</p><h2>코드가 미래정보와 비교 착시를 막는 순서</h2>{svg_code_flow()}<ol><li><code>src/data.py</code>가 51개 FRED 계열, 시장지수, 투자 가능 ETF를 월말에 맞추고 point-in-time·발표시차 메타데이터를 보존한다.</li><li><code>src/features.py</code>는 이동평균·변화율·변동성·시장폭을 해당 시점까지의 값만으로 만든다. 양방향 HP filter는 제외됐다.</li><li><code>src/modeling.py</code>는 향후 3개월 목표를 만들고, 각 예측일에서 최소 3개월 purge 후의 데이터까지만 학습한다. 최소 84개월 뒤 매월 재학습한다.</li><li><code>src/validation.py</code>는 고정 outer fold에서 AUC·Rank IC·coverage·방향 일관성과 비용별 portfolio gate를 분리 평가한다.</li><li><code>src/position_sizing.py</code>는 고정 bin 또는 그 시점까지 누적된 expanding percentile로만 20~80% 비중을 만든다.</li><li><code>src/backtest.py</code>는 모든 전략과 benchmark를 같은 날짜로 자르고 ETF+현금+회전율 비용을 반영한다.</li></ol>
<p class="note">공정 비교 스크립트는 세 input run의 공통 네 Factor matrix가 exact-equal이고 code-files manifest hash가 하나일 때만 실행한다. LOMO 예측은 대상 시장의 feature는 보지만 label은 보지 않는다.</p></section>

<section id="gates"><p class="eyebrow">VALIDATION GATES</p><h2>각 통과 표시의 정확한 의미</h2><div class="grid3"><div class="mini-card"><h3>Signal gate</h3><div class="formula">AUC &gt; 0.5<br>Rank IC &gt; 0<br>eligible fold coverage = 100%<br>공동 방향 fold ≥ 2/3</div><p>분류 순위와 미래수익 정렬이 전체와 여러 시기에 동시에 유지돼야 한다.</p></div><div class="mini-card"><h3>Portfolio gate</h3><div class="formula">중앙 fold Sharpe Δ ≥ 0.10<br>양의 능동수익 fold ≥ 2/3<br>10·25bp 능동수익 &gt; 0<br>MDD Δ ≥ -3%p</div><p>확률이 실제 비중으로 바뀐 뒤에도 비용과 시장노출을 이겨야 한다.</p></div><div class="mini-card"><h3>Historical safety veto</h3><div class="formula">AUC &gt; 0.5 · Rank IC &gt; 0<br>Sharpe Δ ≥ 0 · CAGR Δ ≥ 0<br>MDD Δ ≥ -3%p</div><p>2020+ 통과는 승격하지 못한다. 하나라도 실패하면 전술 운용을 중단시키는 fail-closed 안전장치다.</p></div></div>
<p class="table-caption">시장별 특화 모델의 게이트 상태</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>Signal</th><th>Portfolio</th><th>2020+ safety</th><th>실자금 운영</th><th>상위 실패조건</th></tr></thead><tbody>{gate_rows(markets)}</tbody></table></div></section>

<section id="operations"><p class="eyebrow">OPERATIONAL READINESS</p><h2>성과가 좋아도 아직 실자금에 못 쓰는 이유</h2><p>통계적·포트폴리오 게이트와 운영 게이트는 독립적이다. 현재 데이터 vintage, release timing, 투자 가능 수익원은 자동 검사를 통과했지만 세 가지 사람 검토가 남아 있다.</p><div class="table-wrap"><table><thead><tr><th>시장</th><th>PIT vintage</th><th>Release timing</th><th>투자가능 수익원</th><th>경제 검토</th><th>발표시차 검토</th><th>중복정보 검토</th><th>Capital</th></tr></thead><tbody>{operational_rows(markets)}</tbody></table></div>
<div class="grid3"><div class="mini-card"><h3>경제적 검토</h3><p>선택 Factor의 방향과 상호작용이 경제적으로 설명되는지, 위기 한두 번에만 의존하지 않는지 사람이 승인해야 한다.</p></div><div class="mini-card"><h3>발표시차 검토</h3><p>원자료 release calendar와 적용 lag가 실제 운용일에 맞는지 표본별로 확인해야 한다. 자동 메타데이터 통과만으로 절차가 끝나지 않는다.</p></div><div class="mini-card"><h3>중복정보 검토</h3><p>상관 제거 후에도 경제적으로 같은 정보가 여러 변형으로 반복되는지 최종 입력과 후보 생성 규칙을 사람이 검토해야 한다.</p></div></div>
<p class="warning"><code>research_waiver</code>는 문제를 없애는 면허가 아니라, 위험을 명시한 채 shadow 연구를 계속하도록 허용하는 상태다. 기준 실행의 <code>operational_gate={str(as_bool(baseline_gates['operational_gate'])).lower()}</code>와 <code>waiver_gate={str(as_bool(baseline_gates['waiver_gate'])).lower()}</code>가 참이어도 <code>strict_operational_gate=false</code>, <code>deployment_eligible=false</code>다.</p></section>

<section id="reproduce"><p class="eyebrow">REPRODUCIBILITY</p><h2>동일 결과를 다시 만드는 순서</h2><ol><li>시장별 pipeline을 각각 새로운 run-id로 실행한다.</li><li>세 run을 입력으로 공정 비교를 실행한다. 입력 matrix나 code manifest가 다르면 즉시 실패한다.</li><li>새 baseline·research·fairness 디렉터리를 보고서 생성기에 전달한다.</li><li>CSV 요약과 JSON 결론을 먼저 확인한 뒤 HTML을 읽는다.</li></ol><pre>{esc(commands)}\n\n{esc(report_command)}</pre><p class="note">run 디렉터리는 불변 산출물로 취급한다. 같은 run-id를 덮어쓰지 말고 날짜·버전을 올린다. 재현 시 원자료가 개정되면 결과가 달라질 수 있으므로 manifest와 protocol hash를 함께 보관한다.</p>
<p class="table-caption">이 보고서가 직접 읽은 핵심 산출물</p><div class="table-wrap"><table><thead><tr><th>근거</th><th>파일</th><th>크기</th></tr></thead><tbody>{artifact_rows}</tbody></table></div></section>

<section id="limits"><p class="eyebrow">LIMITS AND NEXT EVIDENCE</p><h2>이 결과가 말하지 않는 것과 다음 검증</h2><ul><li><strong>범용 모델:</strong> 현재 미입증이다. 동일 사양 0/3, LOMO 1/3이므로 S&amp;P 한 시장의 성공을 전체 성공으로 확장할 수 없다.</li><li><strong>시장별 특화 모델:</strong> 역사적 연구 게이트 통과다. 서로 다른 목표·구조·정책을 사용하므로 같은 모델의 재현 증거가 아니다.</li><li><strong>2020+ 구간:</strong> 이미 여러 차례 보고·진단한 historical holdout이다. 더 이상 완전히 untouched한 test로 부를 수 없고, 통과는 승격에 사용하지 않는다.</li><li><strong>실전 마찰:</strong> ETF 추적오차, 세금, 체결 지연, 거래비용 급등과 regime change를 완전히 제거하지 못한다.</li><li><strong>남은 증거:</strong> 2026-08 이후 frozen shadow ledger를 사전 선언된 규칙으로 누적하고, 경제·발표시차·중복정보 검토를 승인한 뒤에만 strict operational gate를 재평가해야 한다.</li></ul>
<p class="callout"><strong>현재 권고:</strong> 시장별 특화 모델은 shadow 관찰을 계속하고, 범용 MLP는 연구 가설로 유지한다. 2020+ 결과를 보고 feature·target·정책을 다시 고른 뒤 같은 구간으로 재검증하는 방식은 금지한다.</p></section>
<div class="footer">기준 실행: {esc(baseline_dir)}<br>시장별 특화 실행 디렉터리<br>{sources}<br><br>KOSPI 후보 연구: {esc(research_dir)}<br>동일모델 공정 비교: {esc(fairness_dir)}<br><br>본 문서는 연구용이며 투자 권유가 아니다.</div>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kospi-run", required=True)
    parser.add_argument("--sp500-run", required=True)
    parser.add_argument("--nasdaq-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--research-dir", required=True)
    parser.add_argument("--fairness-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    markets = [
        Market.load(args.kospi_run),
        Market.load(args.sp500_run),
        Market.load(args.nasdaq_run),
    ]
    output = build(
        markets,
        Path(args.baseline_run),
        Path(args.research_dir),
        Path(args.fairness_dir),
        Path(args.output).resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
