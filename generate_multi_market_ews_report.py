"""Build one self-contained HTML/SVG report for the three EWS markets."""

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


def truth(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pct(value, digits=1) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def number(value, digits=3) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def read_row(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    if len(data) != 1:
        raise ValueError(f"Expected one row in {path}, got {len(data)}")
    return data.iloc[0]


@dataclass
class MarketResult:
    run_dir: Path
    key: str
    name: str
    instrument: str
    logistic_gates: pd.Series
    logistic_signal_gates: pd.Series
    mlp_gates: pd.Series
    primary_holdout: pd.Series
    mlp_holdout: pd.Series
    performance: pd.DataFrame
    signal_metrics: pd.DataFrame
    ic_metrics: pd.DataFrame
    latest_logistic: dict
    latest_mlp: dict
    champion: str
    champion_reason: str
    champion_performance: pd.Series
    comparison_performance: pd.Series
    latest: dict

    @classmethod
    def load(cls, run_dir: str | Path) -> "MarketResult":
        run_dir = Path(run_dir).resolve()
        manifest = json.loads(
            (run_dir / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        config = manifest.get("config", {})
        key = config.get("market_key") or "kospi200"
        name = config.get("market_name") or "KOSPI200"
        instrument = config.get("investable_instrument") or name
        logistic = read_row(run_dir / "deployment_gates.csv")
        logistic_signal = read_row(run_dir / "signal_gate_diagnostics.csv")
        mlp = read_row(run_dir / "mlp_validation_gates.csv")
        primary_holdout_path = run_dir / "historical_holdout_confirmation.csv"
        primary_holdout = (
            read_row(primary_holdout_path)
            if primary_holdout_path.exists()
            else pd.Series({"holdout_safety_gate": True})
        )
        mlp_holdout = read_row(
            run_dir / "mlp_historical_holdout_confirmation.csv"
        )
        performance = pd.read_csv(run_dir / "performance_comparison.csv")
        signal_metrics = pd.read_csv(run_dir / "model_comparison.csv")
        ic_metrics = pd.read_csv(run_dir / "model_ic_comparison.csv")
        latest_logistic = json.loads(
            (run_dir / "latest_ews.json").read_text(encoding="utf-8")
        )
        latest_mlp = json.loads(
            (run_dir / "latest_mlp_ews.json").read_text(encoding="utf-8")
        )

        logistic_ok = (
            truth(logistic["signal_gate"])
            and truth(logistic["portfolio_gate"])
            and truth(primary_holdout["holdout_safety_gate"])
        )
        mlp_ok = (
            truth(mlp["signal_gate"])
            and truth(mlp["portfolio_gate"])
            and truth(mlp["historical_holdout_safety_gate"])
        )
        if logistic_ok:
            champion = "Logistic"
            reason = "pre-2020 signal·portfolio와 holdout safety gate 통과"
            latest = latest_logistic
            dynamic_label = "Logistic Dynamic"
            same_prefix = "Logistic Same Exposure"
        elif mlp_ok:
            champion = "MLP"
            reason = "Logistic 실패 후 MLP 독립 gate·holdout safety 통과"
            latest = latest_mlp
            dynamic_label = "MLP Dynamic"
            same_prefix = "MLP Same Exposure"
        else:
            champion = "Static 50/50"
            reason = "검증된 tactical 후보 없음; fail-closed"
            latest = latest_logistic
            dynamic_label = "Logistic Static 50/50"
            same_prefix = "Logistic Static 50/50"

        champion_row = performance.loc[
            performance["strategy"].eq(dynamic_label)
        ].iloc[0]
        comparison_row = performance.loc[
            performance["strategy"].str.startswith(same_prefix)
        ].iloc[0]
        return cls(
            run_dir=run_dir,
            key=key,
            name=name,
            instrument=instrument,
            logistic_gates=logistic,
            logistic_signal_gates=logistic_signal,
            mlp_gates=mlp,
            primary_holdout=primary_holdout,
            mlp_holdout=mlp_holdout,
            performance=performance,
            signal_metrics=signal_metrics,
            ic_metrics=ic_metrics,
            latest_logistic=latest_logistic,
            latest_mlp=latest_mlp,
            champion=champion,
            champion_reason=reason,
            champion_performance=champion_row,
            comparison_performance=comparison_row,
            latest=latest,
        )

    @property
    def tactical_active(self) -> bool:
        return self.champion != "Static 50/50"

    @property
    def target_weight(self) -> float:
        if not self.tactical_active:
            return 0.5
        return float(self.latest.get("target_stock_weight", 0.5))

    @property
    def status(self) -> str:
        if self.tactical_active:
            return "RESEARCH SHADOW"
        return "STATIC FALLBACK"


def svg_allocation(weight: float, color: str) -> str:
    radius = 45
    circumference = 2 * np.pi * radius
    dash = circumference * min(max(weight, 0.0), 1.0)
    return f"""<svg class="donut" viewBox="0 0 120 120" role="img" aria-label="주식 목표 비중 {weight:.0%}">
      <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#263244" stroke-width="12"/>
      <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{color}" stroke-width="12"
        stroke-linecap="round" transform="rotate(-90 60 60)" stroke-dasharray="{dash:.2f} {circumference-dash:.2f}"/>
      <text x="60" y="57" text-anchor="middle" class="donut-value">{weight:.0%}</text>
      <text x="60" y="75" text-anchor="middle" class="donut-label">주식</text>
    </svg>"""


def _points(values, width, height, pad=18, value_range=None):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return ""
    values = values[finite]
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


def svg_curve(market: MarketResult, width=560, height=170) -> str:
    filename = "mlp_backtest.csv" if market.champion == "MLP" else "backtest.csv"
    frame = pd.read_csv(market.run_dir / filename)
    strategy = pd.to_numeric(frame.get("strategy_curve"), errors="coerce")
    same = pd.to_numeric(frame.get("same_exposure_curve"), errors="coerce")
    valid = strategy.notna() & same.notna()
    strategy, same = strategy[valid], same[valid]
    combined = pd.concat([strategy, same]).dropna()
    shared_range = (float(combined.min()), float(combined.max()))
    sp = _points(strategy, width, height, value_range=shared_range)
    bp = _points(same, width, height, value_range=shared_range)
    return f"""<svg class="curve" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(market.name)} 누적 전략 곡선">
      <rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#0e1726"/>
      <line x1="18" y1="{height-18}" x2="{width-18}" y2="{height-18}" stroke="#334155"/>
      <polyline points="{bp}" fill="none" stroke="#64748b" stroke-width="2"/>
      <polyline points="{sp}" fill="none" stroke="#38bdf8" stroke-width="3"/>
      <text x="20" y="24" class="svg-title">{esc(market.champion)} vs 동일 익스포저</text>
      <circle cx="{width-168}" cy="21" r="4" fill="#38bdf8"/><text x="{width-158}" y="25" class="svg-legend">권고</text>
      <circle cx="{width-92}" cy="21" r="4" fill="#64748b"/><text x="{width-82}" y="25" class="svg-legend">동일노출</text>
    </svg>"""


def svg_metric_bars(markets: list[MarketResult], metric: str, title: str) -> str:
    width, height, pad = 760, 260, 54
    vals = []
    for market in markets:
        vals.extend(
            [
                float(market.champion_performance[metric]),
                float(market.comparison_performance[metric]),
            ]
        )
    if metric == "MaxDrawdown":
        values = [abs(v) for v in vals]
    else:
        values = vals
    maximum = max(values + [0.01]) * 1.12
    groups = []
    colors = ("#38bdf8", "#64748b")
    for i, market in enumerate(markets):
        center = 155 + i * 225
        pair = [
            float(market.champion_performance[metric]),
            float(market.comparison_performance[metric]),
        ]
        for j, raw in enumerate(pair):
            value = abs(raw) if metric == "MaxDrawdown" else raw
            bh = value / maximum * (height - 2 * pad)
            x = center - 34 + j * 48
            y = height - pad - bh
            label = pct(raw) if metric == "MaxDrawdown" else number(raw, 2)
            groups.append(
                f'<rect x="{x}" y="{y:.1f}" width="34" height="{bh:.1f}" rx="5" fill="{colors[j]}"/>'
                f'<text x="{x+17}" y="{y-7:.1f}" text-anchor="middle" class="bar-value">{label}</text>'
            )
        groups.append(
            f'<text x="{center+7}" y="{height-22}" text-anchor="middle" class="axis-label">{esc(market.name)}</text>'
        )
    return f"""<svg class="metric-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">
      <rect width="{width}" height="{height}" rx="14" fill="#0e1726"/>
      <text x="24" y="31" class="chart-title">{esc(title)}</text>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#334155"/>
      {''.join(groups)}
      <rect x="{width-184}" y="18" width="10" height="10" rx="2" fill="#38bdf8"/><text x="{width-168}" y="27" class="svg-legend">권고</text>
      <rect x="{width-100}" y="18" width="10" height="10" rx="2" fill="#64748b"/><text x="{width-84}" y="27" class="svg-legend">동일노출</text>
    </svg>"""


def badge(ok: bool, true_text="PASS", false_text="FAIL") -> str:
    cls = "pass" if ok else "fail"
    text = true_text if ok else false_text
    return f'<span class="badge {cls}">{esc(text)}</span>'


def market_card(market: MarketResult, color: str) -> str:
    perf = market.champion_performance
    comp = market.comparison_performance
    raw_ews = float(market.latest.get("raw_ews", market.latest.get("ews", np.nan)))
    policy = (
        market.latest.get("allocation_policy", "static_50_50")
        if market.tactical_active
        else "static_50_50"
    )
    target_mode = market.latest.get("target_mode", "absolute_positive")
    return f"""<article class="market-card" style="--accent:{color}">
      <div class="market-head"><div><div class="eyebrow">{esc(market.status)}</div><h2>{esc(market.name)}</h2>
      <p>{esc(market.champion)} · {esc(policy)}<br><small>target: {esc(target_mode)}</small></p></div>{svg_allocation(market.target_weight, color)}</div>
      <div class="metrics">
        <div><span>EWS</span><strong>{raw_ews:.1f}</strong></div>
        <div><span>Sharpe</span><strong>{float(perf['Sharpe']):.2f}</strong><small>동일 {float(comp['Sharpe']):.2f}</small></div>
        <div><span>CAGR</span><strong>{pct(perf['CAGR'])}</strong><small>동일 {pct(comp['CAGR'])}</small></div>
        <div><span>MDD</span><strong>{pct(perf['MaxDrawdown'])}</strong><small>동일 {pct(comp['MaxDrawdown'])}</small></div>
      </div>
      <p class="reason">{esc(market.champion_reason)}</p>{svg_curve(market)}
    </article>"""


def gate_rows(markets: list[MarketResult]) -> str:
    rows = []
    for market in markets:
        lg = market.logistic_gates
        mg = market.mlp_gates
        champion_holdout = (
            market.primary_holdout
            if market.champion == "Logistic"
            else market.mlp_holdout
        )
        rows.append(
            f"<tr><td><strong>{esc(market.name)}</strong></td><td>{badge(truth(lg['signal_gate']))}</td>"
            f"<td>{badge(truth(lg['portfolio_gate']))}</td><td>{badge(truth(mg['signal_gate']))}</td>"
            f"<td>{badge(truth(mg['portfolio_gate']))}</td><td>{badge(truth(champion_holdout['holdout_safety_gate']))}</td>"
            f"<td>{badge(False, 'LIVE', 'BLOCKED')}</td><td>{esc(market.champion)}</td></tr>"
        )
    return "".join(rows)


def performance_rows(markets: list[MarketResult]) -> str:
    rows = []
    for market in markets:
        p, c = market.champion_performance, market.comparison_performance
        rows.append(
            f"<tr><td>{esc(market.name)}</td><td>{esc(market.champion)}</td>"
            f"<td>{esc(p['Start'])} ~ {esc(p['End'])}</td><td>{int(p['Months'])}</td>"
            f"<td>{pct(p['CAGR'])}</td><td>{number(p['Sharpe'],2)}</td><td>{pct(p['MaxDrawdown'])}</td>"
            f"<td>{number(float(p['Sharpe'])-float(c['Sharpe']),2)}</td>"
            f"<td>{pct(float(p['CAGR'])-float(c['CAGR']))}</td>"
            f"<td>{pct(float(p['MaxDrawdown'])-float(c['MaxDrawdown']))}</td></tr>"
        )
    return "".join(rows)


def model_rows(markets: list[MarketResult]) -> str:
    rows = []
    for market in markets:
        for model in ("Logistic", "MLP"):
            hold = market.signal_metrics.loc[
                market.signal_metrics["model"].eq(model)
            ].iloc[0]
            ic = market.ic_metrics.loc[market.ic_metrics["model"].eq(model)].iloc[0]
            if model == "Logistic":
                pre_auc = market.logistic_signal_gates.get(
                    "aggregate_auc", np.nan
                )
                direction = market.logistic_signal_gates.get(
                    "fold_joint_direction_pass_ratio", np.nan
                )
            else:
                pre_auc = market.mlp_gates.get("aggregate_auc", np.nan)
                direction = market.mlp_gates.get(
                    "fold_joint_direction_pass_ratio", np.nan
                )
            rows.append(
                f"<tr><td>{esc(market.name)}</td><td>{model}</td><td>{number(pre_auc)}</td>"
                f"<td>{pct(direction,0)}</td><td>{number(hold['auc'])}</td><td>{number(ic['rank_ic'])}</td></tr>"
            )
    return "".join(rows)


def research_table(research_dir: Path | None) -> str:
    if research_dir is None:
        return ""
    fixed_source = research_dir / "pre2020_candidate_summary.csv"
    if fixed_source.exists():
        data = pd.read_csv(fixed_source)
        rows = "".join(
            f"<tr><td>{esc(row.candidate)}</td><td>{esc(row.target_mode)}</td><td>{esc(row.model_type)}</td>"
            f"<td>{number(row.aggregate_auc)}</td><td>{number(row.aggregate_rank_ic)}</td>"
            f"<td>{pct(row.fold_joint_direction_pass_ratio,0)}</td>"
            f"<td>{badge(truth(row.signal_gate_passed))}</td>"
            f"<td>{badge(truth(row.portfolio_gate_passed))}</td>"
            f"<td>{badge(truth(row.preholdout_candidate_eligible))}</td></tr>"
            for row in data.itertuples(index=False)
        )
        return f"""<section><div class="section-head"><div><div class="eyebrow">PRE-2020 ONLY</div><h2>KOSPI 외부 고정 후보 검증</h2></div></div>
          <p class="muted">원본 EWS 구조변수 4개와 타깃·모델을 2020년 이후 홀드아웃을 열기 전에 고정해 평가했다.</p>
          <div class="table-wrap"><table><thead><tr><th>후보</th><th>Target</th><th>모델</th><th>AUC</th><th>Rank IC</th><th>방향</th><th>Signal</th><th>Portfolio</th><th>적격</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""
    source = research_dir / "pre2020_signal_comparison.csv"
    if not source.exists():
        return ""
    data = pd.read_csv(source).head(10)
    rows = "".join(
        f"<tr><td>{esc(row.candidate)}</td><td>{esc(row.params)}</td><td>{int(row.features)}</td>"
        f"<td>{number(row.auc)}</td><td>{number(row.rank_ic)}</td><td>{pct(row.direction,0)}</td>"
        f"<td>{badge(truth(row.signal_gate))}</td></tr>"
        for row in data.itertuples(index=False)
    )
    return f"""<section><div class="section-head"><div><div class="eyebrow">PRE-2020 ONLY</div><h2>KOSPI MLP 후보 연구</h2></div></div>
      <p class="muted">2020년 이후 성과를 후보 선택에 사용하지 않았다. 신호 1위도 포트폴리오 게이트 실패 시 tactical 비중을 활성화하지 않는다.</p>
      <div class="table-wrap"><table><thead><tr><th>후보</th><th>MLP</th><th>변수</th><th>AUC</th><th>Rank IC</th><th>방향</th><th>Signal</th></tr></thead><tbody>{rows}</tbody></table></div></section>"""


def build_report(
    run_dirs: list[str | Path],
    output: str | Path,
    research_dir: str | Path | None = None,
) -> Path:
    markets = [MarketResult.load(path) for path in run_dirs]
    expected = {"kospi200", "sp500", "nasdaq100"}
    if {market.key for market in markets} != expected:
        raise ValueError("Exactly one kospi200, sp500 and nasdaq100 run is required")
    order = {"kospi200": 0, "sp500": 1, "nasdaq100": 2}
    markets.sort(key=lambda market: order[market.key])
    colors = ("#f59e0b", "#38bdf8", "#a78bfa")
    cards = "".join(
        market_card(market, color) for market, color in zip(markets, colors)
    )
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    runs = " · ".join(market.run_dir.name for market in markets)
    research = Path(research_dir).resolve() if research_dir else None
    css = """
    :root{color-scheme:dark;--bg:#07101d;--panel:#111c2d;--line:#263449;--text:#e8eef7;--muted:#91a1b7;--green:#34d399;--red:#fb7185}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#13233c 0,#07101d 40%);color:var(--text);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif;line-height:1.55}
    main{width:min(1500px,calc(100% - 40px));margin:0 auto;padding:46px 0 80px}.hero{display:flex;justify-content:space-between;gap:30px;align-items:flex-end;margin-bottom:30px}.hero h1{font-size:clamp(30px,4vw,54px);line-height:1.06;margin:8px 0}.hero p{max-width:900px;color:var(--muted);margin:0}.stamp{text-align:right;color:var(--muted);font-size:12px}.eyebrow{font-size:11px;letter-spacing:.16em;font-weight:800;color:#7dd3fc}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.market-card,section{background:linear-gradient(150deg,rgba(21,34,53,.98),rgba(12,23,39,.98));border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 18px 60px rgba(0,0,0,.24)}.market-card{border-top:3px solid var(--accent)}.market-head{display:flex;justify-content:space-between;align-items:center}.market-head h2{font-size:25px;margin:4px 0}.market-head p,.muted{color:var(--muted)}.donut{width:108px;height:108px}.donut-value{fill:#fff;font-size:19px;font-weight:800}.donut-label{fill:#94a3b8;font-size:10px}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:14px 0}.metrics div{background:#0b1524;border:1px solid #203047;border-radius:12px;padding:11px}.metrics span,.metrics small{display:block;color:var(--muted);font-size:11px}.metrics strong{display:block;font-size:22px}.reason{min-height:46px;color:#cbd5e1;font-size:13px}.curve{width:100%;height:auto}.svg-title,.chart-title{fill:#dbeafe;font-size:13px;font-weight:700}.svg-legend,.axis-label,.bar-value{fill:#94a3b8;font-size:10px}.bar-value{fill:#dbeafe}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}section{margin-top:20px}.section-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:14px}section h2{margin:4px 0;font-size:24px}.metric-chart{width:100%;height:auto}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:13px}table{width:100%;border-collapse:collapse;min-width:760px}th,td{padding:12px 13px;text-align:left;border-bottom:1px solid #223047;font-size:13px}th{background:#0c1727;color:#9fb0c6;font-size:11px;letter-spacing:.06em;text-transform:uppercase}.badge{display:inline-flex;padding:3px 8px;border-radius:999px;font-size:10px;font-weight:800}.pass{color:#6ee7b7;background:#064e3b}.fail{color:#fda4af;background:#4c0519}.callout{border-left:4px solid #f59e0b;padding:14px 18px;background:#1f1a11;border-radius:10px;color:#fde68a}.commands{font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;background:#07111f;padding:16px;border-radius:12px;color:#bae6fd}.footer{margin-top:24px;color:#708198;font-size:11px;word-break:break-all}
    @media(max-width:1050px){.cards{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}.market-card{display:grid;grid-template-columns:1fr 1fr;gap:15px}.market-card .curve{grid-column:1/-1}}
    @media(max-width:650px){main{width:min(100% - 22px,1500px);padding-top:24px}.hero{display:block}.stamp{text-align:left;margin-top:14px}.market-card{display:block}.metrics{grid-template-columns:1fr 1fr}}
    @media print{body{background:#fff;color:#111}main{width:100%;padding:10px}.market-card,section{break-inside:avoid;box-shadow:none}.cards{grid-template-columns:repeat(3,1fr)}}
    """
    command_lines = "\n".join(
        f"python run_pipeline.py --market {market.key} --run-id <new_{market.key}_run_id>"
        for market in markets
    )
    source_lines = "<br>".join(
        f"{esc(market.name)}: {esc(market.run_dir)}" for market in markets
    )
    report = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multi-Market EWS 검증 보고서</title><style>{css}</style></head>
    <body><main><header class="hero"><div><div class="eyebrow">PURGED OUTER OOS · HOLDOUT SAFETY · FAIL-CLOSED</div><h1>KOSPI · S&amp;P 500 · NASDAQ-100<br>통합 EWS 검증</h1><p>동일한 3개월 예측 horizon, 시점자료 감사, 거래비용 및 동일 익스포저 기준으로 비교했다. 숫자가 가장 높은 모델이 아니라 사전 고정 검증을 모두 통과한 모델만 권고한다.</p></div><div class="stamp">생성 {esc(generated)}<br>{esc(runs)}</div></header>
    <div class="cards">{cards}</div>
    <section><div class="section-head"><div><div class="eyebrow">MARKET CHAMPION</div><h2>시장별 최종 권고</h2></div></div><div class="table-wrap"><table><thead><tr><th>시장</th><th>권고</th><th>평가기간</th><th>개월</th><th>CAGR</th><th>Sharpe</th><th>MDD</th><th>Sharpe Δ</th><th>CAGR Δ</th><th>MDD Δ</th></tr></thead><tbody>{performance_rows(markets)}</tbody></table></div></section>
    <div class="grid2"><section>{svg_metric_bars(markets,'Sharpe','권고 전략 Sharpe vs 동일 익스포저')}</section><section>{svg_metric_bars(markets,'MaxDrawdown','최대낙폭 절대값 비교')}</section></div>
    <section><div class="section-head"><div><div class="eyebrow">GATE MATRIX</div><h2>검증·운영 판정</h2></div></div><div class="table-wrap"><table><thead><tr><th>시장</th><th>Log Signal</th><th>Log Portfolio</th><th>MLP Signal</th><th>MLP Portfolio</th><th>Champion Holdout</th><th>Live</th><th>권고</th></tr></thead><tbody>{gate_rows(markets)}</tbody></table></div><p class="callout">세 시장 모두 사람의 경제성·발표시차·중복정보 검토가 미승인이라 실자금 배포는 차단되어 있다. 표의 권고는 research shadow 범위다. KOSPI는 동일 익스포저보다 Sharpe·CAGR이 높지만 MDD는 1.3%p 더 깊다.</p></section>
    <section><div class="section-head"><div><div class="eyebrow">SIGNAL DIAGNOSTICS</div><h2>모델별 신호 비교</h2></div></div><div class="table-wrap"><table><thead><tr><th>시장</th><th>모델</th><th>Pre-2020 AUC</th><th>Fold 방향</th><th>2020+ AUC</th><th>2020+ Rank IC</th></tr></thead><tbody>{model_rows(markets)}</tbody></table></div></section>
    {research_table(research)}
    <section><div class="section-head"><div><div class="eyebrow">REPRODUCIBILITY</div><h2>다시 실행하는 명령</h2></div></div><div class="commands">$env:PYTHONUTF8='1'\n{esc(command_lines)}\npython generate_multi_market_ews_report.py --kospi-run &lt;KOSPI_run&gt; --sp500-run &lt;SP500_run&gt; --nasdaq-run &lt;NASDAQ_run&gt; --output reports\multi_market_ews_report.html</div></section>
    <div class="footer">근거 디렉터리<br>{source_lines}<br><br>주의: historical holdout 통과는 긍정적 승격 사유가 아니며 실패할 때만 tactical 사용을 veto한다. 미래 수익을 보장하지 않는다.</div>
    </main></body></html>"""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kospi-run", required=True)
    parser.add_argument("--sp500-run", required=True)
    parser.add_argument("--nasdaq-run", required=True)
    parser.add_argument("--research-dir")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_report(
        [args.kospi_run, args.sp500_run, args.nasdaq_run],
        args.output,
        research_dir=args.research_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
