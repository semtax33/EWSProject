"""Generate a self-contained Korean HTML report from an immutable EWS run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value, default=math.nan):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def truth(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def pct(value, digits=1, signed=False):
    value = number(value)
    if not math.isfinite(value):
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{digits}f}%"


def decimal(value, digits=3):
    value = number(value)
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def esc(value):
    return html.escape(str(value), quote=True)


def parse_date(value):
    return datetime.strptime(value[:10], "%Y-%m-%d")


def status_pill(label, passed, pending=False):
    state = "pending" if pending else "pass" if passed else "fail"
    marker = "●"
    return f'<span class="status {state}">{marker} {esc(label)}</span>'


def nice_ticks(low, high, count=5):
    if not math.isfinite(low) or not math.isfinite(high):
        return [0]
    if low == high:
        return [low]
    raw_step = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude
    step = next(choice for choice in (1, 2, 2.5, 5, 10) if normalized <= choice) * magnitude
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks = []
    value = start
    while value <= end + step * 0.1 and len(ticks) < 20:
        ticks.append(value)
        value += step
    return ticks


def svg_line_chart(
    title,
    rows,
    series,
    *,
    y_label="",
    percent_axis=False,
    y_domain=None,
    reference_lines=(),
    height=300,
):
    """Create an accessible inline SVG multi-line chart."""
    width = 760
    left, right, top, bottom = 64, 22, 38, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    parsed = []
    for row in rows:
        try:
            date = parse_date(row["date"])
        except (KeyError, TypeError, ValueError):
            continue
        values = {key: number(row.get(key)) for key, _, _ in series}
        if any(math.isfinite(value) for value in values.values()):
            parsed.append((date, values))
    if not parsed:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><title>{esc(title)}</title><text x="20" y="40">데이터 없음</text></svg>'

    dates = [item[0] for item in parsed]
    all_values = [
        values[key]
        for _, values in parsed
        for key, _, _ in series
        if math.isfinite(values[key])
    ]
    low, high = (min(all_values), max(all_values)) if y_domain is None else y_domain
    if reference_lines:
        low = min(low, *(value for value, _, _ in reference_lines))
        high = max(high, *(value for value, _, _ in reference_lines))
    padding = (high - low) * 0.08 if high > low else 1
    if y_domain is None:
        low -= padding
        high += padding
    ticks = nice_ticks(low, high, 5)
    low, high = min(ticks), max(ticks)

    def sx(index):
        return left + (plot_w * index / max(len(parsed) - 1, 1))

    def sy(value):
        return top + (high - value) / max(high - low, 1e-12) * plot_h

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{esc(title)}</title>',
        f'<desc>{esc(title)} 시계열. 기간 {dates[0].date()}부터 {dates[-1].date()}까지.</desc>',
        '<g class="grid">',
    ]
    for tick in ticks:
        y = sy(tick)
        label = f"{tick:.0f}%" if percent_axis else f"{tick:.2f}" if abs(tick) < 10 else f"{tick:.0f}"
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{esc(label)}</text>',
            ]
        )
    tick_indices = sorted({round(i * (len(parsed) - 1) / 5) for i in range(6)})
    for index in tick_indices:
        x = sx(index)
        label = parsed[index][0].strftime("%Y-%m")
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>',
                f'<text x="{x:.1f}" y="{height-14}" text-anchor="middle">{label}</text>',
            ]
        )
    parts.append("</g>")
    for ref_value, ref_label, ref_class in reference_lines:
        y = sy(ref_value)
        parts.append(
            f'<line class="reference {esc(ref_class)}" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="reference-label" x="{width-right-4}" y="{y-5:.1f}" text-anchor="end">{esc(ref_label)}</text>'
        )

    legend_x = left
    for _, label, css_class in series:
        parts.append(f'<line class="series {esc(css_class)}" x1="{legend_x}" y1="18" x2="{legend_x+24}" y2="18"/>')
        parts.append(f'<text class="legend" x="{legend_x+30}" y="22">{esc(label)}</text>')
        legend_x += max(116, len(label) * 9 + 48)

    for key, label, css_class in series:
        segments = []
        current = []
        for index, (date, values) in enumerate(parsed):
            value = values[key]
            if math.isfinite(value):
                current.append((index, date, value))
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        for segment in segments:
            points = " ".join(f"{sx(index):.1f},{sy(value):.1f}" for index, _, value in segment)
            parts.append(f'<polyline class="series {esc(css_class)}" points="{points}"/>')
        for index, date, value in [point for segment in segments for point in segment]:
            display = f"{value:.1f}%" if percent_axis else f"{value:.3f}"
            parts.append(
                f'<circle class="hover-point {esc(css_class)}" cx="{sx(index):.1f}" cy="{sy(value):.1f}" r="4"><title>{date.date()} · {esc(label)} {display}</title></circle>'
            )
    if y_label:
        parts.append(
            f'<text class="axis-title" x="16" y="{top + plot_h/2:.1f}" transform="rotate(-90 16 {top + plot_h/2:.1f})" text-anchor="middle">{esc(y_label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(title, rows, market_name="KOSPI200"):
    width, height = 760, 300
    left, right, top, bottom = 64, 22, 28, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [
        (number(row.get("ews")), number(row.get("future_return")) * 100, row.get("date", ""))
        for row in rows
    ]
    points = [(x, y, date) for x, y, date in points if math.isfinite(x) and math.isfinite(y)]
    if not points:
        return ""
    x_low, x_high = min(x for x, _, _ in points), max(x for x, _, _ in points)
    y_low, y_high = min(y for _, y, _ in points), max(y for _, y, _ in points)
    x_ticks, y_ticks = nice_ticks(x_low, x_high, 6), nice_ticks(y_low, y_high, 5)
    x_low, x_high = min(x_ticks), max(x_ticks)
    y_low, y_high = min(y_ticks), max(y_ticks)

    def sx(value):
        return left + (value - x_low) / max(x_high - x_low, 1e-12) * plot_w

    def sy(value):
        return top + (y_high - value) / max(y_high - y_low, 1e-12) * plot_h

    mean_x = sum(x for x, _, _ in points) / len(points)
    mean_y = sum(y for _, y, _ in points) / len(points)
    denom = sum((x - mean_x) ** 2 for x, _, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y, _ in points) / denom if denom else 0
    intercept = mean_y - slope * mean_x
    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{esc(title)}</title><desc>EWS와 이후 3개월 {esc(market_name)} 수익률 산점도, {len(points)}개 관측.</desc>',
        '<g class="grid">',
    ]
    for tick in y_ticks:
        parts.append(f'<line x1="{left}" y1="{sy(tick):.1f}" x2="{width-right}" y2="{sy(tick):.1f}"/>')
        parts.append(f'<text x="{left-10}" y="{sy(tick)+4:.1f}" text-anchor="end">{tick:.0f}%</text>')
    for tick in x_ticks:
        parts.append(f'<line x1="{sx(tick):.1f}" y1="{top}" x2="{sx(tick):.1f}" y2="{height-bottom}"/>')
        parts.append(f'<text x="{sx(tick):.1f}" y="{height-14}" text-anchor="middle">{tick:.0f}</text>')
    parts.append("</g>")
    y1, y2 = intercept + slope * x_low, intercept + slope * x_high
    parts.append(f'<line class="trend" x1="{sx(x_low):.1f}" y1="{sy(y1):.1f}" x2="{sx(x_high):.1f}" y2="{sy(y2):.1f}"/>')
    for x, y, date in points:
        parts.append(f'<circle class="scatter-point" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5"><title>{esc(date)} · EWS {x:.1f} · 이후 3개월 {y:+.1f}%</title></circle>')
    parts.extend(
        [
            f'<text class="axis-title" x="{left + plot_w/2:.1f}" y="{height-2}" text-anchor="middle">EWS</text>',
            f'<text class="axis-title" x="16" y="{top + plot_h/2:.1f}" transform="rotate(-90 16 {top + plot_h/2:.1f})" text-anchor="middle">이후 3개월 수익률</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def svg_auc_bars(models):
    width, height = 760, 250
    left, right, top, bottom = 90, 24, 28, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    bars = [(row["model"], number(row["auc"])) for row in models]
    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img"><title>Holdout 모델 AUC 비교</title>',
        '<desc>2020년 4월 이후 73개 관측의 Logistic, SVM, MLP AUC 비교.</desc>',
        '<g class="grid">',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-14}" text-anchor="middle">{tick:.2f}</text>')
    parts.append("</g>")
    threshold_x = left + 0.5 * plot_w
    parts.append(f'<line class="reference warn" x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{height-bottom}"/>')
    bar_h = 34
    gap = 22
    for index, (label, value) in enumerate(bars):
        y = top + index * (bar_h + gap) + 12
        css = "bar-selected" if label == "Logistic" else "bar-other"
        parts.append(f'<text x="{left-12}" y="{y+23}" text-anchor="end">{esc(label)}</text>')
        parts.append(f'<rect class="{css}" x="{left}" y="{y}" width="{value*plot_w:.1f}" height="{bar_h}" rx="5"><title>{esc(label)} AUC {value:.3f}</title></rect>')
        parts.append(f'<text class="bar-label" x="{left+value*plot_w+9:.1f}" y="{y+23}">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_report(run_dir, output_path):
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path).resolve()
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    market_name = manifest["config"].get("market_name", "KOSPI200")
    market_key = manifest["config"].get("market_key", "kospi200")
    investable_instrument = manifest["config"].get(
        "investable_instrument", "KODEX 200 ETF"
    )
    portfolio_benchmark = manifest["config"].get(
        "portfolio_benchmark_instrument", investable_instrument
    )
    quick_smoke = bool(manifest.get("quick_smoke_test", False))
    validation_label = "Pre-2020 smoke OOS" if quick_smoke else "Pre-2020 nested OOS"
    mode_chip = "Quick smoke · 연구판정 금지" if quick_smoke else "Full research run"
    latest = json.loads((run_dir / "latest_ews.json").read_text(encoding="utf-8"))
    gates = read_csv(run_dir / "deployment_gates.csv")[0]
    signal = read_csv(run_dir / "signal_gate_diagnostics.csv")[0]
    policies = read_csv(run_dir / "position_sizing_policy_gate.csv")
    selected_policy_name = latest.get(
        "allocation_policy", manifest.get("selected_allocation_policy", "linear")
    )
    selected_policy_gate = next(
        row for row in policies if row["policy"] == selected_policy_name
    )
    fallback_used = truth(latest.get("fail_closed_fallback_used", False))
    allocation_headline = (
        f"Fail-closed · 주식 {latest['target_stock_weight']*100:.0f}% / "
        f"현금 {latest['cash_weight']*100:.0f}%"
        if fallback_used
        else (
            f"{latest.get('signal_state', 'EWS')} · 주식 "
            f"{latest['target_stock_weight']*100:.0f}% / "
            f"현금 {latest['cash_weight']*100:.0f}%"
        )
    )
    allocation_explanation = (
        "신호 또는 포트폴리오 gate가 실패해 검증되지 않은 market timing을 "
        "비활성화하고 사전 지정 50/50 전략 배분을 적용했다. EWS 점수는 연구 "
        "진단으로만 표시한다."
        if fallback_used
        else (
            f"Pre-holdout gate를 통과한 {selected_policy_name} 배분을 적용했다. "
            "실전 사용 여부는 아래 strict deployment gate를 별도로 확인해야 한다."
        )
    )
    models = read_csv(run_dir / "model_comparison.csv")
    model_ic = read_csv(run_dir / "model_ic_comparison.csv")
    performance = read_csv(run_dir / "performance_comparison.csv")
    logistic_performance = [row for row in performance if row["model"] == "Logistic"]
    mlp_model = next(row for row in models if row["model"] == "MLP")
    mlp_ic = next(row for row in model_ic if row["model"] == "MLP")
    mlp_dynamic = next(
        row for row in performance
        if row["model"] == "MLP" and row["strategy"] == "MLP Dynamic"
    )
    holdout_effective_blocks = math.ceil(
        number(mlp_model["n"]) / number(manifest["config"]["forecast_horizon_months"])
    )
    features = read_csv(run_dir / "required_core_selection.csv")
    stability = {row["required_family"]: row for row in read_csv(run_dir / "required_family_coefficient_sign_stability.csv")}
    folds = read_csv(run_dir / "outer_fold_signal_metrics.csv")
    eligible_fold_count = sum(truth(row["gate_eligible_fold"]) for row in folds)
    passed_fold_count = sum(truth(row["joint_direction_passed"]) for row in folds)
    signal_gate_passed = truth(signal["signal_gate_passed"])
    signal_summary_title = (
        "연구 신호 게이트를 통과했다."
        if signal_gate_passed
        else "연구 신호 게이트를 통과하지 못했다."
    )
    signal_summary_class = "callout" if signal_gate_passed else "callout danger"
    group_summary = read_csv(run_dir / "raw_series_group_summary.csv")
    factor_summary = {row["stage"]: int(row["count"]) for row in read_csv(run_dir / "factor_universe_summary.csv")}
    vintage = read_csv(run_dir / "selected_source_point_in_time_audit.csv")
    blockers = read_csv(run_dir / "deployment_blockers.csv")
    mlp_validation_path = run_dir / "mlp_validation_gates.csv"
    mlp_latest_path = run_dir / "latest_mlp_ews.json"
    mlp_validation = (
        read_csv(mlp_validation_path)[0] if mlp_validation_path.is_file() else None
    )
    mlp_latest = (
        json.loads(mlp_latest_path.read_text(encoding="utf-8"))
        if mlp_latest_path.is_file()
        else None
    )

    backtest_rows = []
    for row in read_csv(run_dir / "logistic_backtest.csv"):
        date = row.get("") or next(iter(row.values()))
        if any(row.get(key) for key in ("strategy_curve", "market_curve", "raw_ews")):
            backtest_rows.append(
                {
                    "date": date,
                    "strategy_curve": number(row.get("strategy_curve")),
                    "market_curve": number(row.get("market_curve")),
                    "raw_ews": number(row.get("raw_ews")),
                    "stock_weight": number(row.get("executed_stock_weight")) * 100,
                    "cash_weight": number(row.get("cash_weight")) * 100,
                    "strategy_drawdown": number(row.get("strategy_drawdown")) * 100,
                    "market_drawdown": number(row.get("market_drawdown")) * 100,
                }
            )

    rolling_ic = [
        {"date": row["observation_date"], "rolling_rank_ic": number(row["rolling_rank_ic"])}
        for row in read_csv(run_dir / "logistic_rolling_ic.csv")
    ]
    rolling_sharpe = []
    for row in read_csv(run_dir / "rolling_sharpe.csv"):
        rolling_sharpe.append(
            {
                "date": row.get("") or next(iter(row.values())),
                "rolling_sharpe": number(row.get("rolling_sharpe")),
            }
        )
    ews_rows = {row["observation_date"]: number(row["ews"]) for row in read_csv(run_dir / "logistic_test_ews.csv")}
    target_rows = {row["observation_date"]: number(row["future_return"]) for row in read_csv(run_dir / "target.csv")}
    scatter_rows = [
        {"date": date, "ews": value, "future_return": target_rows.get(date, math.nan)}
        for date, value in ews_rows.items()
        if date in target_rows
    ]

    chart_cumulative = svg_line_chart(
        "누적 수익 배수",
        backtest_rows,
        (("strategy_curve", "EWS 전략", "s1"), ("market_curve", portfolio_benchmark, "s2")),
        y_label="성장 배수",
    )
    chart_allocation = svg_line_chart(
        "EWS와 실행 배분",
        backtest_rows,
        (("raw_ews", "EWS", "s0"), ("stock_weight", "주식", "s1"), ("cash_weight", "현금", "s2")),
        y_label="점수·비중",
        percent_axis=True,
        y_domain=(0, 100),
        reference_lines=((50, "중립 50", "neutral"),),
    )
    chart_drawdown = svg_line_chart(
        "낙폭",
        backtest_rows,
        (("strategy_drawdown", "EWS 전략", "s1"), ("market_drawdown", portfolio_benchmark, "s3")),
        y_label="낙폭",
        percent_axis=True,
        y_domain=(-40, 2),
        reference_lines=((0, "고점", "neutral"),),
    )
    chart_ic = svg_line_chart(
        "36개월 Rolling Rank IC",
        rolling_ic,
        (("rolling_rank_ic", "Rank IC", "s4"),),
        y_label="Rank IC",
        reference_lines=((0, "0", "neutral"), (0.1, "0.10", "good"), (-0.1, "-0.10", "warn")),
    )
    chart_sharpe = svg_line_chart(
        "36개월 Rolling Sharpe",
        rolling_sharpe,
        (("rolling_sharpe", "Sharpe", "s5"),),
        y_label="Sharpe",
        reference_lines=((0, "0", "neutral"), (1, "1.0", "good")),
    )
    chart_scatter = svg_scatter(
        "EWS와 이후 3개월 수익률", scatter_rows, market_name=market_name
    )
    chart_auc = svg_auc_bars(models)

    family_labels = {
        "turnover_trend": "거래대금/시가총액 추세",
        "term_spread": "미국 Term spread",
        "pairwise_correlation": "KOSPI 종목 Pairwise correlation",
        "return_skew_1m": "KOSPI 종목 1개월 수익률 왜도",
        "realized_volatility": f"{market_name} 실현 변동성",
        "downside_risk": f"{market_name} 하방 변동성",
        "absolute_trend": f"{market_name} 절대 가격 추세",
    }
    transform_labels = {
        "turnover_trend": "9개월 평균의 9개월 변화",
        "term_spread": "10Y−2Y의 48개월 평균, 3개월 변화",
        "pairwise_correlation": "12개월 EWMA의 24개월 변화",
        "return_skew_1m": "60개월 평균의 3개월 변화",
        "realized_volatility": "지수 일별 수익률 기반 월간 연율화 변동성",
        "downside_risk": "지수 음(-)의 일별 수익률 기반 하방 변동성",
        "absolute_trend": "12개월 모멘텀 또는 월말 종가의 10개월 이동평균 괴리",
    }

    gate_items = [
        ("신호", gates["signal_gate"], "AUC·Rank IC·fold 방향"),
        ("포트폴리오", gates["portfolio_gate"], "비용·동일 익스포저·낙폭"),
        ("Vintage", gates["point_in_time_vintage_gate"], "ALFRED 월말 스냅샷"),
        ("Release timing", gates["release_timing_gate"], "월말 정보, 다음 달 실행"),
        ("투자 가능 수익률", gates["investable_return_source_gate"], f"{portfolio_benchmark} 원천"),
        ("운영", gates["operational_gate"], "연구·forward shadow"),
        ("Strict 운영", gates["strict_operational_gate"], "사람 경제성 검토 필요"),
        ("실전 배포", gates["deployment_eligible"], "strict gate 종속"),
    ]
    gate_rows = "".join(
        f"<tr><td>{esc(label)}</td><td>{status_pill('통과' if truth(value) else '차단', truth(value), label in {'Strict 운영', '실전 배포'} and not truth(value))}</td><td>{esc(note)}</td></tr>"
        for label, value, note in gate_items
    )

    factor_rows = "".join(
        f"""
        <tr>
          <td>{esc(family_labels.get(row['required_family'], row['required_family'].replace('_', ' ').title()))}</td>
          <td>{esc(transform_labels.get(row['required_family'], row.get('feature', '')))}</td>
          <td class="num">{number(row['screen_auc']):.3f}</td>
          <td class="num">{number(row['screen_rank_score']):.3f}</td>
          <td class="num">{number(stability[row['required_family']]['sign_consistency_ratio'])*100:.0f}%</td>
        </tr>"""
        for row in features
    )

    fold_rows = "".join(
        f"""
        <tr>
          <td class="num">{row['fold']}</td><td>{esc(row['start'][:7])}–{esc(row['end'][:7])}</td>
          <td class="num">{row['observations']}</td><td class="num">{decimal(row['auc'])}</td>
          <td class="num">{decimal(row['rank_ic'])}</td>
          <td>{status_pill('통과' if truth(row['joint_direction_passed']) else ('제외' if not truth(row['gate_eligible_fold']) else '실패'), truth(row['joint_direction_passed']), not truth(row['gate_eligible_fold']))}</td>
          <td>{esc(row['failure_reason'].replace('|', ', '))}</td>
        </tr>"""
        for row in folds
    )

    model_rows = "".join(
        f"""
        <tr class="{'selected-row' if row['model'] == 'Logistic' else ''}">
          <td>{esc(row['model'])}{' · Champion' if row['model'] == 'Logistic' else ' · Challenger'}</td>
          <td class="num">{row['n']}</td><td class="num">{number(row['auc']):.3f}</td>
          <td class="num">{number(row['brier']):.3f}</td><td class="num">{pct(row['accuracy'])}</td>
          <td class="num">{pct(row['naive_accuracy'])}</td><td class="num">{pct(row['accuracy_lift_vs_naive'], signed=True)}</td>
        </tr>"""
        for row in models
    )
    if mlp_validation is not None and mlp_latest is not None:
        mlp_validation_note = f"""
    <div class="callout"><p><strong>MLP 독립 검증 및 현재 사용 범위</strong></p>
      <p>MLP는 이제 변수 선별부터 재학습까지 pre-2020 outer fold 안에서 반복 검증된다. 결과는 nested AUC {number(mlp_validation['aggregate_auc']):.3f}, Rank IC {number(mlp_validation['aggregate_rank_ic']):.3f}, fold 동시 통과율 {pct(mlp_validation['fold_joint_direction_pass_ratio'])}로 signal gate가 {'통과' if truth(mlp_validation['signal_gate']) else '실패'}했고, portfolio gate도 {'통과' if truth(mlp_validation['portfolio_gate']) else '실패'}했다.</p>
      <p>2020년 이후 holdout의 AUC {number(mlp_model['auc']):.3f}, Dynamic Sharpe {number(mlp_dynamic['Sharpe']):.2f}, MDD {pct(mlp_dynamic['MaxDrawdown'])}는 계속 유의미한 연구 진단이지만 승격에는 사용하지 않는다. 현재 MLP EWS는 {number(mlp_latest['ews']):.1f}, 목표 주식 비중은 {pct(mlp_latest['target_stock_weight'])}이며 <strong>research shadow only</strong>로 기록할 수 있다. 실제 자본 사용은 {status_pill('허용' if truth(mlp_validation['capital_use_allowed']) else '차단', truth(mlp_validation['capital_use_allowed']))} 상태다.</p>
    </div>"""
    else:
        mlp_validation_note = f"""
    <div class="callout"><p><strong>MLP를 현재 실전 전략으로 쓰지 않는 이유</strong></p>
      <p>MLP의 holdout 수치는 좋다: Rank IC {number(mlp_ic['rank_ic']):.3f}, Dynamic Sharpe {number(mlp_dynamic['Sharpe']):.2f}, MDD {pct(mlp_dynamic['MaxDrawdown'])}. 그러나 이 실행에서 nested outer-fold 재선정과 signal·portfolio gate를 통과한 사전 지정 champion은 Logistic뿐이다. MLP는 pre-holdout에서 별도 변수 조합을 고른 뒤 2020년 이후 holdout에서 비교한 challenger라, 이 성과를 보고 교체하면 holdout이 사실상 모델 선택 데이터가 된다.</p>
      <p>또한 3개월 중첩 타깃의 holdout 73개월은 단순 비중첩 환산 시 약 {holdout_effective_blocks}개 블록에 불과하고, 분류 정확도는 {pct(mlp_model['accuracy'])}로 naive {pct(mlp_model['naive_accuracy'])}보다 {pct(mlp_model['accuracy_lift_vs_naive'], signed=True)} 낮다. 정확도 자체가 핵심 gate는 아니지만 표본이 작다는 경고다. 따라서 MLP는 별도 nested OOS·배분 정책·게이트 검증을 거친 뒤 새 untouched 기간 또는 forward shadow에서 확인해야 승격할 수 있다.</p>
    </div>"""

    strategy_order = ["Dynamic", "Same Exposure", "Static 50/50", "100%"]
    performance_sorted = sorted(
        logistic_performance,
        key=lambda row: next((i for i, key in enumerate(strategy_order) if key in row["strategy"]), 99),
    )
    performance_rows = "".join(
        f"""
        <tr class="{'selected-row' if row['strategy'] == 'Logistic Dynamic' else ''}">
          <td>{esc(row['strategy'].replace('Logistic ', ''))}</td><td class="num">{pct(row['CAGR'])}</td>
          <td class="num">{decimal(row['Sharpe'])}</td><td class="num">{pct(row['MaxDrawdown'])}</td>
          <td class="num">{decimal(row['Calmar'])}</td><td class="num">{pct(row['MonthlyHitRate'])}</td>
        </tr>"""
        for row in performance_sorted
    )

    source_rows = "".join(
        f"""
        <tr><td>{esc(row['usage_role'])}</td><td>{esc(row['base'])}</td><td>{esc(row['source'])}</td>
        <td>{esc(row['release_lag_status'])}</td><td>{status_pill('검증', truth(row['strict_vintage_gate_passed']))}</td></tr>"""
        for row in vintage
    )

    group_total = sum(int(row["configured"]) for row in group_summary)
    dynamic = next(row for row in logistic_performance if row["strategy"] == "Logistic Dynamic")
    same_exposure = next(row for row in logistic_performance if "Same Exposure" in row["strategy"])
    market = next(row for row in logistic_performance if row["strategy"].endswith("100%"))
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    css = """
    :root{color-scheme:light;--ink:#152236;--muted:#607089;--line:#dbe3ee;--paper:#f5f7fb;--card:#fff;--navy:#0c1b31;--blue:#2274d6;--cyan:#24a7bd;--orange:#ef8b2c;--red:#d95050;--green:#168465;--violet:#7457c8;--shadow:0 12px 30px rgba(18,37,63,.08)}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Pretendard,"Noto Sans KR","Segoe UI",sans-serif;line-height:1.55}main{max-width:1320px;margin:auto;padding:34px 24px 72px}header{background:linear-gradient(135deg,#0b182b,#16375e);color:#fff;border-radius:24px;padding:32px 36px;box-shadow:var(--shadow)}h1{font-size:clamp(26px,4vw,42px);line-height:1.18;margin:8px 0 10px}h2{font-size:24px;margin:0 0 16px}h3{font-size:18px;margin:0 0 10px}.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#a9c7ed}.subtitle{max-width:920px;color:#d8e4f2;margin:0}.meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}.chip{display:inline-flex;gap:7px;align-items:center;border:1px solid rgba(255,255,255,.22);border-radius:999px;padding:6px 11px;font-size:13px;color:#eef6ff}.hero-grid{display:grid;grid-template-columns:180px 1fr;gap:26px;align-items:center;margin-top:28px}.gauge{width:160px;height:160px}.gauge text{fill:#fff;text-anchor:middle}.hero-copy strong{font-size:28px;font-weight:500}.hero-copy p{margin:7px 0 0;color:#d8e4f2}.section{margin-top:32px}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.kpi,.panel,.chart-card{min-width:0;background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}.kpi{padding:18px}.kpi-label{font-size:13px;color:var(--muted)}.kpi-value{font-size:27px;font-weight:500;margin-top:3px}.kpi-note{font-size:12px;color:var(--muted);margin-top:3px}.panel{padding:24px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.chart-card{padding:18px;margin:0}.chart-card figcaption{font-weight:500;margin-bottom:8px}.chart-note{font-size:12px;color:var(--muted);font-weight:400;margin-left:7px}.chart-svg{display:block;width:100%;height:auto}.chart-svg text{font-family:inherit;fill:var(--muted);font-size:12px}.chart-svg .legend,.chart-svg .axis-title,.chart-svg .bar-label{fill:var(--ink)}.grid line{stroke:var(--line);stroke-width:1}.series{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}.s0{stroke:var(--ink)}.s1{stroke:var(--blue)}.s2{stroke:var(--orange)}.s3{stroke:var(--red)}.s4{stroke:var(--violet)}.s5{stroke:var(--green)}.hover-point{fill:transparent;stroke:transparent}.hover-point:hover{fill:var(--card);stroke-width:2}.reference{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4}.reference.good{stroke:var(--green)}.reference.warn{stroke:var(--red)}.reference-label{font-size:11px}.scatter-point{fill:var(--blue);fill-opacity:.68;stroke:var(--card);stroke-width:1}.trend{stroke:var(--red);stroke-width:2.4}.bar-selected{fill:var(--blue)}.bar-other{fill:var(--cyan);fill-opacity:.72}.table-wrap{max-width:100%;overflow-x:auto}.table{width:100%;border-collapse:collapse;font-size:14px}.table th{text-align:left;color:var(--muted);font-weight:500;border-bottom:2px solid var(--line);padding:10px 9px;white-space:nowrap}.table td{border-bottom:1px solid var(--line);padding:11px 9px;vertical-align:top}.table .num{text-align:right;font-variant-numeric:tabular-nums}.selected-row{background:#edf5ff}.status{display:inline-flex;align-items:center;gap:5px;white-space:nowrap;font-size:12px;font-weight:500}.status.pass{color:var(--green)}.status.fail{color:var(--red)}.status.pending{color:#a66b11}.callout{border-left:4px solid var(--orange);padding:12px 16px;background:#fff7ec;border-radius:0 12px 12px 0}.callout.danger{border-left-color:var(--red);background:#fff2f2}.callout p{margin:4px 0}.lead{font-size:17px}.muted{color:var(--muted)}.metric-line{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding:9px 0}.metric-line:last-child{border:0}.metric-line strong{font-weight:500;text-align:right}.footer{margin-top:30px;color:var(--muted);font-size:12px}.command{background:#0e192a;color:#dce9f7;border-radius:12px;padding:14px 16px;overflow-wrap:anywhere;font-family:Consolas,monospace;font-size:13px}.small{font-size:12px}.nowrap{white-space:nowrap}
    html,body{max-width:100%;overflow-x:hidden}
    @media(max-width:960px){.kpis{grid-template-columns:repeat(2,1fr)}.chart-grid,.two-col{grid-template-columns:1fr}.hero-grid{grid-template-columns:150px 1fr}.gauge{width:140px;height:140px}}
    @media(max-width:560px){main{padding:18px 12px 42px}header{padding:24px 20px;border-radius:18px}.hero-grid{grid-template-columns:1fr}.kpis{grid-template-columns:1fr}.panel,.chart-card{padding:16px}.gauge{margin:auto}.meta{gap:6px}.chip{font-size:12px}}
    @media print{body{background:#fff}main{max-width:none;padding:0}.panel,.kpi,.chart-card,header{box-shadow:none;break-inside:avoid}.section{break-inside:avoid}.chart-grid{grid-template-columns:1fr 1fr}}
    """

    report = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(market_name)} EWS 실행 결과 요약 · {esc(manifest['run_id'])}</title><style>{css}</style></head>
<body><main>
<header>
  <div class="eyebrow">Macro EWS · Reproducible run report</div>
  <h1>{esc(market_name)} EWS 실행 결과 요약</h1>
  <p class="subtitle">실행 로그와 <strong>{esc(manifest['run_id'])}</strong>의 immutable 산출물을 결합한 보고서. {esc(validation_label)} 검증, 2020년 이후 holdout 진단, 투자 가능 수익률과 운영 상태를 서로 분리해 표시했다.</p>
  <div class="meta"><span class="chip">Run complete</span><span class="chip">{esc(mode_chip)}</span><span class="chip">기준일 {esc(latest['date'])}</span><span class="chip">성과 {esc(latest['performance_start'])}–{esc(latest['performance_end'])}</span><span class="chip">Logistic champion</span></div>
  <div class="hero-grid">
    <svg class="gauge" viewBox="0 0 120 120" role="img"><title>현재 EWS {latest['ews']:.1f}점</title>
      <circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255,255,255,.14)" stroke-width="11"/>
      <circle cx="60" cy="60" r="48" fill="none" stroke="#5fe0b7" stroke-width="11" stroke-linecap="round" pathLength="100" stroke-dasharray="{latest['ews']:.1f} 100" transform="rotate(-90 60 60)"/>
      <text x="60" y="58" font-size="28" font-weight="500">{latest['ews']:.1f}</text><text x="60" y="76" font-size="11">EWS / 100</text>
    </svg>
    <div class="hero-copy"><strong>{esc(allocation_headline)}</strong>
      <p>{esc(allocation_explanation)}</p>
    </div>
  </div>
</header>

<section class="section kpis">
  <div class="kpi"><div class="kpi-label">{esc(validation_label)} AUC</div><div class="kpi-value">{number(signal['aggregate_auc']):.3f}</div><div class="kpi-note">기준 0.50 초과 · signal gate</div></div>
  <div class="kpi"><div class="kpi-label">Pre-2020 Rank IC</div><div class="kpi-value">{number(signal['aggregate_rank_ic']):.3f}</div><div class="kpi-note">fold 공동 통과 {number(signal['fold_joint_direction_pass_ratio'])*100:.0f}%</div></div>
  <div class="kpi"><div class="kpi-label">Holdout 선택전략 Sharpe</div><div class="kpi-value">{number(dynamic['Sharpe']):.2f}</div><div class="kpi-note">{esc(selected_policy_name)} · 동일 익스포저 {number(same_exposure['Sharpe']):.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Holdout 최대 낙폭</div><div class="kpi-value">{pct(dynamic['MaxDrawdown'])}</div><div class="kpi-note">{esc(portfolio_benchmark)} {pct(market['MaxDrawdown'])}</div></div>
</section>

<section class="section two-col">
  <article class="panel"><h2>판정 요약</h2><div class="table-wrap"><table class="table"><thead><tr><th>Gate</th><th>판정</th><th>근거</th></tr></thead><tbody>{gate_rows}</tbody></table></div></article>
  <article class="panel"><h2>해석의 핵심</h2>
    <div class="{signal_summary_class}"><p><strong>{esc(signal_summary_title)}</strong></p><p>{esc(validation_label)}에서 AUC {number(signal['aggregate_auc']):.3f}, Rank IC {number(signal['aggregate_rank_ic']):.3f}, 평가 가능 fold {eligible_fold_count}개 중 {passed_fold_count}개가 방향 조건을 함께 만족했다.</p></div>
    <div class="callout danger" style="margin-top:14px"><p><strong>하지만 2020년 이후 Logistic AUC는 {number(latest['test_auc']):.2f}다.</strong></p><p>정확도 {pct(latest['test_accuracy'])}는 naive {pct(latest['naive_accuracy'])}보다 {pct(number(latest['test_accuracy'])-number(latest['naive_accuracy']),signed=True)}p 낮다. 예측 방향은 사후에 뒤집지 않았고, Sharpe만으로 실전 배포하지 않는다.</p></div>
    <p class="small muted">`operational_gate=True`는 연구용 waiver를 포함한 운영 허용이다. `strict_operational_gate=False` 및 `deployment_eligible=False`가 실전 판정이다.</p>
  </article>
</section>

<section class="section panel"><h2>데이터와 연구 설계</h2>
  <div class="two-col"><div>
    <div class="metric-line"><span>FRED 원자료 / 월간 패널</span><strong>51개 / 1,291행</strong></div>
    <div class="metric-line"><span>전체 raw universe</span><strong>{factor_summary['available_raw_series']} / {group_total}개 사용 가능</strong></div>
    <div class="metric-line"><span>연구 적격 raw series</span><strong>{factor_summary['research_eligible_raw_series']}개</strong></div>
    <div class="metric-line"><span>최종 factor matrix</span><strong>{factor_summary['all_factor_matrix']:,}개</strong></div>
  </div><div>
    <div class="metric-line"><span>Development</span><strong>1996-03–2014-03</strong></div>
    <div class="metric-line"><span>Validation</span><strong>2014-04–2020-03</strong></div>
    <div class="metric-line"><span>Research holdout</span><strong>2020-04–2026-04</strong></div>
    <div class="metric-line"><span>최소 초기 학습</span><strong>60개월 · 최종 310표본</strong></div>
  </div></div>
  <p class="small muted">선택 시장의 표준 OHLCV CSV를 검증해 사용한다. 기존 KOSPI200의 밀린 yfinance 헤더 형식도 감지 후 복구하며, 고가·저가 범위와 거래량을 다시 검증한다.</p>
</section>

<section class="section panel"><h2>필수 {len(features)}개 Factor family</h2>
  <div class="table-wrap"><table class="table"><thead><tr><th>Family</th><th>선택 변환</th><th class="num">Screen AUC</th><th class="num">Rank score</th><th class="num">계수부호 일관성</th></tr></thead><tbody>{factor_rows}</tbody></table></div>
  <p class="small muted">변환은 2020-03 이전 causal screen에서 선택됐다. 계수부호 일관성은 family 수준의 진단이며 인과 승인이나 deployment gate가 아니다.</p>
</section>

<section class="section panel"><h2>{esc(validation_label)} outer-fold 검증</h2>
  <div class="table-wrap"><table class="table"><thead><tr><th class="num">Fold</th><th>기간</th><th class="num">N</th><th class="num">AUC</th><th class="num">Rank IC</th><th>판정</th><th>사유</th></tr></thead><tbody>{fold_rows}</tbody></table></div>
  <p class="small muted">사전 최소 관측기간보다 짧은 fold는 방향 통과 비율에서 제외하되, 각 fold의 실제 AUC·IC와 제외 사유는 결과 파일에 그대로 보존한다.</p>
</section>

<section class="section panel"><h2>포트폴리오 Gate</h2>
  <div class="two-col"><div>
    <div class="metric-line"><span>선택 정책</span><strong>{esc(selected_policy_name)}{' · fail-closed' if fallback_used else ''}</strong></div>
    <div class="metric-line"><span>Fold 중앙 Sharpe 개선</span><strong>{number(selected_policy_gate['median_fold_Sharpe_difference']):+.3f} <span class="muted">(기준 +0.10)</span></strong></div>
    <div class="metric-line"><span>양의 active-return fold</span><strong>{pct(selected_policy_gate['positive_fold_ratio'])}</strong></div>
  </div><div>
    <div class="metric-line"><span>10bp 후 연환산 active return</span><strong>{pct(selected_policy_gate['annualized_active_return_10bps'],2,signed=True)}</strong></div>
    <div class="metric-line"><span>25bp 후 연환산 active return</span><strong>{pct(selected_policy_gate['annualized_active_return_25bps'],2,signed=True)}</strong></div>
    <div class="metric-line"><span>25bp MDD 차이</span><strong>{pct(selected_policy_gate['drawdown_difference_25bps'],2,signed=True)}p</strong></div>
  </div></div>
</section>

<section class="section"><h2>EWS 대시보드</h2><div class="chart-grid">
  <figure class="chart-card"><figcaption>누적 수익 <span class="chart-note">{esc(portfolio_benchmark)} 기준</span></figcaption>{chart_cumulative}</figure>
  <figure class="chart-card"><figcaption>EWS와 실행 배분</figcaption>{chart_allocation}</figure>
  <figure class="chart-card"><figcaption>낙폭 비교</figcaption>{chart_drawdown}</figure>
  <figure class="chart-card"><figcaption>Rolling Rank IC</figcaption>{chart_ic}</figure>
  <figure class="chart-card"><figcaption>Rolling Sharpe</figcaption>{chart_sharpe}</figure>
  <figure class="chart-card"><figcaption>EWS와 이후 수익률</figcaption>{chart_scatter}</figure>
</div></section>

<section class="section two-col">
  <article class="panel"><h2>2020년 이후 모델 진단</h2>{chart_auc}
    <div class="table-wrap"><table class="table"><thead><tr><th>모델</th><th class="num">N</th><th class="num">AUC</th><th class="num">Brier</th><th class="num">Accuracy</th><th class="num">Naive</th><th class="num">Lift</th></tr></thead><tbody>{model_rows}</tbody></table></div>
    <p class="small muted">MLP AUC {number(mlp_model['auc']):.3f}는 challenger 진단이다. Holdout 성과를 이용해 champion을 교체하지 않았다.</p>
    {mlp_validation_note}
  </article>
  <article class="panel"><h2>Logistic 동일기간 성과</h2>
    <div class="table-wrap"><table class="table"><thead><tr><th>전략</th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">MDD</th><th class="num">Calmar</th><th class="num">월 승률</th></tr></thead><tbody>{performance_rows}</tbody></table></div>
    <p class="small muted">Dynamic은 {esc(portfolio_benchmark)}의 CAGR {pct(market['CAGR'])}와 비교되며 MDD는 {pct(dynamic['MaxDrawdown'])}다. Static 50/50의 MDD는 {pct(next(row for row in logistic_performance if 'Static 50/50' in row['strategy'])['MaxDrawdown'])}다.</p>
  </article>
</section>

<section class="section panel"><h2>Point-in-time 및 투자 가능성 감사</h2>
  <div class="table-wrap"><table class="table"><thead><tr><th>역할</th><th>원천</th><th>종류</th><th>가용시점 처리</th><th>판정</th></tr></thead><tbody>{source_rows}</tbody></table></div>
  <p class="small muted">T10Y2Y와 현금 leg DGS3MO는 월말 ALFRED vintage를 사용한다. 시장내부 지표는 완료된 월말 관측을 다음 달에 실행한다. 포트폴리오 성과 기준은 {esc(portfolio_benchmark)}다.</p>
</section>

<section class="section two-col">
  <article class="panel"><h2>남은 차단 사항</h2>
    <div class="callout danger"><p><strong>Strict deployment는 아직 불가</strong></p><p>{len(blockers)}개 blocker는 모두 사람의 경제성 검토 범주다: 경제적 채널 승인, 실제 발표 캘린더 검토, 중복정보 검토.</p></div>
    <p>현재 허용 범위는 <strong>forward shadow research</strong>다. 실거래·자동 배포는 승인되지 않았다.</p>
  </article>
  <article class="panel"><h2>재현 정보</h2>
    <div class="metric-line"><span>Run ID</span><strong>{esc(manifest['run_id'])}</strong></div>
    <div class="metric-line"><span>Manifest 상태</span><strong>{esc(manifest['status'])}</strong></div>
    <div class="metric-line"><span>Python / pandas</span><strong>{esc(manifest['python'])} / {esc(manifest['packages']['pandas'])}</strong></div>
    <div class="metric-line"><span>랜덤 시드</span><strong>{esc(manifest['config']['random_seed'])}</strong></div>
    <p class="command">$env:PYTHONUTF8='1'; python run_pipeline.py --market {esc(market_key)} --run-id &lt;새_run_id&gt;</p>
  </article>
</section>

<footer class="footer">생성 시각 {esc(generated_at)} · 근거 폴더 {esc(run_dir)} · 모든 차트는 해당 run의 CSV에서 인라인 SVG로 재계산됨.</footer>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = build_report(args.run_dir, args.output)
    print(path)


if __name__ == "__main__":
    main()
