# EWS 연구 파이프라인 실행 가이드

## 기준 실행

```powershell
$env:PYTHONIOENCODING='utf-8'
python build_market_breadth.py
python download_alfred_vintages.py
python download_investable_returns.py
python run_pipeline.py --run-id <고유한_run_id>
python run_regression_research.py --classification-run runs\<고유한_run_id> --run-id <회귀_run_id>
python run_benchmark_allocation_research.py --parent-run runs\<고유한_run_id> --run-id <배분_run_id>
python -m unittest discover -s tests -v
```

기본 설정은 `download_investable_returns.py`가 만든 KODEX 200 adjusted close를
포트폴리오 수익률에 사용한다. 다른 KOSPI200 TR/NTR 또는 ETF를 쓰려면 adjusted
close 또는 total-return 열이 있는 CSV를 명시한다. 단순 `close` 열만 있는 파일은
strict gate를 통과하지 않는다.

```powershell
python run_pipeline.py --run-id <고유한_run_id> --investable-market-file <ETF_or_TR_CSV>
```

## 지원 시장 선택

`run_pipeline.py`는 아래 세 시장을 같은 검증·게이트 체계로 실행한다.

| `--market` | 예측 타깃(가격지수) | 투자 가능 수익률 |
|---|---|---|
| `kospi200` | KOSPI200 `^KS200` | KODEX 200 `069500.KS` adjusted close |
| `sp500` | S&P 500 `^GSPC` | SPY adjusted close |
| `nasdaq100` | NASDAQ-100 `^NDX` | QQQ adjusted close |

시장 데이터 다운로드와 실행 예시는 다음과 같다.

```powershell
python download_market_data.py --market sp500
python run_pipeline.py --market sp500 --run-id <sp500_run_id>

python download_market_data.py --market nasdaq100
python run_pipeline.py --market nasdaq100 --run-id <nasdaq100_run_id>
```

세 시장의 완료된 run을 한 화면에서 비교하는 독립형 HTML은 다음처럼 생성한다. 결과물은
외부 JavaScript나 이미지 파일 없이 표와 SVG 차트를 HTML 안에 포함한다.

```powershell
python generate_multi_market_ews_report.py `
  --kospi-run runs\<kospi200_run_id> `
  --sp500-run runs\<sp500_run_id> `
  --nasdaq-run runs\<nasdaq100_run_id> `
  --research-dir runs\<kospi_fixed_pre2020_research_id> `
  --output reports\ews_multi_market_report.html
```

`--quick`은 긴 nested feature 재선택을 생략하는 코드 경로 점검용이다. 산출물 manifest에
`quick_smoke_test=true`가 기록되며, 이 결과를 연구 결론이나 배포 승인에 사용하지 않는다.

```powershell
python run_pipeline.py --market sp500 --quick --run-id <sp500_smoke_id>
python run_pipeline.py --market nasdaq100 --quick --run-id <nasdaq100_smoke_id>
```

해외 시장의 필수 코어는 해당 지수의 거래량 추세·실현변동성·하방변동성과 미국 term
spread다. 지수 자체의 6개월·12개월 모멘텀과 월말 종가의 10개월 이동평균 괴리는
별도 후보로 screen하지만, 불안정한 추세를 모델에 강제하지 않는다. 월말까지 확인된
값은 다음 달부터 실행하며 `optional_market_factor_screen.csv`에 채택 전 진단을 남긴다.
한국 종목패널 breadth는 선택 가능한 교차시장 설명변수로 남지만 해외 지수의 구성종목
지표라고 표시하지 않는다.

동적 주식 비중은 pre-holdout 구간에서 신호 gate와 포트폴리오 gate를 모두 통과할 때만
활성화된다. 어떤 tactical policy도 동일 익스포저 대비 fold Sharpe, active return,
거래비용 및 낙폭 기준을 통과하지 못하거나 신호 gate가 실패하면 `static_50_50`으로
자동 복귀한다. 모든 gate를 통과한 정책 중에서는 보수적인 25bp 거래비용 후 active
return을 우선하고 fold Sharpe와 방향 안정성을 보조 순위로 사용한다. historical
holdout은 이 선택을 바꿀 수 없고 실패 시에만 `static_50_50`으로 veto한다.
`smoothed_linear`는 3개월 causal EWMA로 EWS를 완화한 추가 후보이며, 이 정책 역시
동일한 pre-holdout gate를 통과해야 사용할 수 있다.

배분 연구 실행이 끝나면 `<배분_run_id>` 폴더에 다음 시각화가 자동 생성된다.

- `benchmark_allocation_dashboard.png`: 누적성과, 낙폭, 실행 투자비중, KOSPI200 대비 상대자산
- `benchmark_outperformance_diagnostics.png`: rolling·연도별·시장국면별 초과성과
- `benchmark_candidate_selection.png`: 2020년 3월 이전 데이터만 사용한 후보정책 비교
- `baseline_vs_current_diagnostics.png`: baseline과 현재 고정 사양의 동일기간 비교
- `model_comparison.png`: Logistic Champion과 SVM·MLP Challenger의 AUC·Sharpe·MDD 비교

`run-id`는 기존 디렉터리를 덮어쓰지 않는다. 중단된 실행도 manifest의 상태를 확인하고 새 ID로 다시 실행한다.

## 세 연구 트랙

- `original_reference`: KOSPI200 가격지수, exact 원자료, expanding logistic screening, 제한 완전조합, 선형 EWS 배분을 별도 저장한다.
- `robust_research`: release lag, 원천별 상위 3개와 그룹별 상위 15개 funnel, 그룹 다양성, purged nested outer OOS, 다섯 배분정책(선형·평활 선형·고정 구간·확장 백분위·정적 fallback), 비용·bootstrap·동일 익스포저 비교를 수행한다.
- `required_core`: Logistic champion은 거래대금/시가총액의 추세, 미국 term spread, KOSPI 종목 pairwise correlation, 1개월 수익률 왜도를 모든 outer fold에서 반드시 사용한다. 각 family 안의 변환만 해당 fold의 과거 OOS score로 고른다.
- `challengers`: SVM과 소형 MLP가 pre-2020에서 각자의 단일후보 순위와 조합을 다시 고른다. 기본 최종 모델은 Logistic이며 challenger의 historical holdout 결과는 진단에만 쓴다.
- `external_reference`: 사용자가 제공한 2025년 말~2026년 4월 화면의 변수 목록이다. 이 기간은 이미 관찰됐으므로 모델·변수·변환·배분정책 선택에 사용하지 않고 진단표에만 기록한다.

## KOSPI 종목패널 지표

`Data/MARKET/KOREA_STOCK_PRICE.csv`에서 KOSPI 종목을 추출하여 다음을 계산한다.

- 1개월 횡단면 수익률 왜도와 dispersion: 종목별 공식 일간 `ChangesRatio`를 월복리한 뒤 월별 1/99% winsorization
- 상승 종목 비율
- 거래량 12개월 평균 대비 비율
- 거래대금/시가총액과 거래량/상장주식수
- Advance/Decline ratio와 McClellan oscillator
- 월간 평균 pairwise correlation

원시 왜도·dispersion과 관측 종목 수는 품질진단 전용이며 모델 후보에서 제외된다. 현재 패널에는 역사적 KOSPI200 구성종목 membership가 없으므로 이 지표는 `KOSPI 상장종목 universe`의 직접 계산값이지 원본의 KOSPI200 구성종목 지표와 동일하다고 주장하지 않는다.

## 실전 차단 조건

현재 기본 설정은 연구용이다. `research_waiver` 때문에 `operational_gate=True`가
될 수 있지만 이는 연구 실행 허용일 뿐이다. 다음 조건이 모두 충족되기 전에는
`strict_operational_gate=False`, `deployment_eligible=False`가 정상이다.

1. 선택 Factor의 사람 경제성 검토를 `economic_review_registry.csv`에 holdout과 무관한 근거로 승인
2. 선택 거시계열과 포트폴리오 cash leg의 ALFRED 당시 vintage·release timing 검사 통과
3. KOSPI200 TR/NTR 또는 추종 ETF adjusted return 검사 통과
4. signal·portfolio gate 통과

준비된 freeze 사양은 다음처럼 검증한다.

```powershell
python shadow_monitor.py --run-dir runs\<고유한_run_id>
```

gate가 통과한 동결 사양에서만 신규 월 관측 JSON을 append할 수 있다. freeze hash 불일치, 동결 Factor 결측, 범위 위반, active drawdown 중단선은 기록을 거부한다.

### MLP 독립 검증과 research shadow

정상 실행은 MLP도 84개월 최소 학습창으로 별도 purged outer 검증한다.
KOSPI200 주모델은 원본 EWS 자료에서 고정한 거래대금/시가총액·미국 기간스프레드·
종목 pairwise correlation·1개월 수익률 왜도 4개 입력과 `cash_excess` 타깃을 Logistic에
사용한다. `cash_excess`는 3개월 뒤 지수수익률이 신호일 당시 관측 가능한 현금수익
장벽을 넘는지를 뜻한다. 변수·타깃·fixed-bin 정책은 pre-2020에서 잠그며, 주모델에도
historical holdout safety veto를 적용한다.

S&P 500 MLP는 2020년 이전 개발 구간에서 잠근 기업가치·10Y/3M 기간스프레드·
AUD/USD·비금융기업 이익 변동성 4개 입력과 3개월 양(+)의 수익률 타깃을 사용한다.
NASDAQ-100은 동일한 S&P 고정 MLP 사양을 전이하되, 고변동 기술주 시장의 꼬리위험을
직접 겨냥해 향후 3개월 월말 경로가 -5% 아래로 내려가지 않는지를 타깃으로 삼는다.
두 미국 시장 모두 같은 월별 expanding refit과 독립 signal·portfolio·holdout safety
gate를 사용한다.

2020년 이후 historical holdout은 변수·하이퍼파라미터·배분정책 선택이나 긍정적 승격에
사용하지 않는다. 다만 잠긴 tactical overlay가 동일 평균 익스포저보다 Sharpe 또는 CAGR이
낮거나 MDD가 3%p 넘게 나쁘면 `historical_holdout_safety_gate=False`로 fail-closed veto한다.
이 경우 pre-2020 게이트가 통과했더라도 실제 적용 정책과 frozen shadow 사양은
`static_50_50`으로 강제된다. holdout 통과만으로는 모델이 승격되지 않는다.

- `mlp_validation_gates.csv`: signal·portfolio·운영·capital-use 판정
- `mlp_outer_fold_signal_metrics.csv`: fold별 AUC·Rank IC·실패 이유
- `mlp_position_sizing_policy_gate.csv`: 비용·동일 익스포저 기준 배분 게이트
- `mlp_historical_holdout_confirmation.csv`: 잠긴 정책의 진단 성과와 safety veto 근거
- `latest_mlp_ews.json`: 최신 MLP 점수와 허용된 사용 범위
- `mlp_research_shadow_spec.json`: 동결된 MLP shadow 사양
- `mlp_frozen_score_history.csv`: freeze 시점까지 월별 점수·목표·실행 비중
- `mlp_research_shadow_ledger.csv`: MLP 신규 관측 append-only 장부

신호·포트폴리오·시점자료 검사가 통과해도 사람의 경제성 검토는 자동 승인하지 않는다.
`mlp_economic_review_checklist.csv`의 경제 경로, 예상 방향, 발표시차, 중복정보, 검토자와
검토시각을 사람이 채우고 모든 행을 `approved`로 바꾼 별도 CSV를 만든 뒤 다음 명령으로
registry에 반영한다. `suggested_*` 열은 검토를 돕는 초안일 뿐 승인 필드가 아니며,
미완성 또는 `pending` 항목은 명령 자체가 거부한다.

```powershell
python approve_economic_review.py `
  --completed-review runs\<검증_run_id>\mlp_economic_review_checklist_completed.csv
```

승인 후에도 동일한 전체 파이프라인을 새 run ID로 다시 실행해야 하며, 이 승인은 다른
게이트를 우회하지 않는다.

MLP가 승격 게이트를 통과하지 못해도 `research_shadow_only` 상태에서는 자본을 넣지
않는 모의 추적을 할 수 있다. 다음 명령은 동결 hash와 전용 ledger 경로를 확인한다.

```powershell
python shadow_monitor.py --run-dir runs\<고유한_run_id> --spec-name mlp_research_shadow_spec.json
```

신규 관측 JSON을 기록할 때도 같은 `--spec-name`을 지정한다. research shadow 사양에
`capital_authorized=True`가 설정되면 모니터가 즉시 거부하므로 이 경로를 실전 배포
우회로로 사용할 수 없다.

MLP를 월별로 추적할 때는 freeze run을 바꾸지 않는다. 새 월의 원자료로 별도 data run을
완료한 뒤, 아래 점수기로 freeze run의 변수·하이퍼파라미터·3개월 purge·random seed·배분
규칙만 적용한다. 이 명령은 변수 재선정이나 하이퍼파라미터 탐색을 하지 않는다.

```powershell
python score_frozen_mlp.py `
  --spec-run-dir runs\<freeze_run_id> `
  --data-run-dir runs\<new_data_run_id> `
  --asof-date <YYYY-MM-DD_월말> `
  --output runs\<new_data_run_id>\mlp_shadow_observation_<YYYY-MM-DD>.json
```

점수기는 전월 목표 비중을 당월 실행 비중으로 사용하고, 투자 가능 가격·전월 현금금리·
거래비용으로 strategy/same-exposure/active return까지 계산한다. freeze hash나 기준 점수이력
hash가 달라졌거나 직전 shadow 월이 빠졌거나 중복됐거나 미래 label cutoff를 넘으면 중단한다.
freeze 월에 실행하면 재현 진단만 반환하며 장부에는 append할 수 없다.

`packet_status=ready_for_shadow_append`인 신규 월 JSON만 다음처럼 freeze run의 장부에 기록한다.

```powershell
python shadow_monitor.py `
  --run-dir runs\<freeze_run_id> `
  --spec-name mlp_research_shadow_spec.json `
  --observation-json runs\<new_data_run_id>\mlp_shadow_observation_<YYYY-MM-DD>.json
```

`mlp_shadow_observation_template.json`은 비상 수동 입력용이다. `null`이 남아 있거나 active
return이 strategy return - same-exposure return과 일치하지 않으면 append가 거부된다.

`OPERATIONAL_GATE_PROFILE="research_waiver"`에서는 현재 실제로 남아 있는 위험만
`operational_risk_acceptance.csv`로 감사하고 `applied_to_current_run=True`로 표시한다.
이미 ALFRED 또는 ETF 검사를 통과한 위험은 waiver를 적용한 것으로 기록하지 않는다.
이 프로필은 operational gate만 통과시키며 사람의 경제성 승인을 획득했다고 주장하지 않는다.
waiver는 `deployment_eligible`을 True로 만들 수 없으며, signal·portfolio gate까지
통과한 경우에만 별도 `forward_shadow_eligible` 연구 상태를 허용한다.

KOSPI 초과 목적의 별도 배분 트랙은 pre-2020에서만 후보를 선택한다. 현재 선택 규칙은
3개월 KOSPI200 절대모멘텀이 양수면 주식 100%, 아니면 현금 100%이고 다음 달에 실행한다.
전체 역사에서의 초과성과는 과거 진단이며 미래 초과성과를 보장하지 않는다.

## 핵심 감사 파일

- `external_reference_coverage.csv`: 화면 변수의 계산 가능성·정확성·선택 금지 상태
- `external_reference_observed_window_values.csv`: 2025-12~2026-04 값, 진단 전용
- `market_return_role_registry.csv`: 신호 가격지수와 투자수익률 역할 분리
- `point_in_time_audit.csv`: release lag와 ALFRED vintage 상태
- `selected_source_point_in_time_audit.csv`: 실제 선택 Factor 원천만 대상으로 한 strict vintage 차단 근거
- `outer_fold_logistic_coefficients.csv`: fold별 표준화 Logistic 계수와 부호(인과 승인 아님)
- `factor_coefficient_sign_stability.csv`: 여러 fold에 반복 선택된 Factor의 계수 부호 일관성
- `required_family_coefficient_sign_stability.csv`: 변환이 fold마다 달라도 필수 4개 family별로 집계한 계수 부호 일관성(진단 전용)
- `outer_fold_signal_metrics.csv`, `signal_gate_diagnostics.csv`: fold coverage, AUC, Rank IC와 signal gate 산식
- `position_sizing_policy_gate.csv`: portfolio gate의 조건별 실제값·통과 여부·실패 조건
- `economic_review_checklist.csv`: 자동 승인하지 않은 사람 검토 대기 항목
- `deployment_gate_details.csv`, `deployment_blockers.csv`: strict 배포 게이트의 조건별 근거와 조치
- `data_freshness_schema_audit.csv`: 최신성·스키마 검사
- `cash_return_convention_sensitivity.csv`: 단순 연율/12와 월복리 민감도
- `effective_sample_size.csv`, `nonoverlapping_3m_signal_diagnostics.csv`: 3개월 중첩 target 진단
- `forward_shadow_spec.json`, `forward_shadow_ledger.csv`: 동결 hash와 append-only 모니터링

과거 호환 컬럼 `direction_stability`는 Factor 계수 부호 안정성이 아니라, 평가 가능한
outer fold 중 `AUC > 0.5`와 `Rank IC > 0`를 동시에 만족한 비율이다. 실제 계수 부호는
반드시 별도 coefficient 감사 파일에서 확인한다. `accuracy_lift_vs_naive_diagnostic_only`는
이름 그대로 참고 진단이며 deployment 판정식에 포함하지 않는다.

3개월 forward target에 24개월 outer fold를 쓰면 비중첩 결과가 약 8개뿐이므로,
현재 outer fold는 사전에 정한 최소 비중첩 결과 12개에 맞춘 36개월이다. 마지막 잔여
fold가 20개월보다 짧으면 방향 통과 비율에서 제외하되 파일에는 exclusion 사유를 남긴다.

## 2026-08-14 통합 MLP 사양

세 시장의 MLP는 같은 검증 순서(signal → portfolio → historical safety veto)를 사용하지만,
표본 크기와 사전 검증 결과에 따라 신경망의 역할을 다르게 고정한다.

- KOSPI200: 4-unit tanh MLP와 정규화 Logistic 백본을 함께 학습한다. 백본 확률이 65% 이상인데
  MLP가 50% 미만이면 공격적 신호를 50% 중립으로 낮추는 `risk_veto`만 허용한다. 목표는 향후
  3개월 KOSPI200 수익률이 신호일 현금 hurdle을 이기는지인 `cash_excess`, 비중 정책은
  `fixed_bin`이다.
- S&P 500: 8×4 tanh MLP, 향후 3개월 양(+) 수익 목표, `expanding_percentile` 비중 정책이다.
- NASDAQ-100: S&P 500에서 사전 고정한 8×4 MLP 입력·구조를 이전하고, 향후 3개월 경로가
  -5% drawdown을 피하는지 예측한다. 비중 정책은 `expanding_percentile`이다.

KOSPI `risk_veto`는 2020년 이후 holdout에서 고른 규칙이 아니다. 2020년 이전 동일 프로토콜에서
독립 MLP가 portfolio gate를 실패한 원인을 진단한 뒤, 공통 후보 규칙 중 signal과 portfolio gate를
모두 통과한 방어 규칙으로 고정했다. historical holdout은 실패 시 사용을 막는 safety veto로만 쓴다.

재현 명령:

```powershell
python run_pipeline.py --market kospi200 --run-id <new_kospi_run_id>
python run_pipeline.py --market sp500 --run-id <new_sp500_run_id>
python run_pipeline.py --market nasdaq100 --run-id <new_nasdaq_run_id>

python generate_ews_mlp_economic_report.py `
  --kospi-run runs\<new_kospi_run_id> `
  --sp500-run runs\<new_sp500_run_id> `
  --nasdaq-run runs\<new_nasdaq_run_id> `
  --baseline-run runs\ews_reproduce_20260814_v2 `
  --research-dir runs\guarded_mlp_cross_market_pre2020_20260814_v2 `
  --fairness-dir runs\universal_mlp_fairness_20260814_v1 `
  --output reports\EWS_MLP_economic_explainer_20260814.html
```

보고서는 기준 KOSPI 재현 실행, 시장별 특화 모델, 동일 사양 비교, leave-one-market-out을 서로 다른
주장으로 분리한다. 입력한 run의 실제 CSV·JSON에서 Factor funnel, class balance, 동기간 성과,
fold별 실패 사유, 2020년 이후 safety veto, strict operational blocker를 읽어 상세 HTML을 만든다.

현재 기준 검증 실행은 다음과 같다.

- `runs/ews_kospi200_mlp_risk_veto_20260814_v13`
- `runs/ews_sp500_mlp_unified_20260814_v9`
- `runs/ews_nasdaq100_mlp_unified_20260814_v9`

세 실행의 `code_files` manifest hash는 같아야 하며, `--quick` 실행은 최종 검증 근거로 쓰지 않는다.

### 동일 모델 범용성 공정 비교

위 세 실행은 시장별 목표·구조·비중 정책이 다르므로 “같은 모델이 세 시장에서 통과했다”는 근거가
아니다. 범용성은 다음 별도 트랙으로만 판단한다.

```powershell
python run_universal_mlp_fairness.py `
  --kospi-run runs\ews_kospi200_mlp_risk_veto_20260814_v13 `
  --sp500-run runs\ews_sp500_mlp_unified_20260814_v9 `
  --nasdaq-run runs\ews_nasdaq100_mlp_unified_20260814_v9 `
  --output-dir runs\universal_mlp_fairness_20260814_v1
```

공정 비교는 세 시장에 아래 항목을 전부 동일하게 고정한다.

- 정확히 같은 4개 Factor 컬럼
- `cash_excess` 3개월 목표
- 8×4 tanh MLP와 동일 hyperparameter
- 최소 84개월, 3개월 purge, 매월 expanding refit
- 동일한 2008-11~2020-03 outer-fold 달력
- `expanding_percentile` 비중 정책
- 10bp와 25bp 거래비용 gate

`same_spec`은 같은 사양을 시장별로 따로 학습한다. `leave_one_market_out`은 평가 대상 시장의 label을
완전히 제외하고 나머지 두 시장으로 학습한 MLP를 파라미터 변경 없이 이전한다. 두 트랙 모두
2020년 이전 signal·portfolio gate로 판단하며, 2020년 이후는 실패 시 중단하는 safety veto로만 쓴다.

2026-08-14 결과:

- 동일 사양 시장별 학습: 전체 gate 통과 `0/3`
- Leave-one-market-out: 전체 gate 통과 `1/3`(S&P 500만 통과)
- 결론: `universal_model_claim_supported=false`

따라서 현재 세 시장 성과는 “공통 검증체계를 사용한 시장별 특화 모델”로만 보고한다. 동일 모델의
범용성이나 국가 간 이식 가능성을 주장하지 않는다. 근거 파일은 다음과 같다.

- `runs/universal_mlp_fairness_20260814_v1/protocol.json`
- `runs/universal_mlp_fairness_20260814_v1/same_spec_summary.csv`
- `runs/universal_mlp_fairness_20260814_v1/leave_one_market_out_summary.csv`
- `runs/universal_mlp_fairness_20260814_v1/fairness_conclusion.json`
