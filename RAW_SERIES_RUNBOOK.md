# 70개 원자료 확장 실행

## 정식 연구 실행

```powershell
$env:PYTHONIOENCODING='utf-8'
python download_raw_series.py --workers 8
python build_market_breadth.py
python run_pipeline.py --run-id ews_raw70_full_YYYYMMDD
```

각 `run-id`는 새 이름을 사용한다. 정식 실행은 6개 purged outer fold마다
원천 선별과 조합 탐색을 다시 수행한다.

현재 사전 고정 선발 순서는 `9,933개 → 원천별 상위 3개 → 그룹별 상위
15개 → 상관 제거 → 조합 4~7개`다. 최종 조합은 동일 원천 변환의 중복을
허용하지만, 같은 경제 그룹은 최대 2개이고 최소 3개 그룹을 포함해야 한다.
Logistic이 배포 후보 Champion이며 SVM과 소형 MLP는 pre-2020에서 각자
후보 순위와 조합을 다시 고르는 Challenger다.

## 빠른 연결 확인

```powershell
$env:PYTHONIOENCODING='utf-8'
python run_pipeline.py --quick --run-id ews_raw70_smoke_YYYYMMDD
```

`--quick`은 nested feature reselection을 생략하므로 정식 연구 결과로 사용하지
않는다. `--allow-partial-raw-universe`도 데이터 연결 디버깅 용도로만 사용한다.

## 필수 감사 산출물

- `raw_series_coverage.csv`: 70개 원천의 실제 가용성 및 연구구간 적격 여부
- `raw_series_group_summary.csv`: 7개 경제 바스켓별 구성/가용/적격 개수
- `factor_universe_summary.csv`: 70 → 1만여 Factor → 선별 단계별 개수
- `top_ranked_candidates.csv`: 단일 walk-forward 점수 상위 후보
- `raw_series_funnel_candidates.csv`: 원천별 상위 3개 변환 후보
- `group_balanced_candidates.csv`: 경제 그룹별 상위 15개 후보
- `bounded_exhaustive_candidates.csv`: 상관 제거 후 순위 기준 조합 탐색 후보
- `selected_feature_groups.csv`: Logistic Champion의 최종 4~7개 Factor와 경제그룹
- `svm_selected_features.json`, `mlp_selected_features.json`: 모델별 독립 조합
- `model_comparison.csv`, `performance_comparison.csv`: 동일기간 세 모델 진단
- `model_comparison.png`: AUC·Sharpe·MDD 비교 차트
- `exact_indicator_gap_audit.csv`: 유료·시점자료 부재로 선택이 금지된 정확 지표
- `deployment_gates.csv`: signal·portfolio·research operation·strict deployment 분리 결과
- `deployment_gate_details.csv`, `deployment_blockers.csv`: 조건별 실제값, 기준, 실패 이유와 조치
- `outer_fold_signal_metrics.csv`: 모든 예정 fold의 coverage·AUC·Rank IC
- `outer_fold_logistic_coefficients.csv`: fold별 선택 Factor의 표준화 Logistic 계수 방향
- `selected_source_point_in_time_audit.csv`: 선택 원천의 release-lag·ALFRED vintage 상태

정확한 EPS revision, KOSPI200 EPR/PBR, KOSPI-WORLD total-return spread는
`exact_indicator_gap_registry.csv`의 요구조건을 충족하는 데이터가 공급되기
전까지 대체지표와 동일하다고 간주하지 않는다.

2020-04 이후 구간은 연구 과정에서 반복 관찰했으므로 `untouched test`가
아니라 `historical research holdout`이다. 모델 비교 결과는 진단용이며,
최종 일반화 주장은 `forward_shadow_spec.json`을 동결한 뒤 새로 들어오는
월별 자료로만 평가한다.
