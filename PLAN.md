# EWS 신호·비중 변환 및 검증 개선 Plan

> 2026-08-14 구현 상태: 아래의 초기 SVM 중심 기준선 설명은 역사적 기록이다.
> 현재 사양은 원천별 상위 3개 → 그룹별 상위 15개 → 그룹당 최대 2개·최소
> 3개 그룹 → 최종 4~7개이며, Logistic을 Champion으로 사용한다. SVM과
> 소형 MLP는 pre-2020에서 각자의 후보와 조합을 다시 고르는 진단용
> Challenger다. 2020-04 이후는 반복 관찰된 historical research holdout이다.

작성일: 2026-08-12  
상태: **계획만 작성 — 구현 코드 변경 전**

## 1. 결론

피드백의 핵심 판단에 동의한다.

- 현재 SVM은 `AUC 0.731`, `Rank IC 0.347`로 **순위 신호는 유의미한 후보**다.
- `Accuracy 65.8%`는 다수 클래스 기준선 `65.8%`와 같으므로 분류 성과로 인정하지 않는다.
- 현재 선형 비중 전략의 Sharpe `0.923`은 동일 평균 익스포저 정적 전략의 `0.905`보다 `0.019` 높을 뿐이어서, 아직 market-timing alpha로 판정하기 어렵다.
- 따라서 다음 구현의 1순위는 Factor나 모델 수를 늘리는 일이 아니라 **SVM 점수를 포트폴리오 비중으로 변환하는 방식의 검증**이다.

추가로 확인한 원본 기술자료에 따르면 `EWS 20 → KOSPI ETF 20%`처럼 신호값을 비중에 직접 쓰는 방식이 원본의 기본 사용법이다. 따라서 현재 선형 방식은 폐기 대상이 아니라 **원본 충실도 기준선**으로 보존한다. 구간형과 percentile은 원본 복제가 아니라 검증 강화 실험으로 명확히 구분한다.

다만 피드백에 나온 세 방식을 현재 Test에서 비교해 가장 좋은 것을 채택하면 안 된다. `2020-04~2026-04` Test 결과를 이미 확인했으므로, 이 기간은 이후 의사결정에 대해 더 이상 완전한 untouched Test가 아니다. 비중 함수 선택은 Test 이전 데이터의 purged walk-forward 검증에서 끝내고, 현재 Test는 연구용 historical holdout으로만 보고해야 한다. 최종 확인은 동결된 사양의 forward shadow 결과로 수행한다.

## 2. 현재 기준선

현재 `results/` 산출물을 구현 전 기준선으로 보존한다.

| 항목 | Logistic | SVM |
|---|---:|---:|
| Test AUC | 0.4392 | 0.7308 |
| Accuracy | 41.10% | 65.75% |
| Naive accuracy | 65.75% | 65.75% |
| Rank IC | -0.0402 | 0.3467 |
| Pearson IC | 0.1291 | 0.2462 |
| Rolling IC 양수 비율 | 39.47% | 100.00% |
| Dynamic Sharpe | 0.8940 | 0.9234 |
| Same-exposure Sharpe | 0.9048 | 0.9048 |

SVM 포트폴리오 기준:

- 평가기간: `2020-05-31~2026-05-31`, 73개월
- Dynamic: CAGR `20.95%`, Sharpe `0.923`, MDD `-21.60%`
- Same exposure 57.8%: CAGR `19.82%`, Sharpe `0.905`, MDD `-21.79%`
- 현재 실전 상태: `연구용: OOS 진단 실패로 실전 사용 보류`

이 숫자는 기준선 재현 테스트용 baseline이며, 새 방식 채택 기준 그 자체로 사용하지 않는다.

## 3. 이번 개선의 범위

### 포함

1. 원본 충실도 기준선과 검증 강화 개선 트랙의 분리
2. 현재 선형 비중, 고정 구간형, expanding percentile 비중 비교
3. 모든 방식의 동일 기간·동일 수익률·동일 비용 조건 평가
4. Same Exposure 대비 active 성과와 불확실성 평가
5. 분류 정확도 중심 deployment gate를 ranking/portfolio 중심 gate로 재설계
6. 비중 방식이 확정된 뒤 연속 수익률 회귀 모델을 별도 연구 트랙으로 검토
7. 아래 “추가 수정 사항”의 데이터·검증·재현성 보강

### 제외

- 현재 historical Test 결과를 보고 SVM 점수를 `1-p`로 뒤집는 작업
- Test에서 가장 좋은 threshold나 percentile bucket을 탐색하는 작업
- SVM `C`, `gamma`의 대규모 재탐색
- Factor 후보를 무작정 더 늘리는 작업
- 통계적 근거 없이 실전 사용 상태로 승격하는 작업

## 4. 최우선 원칙: Test 재과최적화 방지

### 4.1 기간의 역할

- `1996-03~2020-03`: 모델·비중 함수 연구와 선택에 사용할 수 있는 구간
- `2020-04~2026-04`: 이미 결과를 확인한 **research holdout**
- 사양 동결 이후 신규 월: 진짜 forward shadow 확인 구간

현재 historical holdout에서 세 비중 함수를 모두 계산할 수는 있지만, 그 결과는 설명용으로만 저장한다. 그 기간의 최고 Sharpe를 근거로 함수를 선택하거나 threshold를 수정하지 않는다.

### 4.2 내부 검증 방식

연구 구간에서는 3개월 purge를 유지한 expanding outer walk-forward를 사용한다.

- 최소 학습기간: 현재 설정과 동일하게 Logistic 60개월, SVM 84개월
- Outer validation window: 24개월을 기본값으로 사전 고정
- 각 fold에서 feature screening, 조합 선택, 모델 학습을 해당 시점 이전 데이터만으로 다시 수행
- 각 fold의 예측은 완전 OOS score로 저장
- 비중 함수는 이 OOS score만 사용해 비교
- 한 fold의 결과로 threshold를 바꾸지 않고 모든 fold의 집계로 한 번만 선택

“현재 선택된 두 Factor를 고정한 간단 비교”도 진단용으로 만들 수 있지만, 최종 결정은 전체 feature-selection 과정을 fold 안에서 다시 수행한 결과를 따른다.

## 5. 비중 변환 설계

비중 변환 로직을 모델 예측과 백테스트에서 분리한다. 향후 구현 시 `src/position_sizing.py` 같은 독립 모듈에 순수 함수로 둔다.

### 5.1 방식 A — 현재 선형 비중

원본 기술자료와 현재 시스템의 기준선이며 현재 결과가 정확히 재현되어야 한다.

```text
weight = clip(EWS / 100, 20%, 80%)
```

### 5.2 방식 B — 고정 구간형

피드백의 사양을 그대로 첫 고정 후보로 사용한다.

| EWS | KOSPI 비중 |
|---:|---:|
| `< 35` | 20% |
| `35 ≤ EWS < 50` | 40% |
| `50 ≤ EWS < 65` | 60% |
| `EWS ≥ 65` | 80% |

경계 `35`, `50`, `65`의 포함 방향을 단위 테스트로 고정한다. Historical Test 성과를 보고 이 경계를 미세 조정하지 않는다.

### 5.3 방식 C — Expanding percentile

SVM 확률의 절대 calibration보다 상대 순위를 이용한다.

| 과거 OOS EWS percentile | KOSPI 비중 |
|---:|---:|
| 하위 20% | 20% |
| 20~40% | 35% |
| 40~60% | 50% |
| 60~80% | 65% |
| 상위 20% | 80% |

필수 조건:

- 시점 `t`의 percentile은 반드시 `t` 이전 OOS score만으로 계산한다.
- 현재 점수를 포함해 quantile을 계산하지 않는다.
- 최소 과거 score 36개월을 기본값으로 사전 고정한다.
- 36개월 미만은 `NaN`으로 두고 성과 비교를 시작하지 않는다.
- tie, 상수 score, 결측값 처리 규칙을 명시하고 테스트한다.
- percentile 방식 때문에 평가 시작일이 늦어지므로 세 방식은 모두 동일한 교집합 기간에서 비교한다.

### 5.4 실행 시차와 비용

모든 방식에서 현재의 미래참조 방지 규칙을 유지한다.

```text
t월 말 score → t월 말 목표 비중 결정 → t+1월 수익률에 적용
```

- 최소/최대 비중: 20%/80%
- 기본 거래비용: 10 bps × 월간 절대 turnover
- 민감도: 0, 10, 25, 50 bps
- 첫 거래는 현금 100%에서 시작한 매수로 계산
- 각 방식마다 자신의 평균 주식비중으로 Same Exposure benchmark를 다시 계산

## 6. 비교 지표와 선택 기준

### 6.1 신호 지표

- AUC
- Rank IC와 Pearson IC
- Brier score
- calibration slope/intercept와 reliability plot
- Risk-On 비율과 naive accuracy
- rolling IC의 실제 관측 개수

Accuracy는 계속 출력하되 비중 방식 선택의 primary metric으로 사용하지 않는다.

### 6.2 포트폴리오 지표

절대 성과보다 같은 평균 익스포저 대비 active 성과를 우선한다.

- CAGR, 변동성, Sharpe, Sortino, MDD, Calmar
- 평균 주식비중
- 연환산 turnover와 총 거래비용
- Same Exposure 대비 active CAGR
- Sharpe 차이
- 평균 active return, tracking error, information ratio
- 12/24/36개월 rolling active 성과
- 상승장·하락장·고변동성장 구간별 성과

### 6.3 불확실성

73개월과 3개월 중첩 target은 독립 표본으로 볼 수 없다.

- 3~6개월 block bootstrap으로 Sharpe 차이와 평균 active return의 신뢰구간 계산
- Rolling IC 100%는 서로 겹치는 36개월 window들의 비율임을 명시
- 가능한 경우 비중첩 3개월 관측과 경기 국면별 결과도 병기
- 세 비중 방식 비교에 따른 multiple-comparison 위험을 보고

### 6.4 사전 선택 규칙

Research 구간의 outer folds에서 다음을 만족하는 방식을 우선한다.

1. Same Exposure 대비 median Sharpe 개선 `≥ 0.10`
2. Outer fold의 최소 2/3에서 평균 active return이 양수
3. 10 bps와 25 bps 비용 모두에서 active 성과 양수
4. Same Exposure 대비 MDD가 3%p 이상 악화되지 않음
5. 한 시기 성과에만 의존하지 않고 fold별 순위가 안정적

`+0.10`은 연구 후보를 거르는 실무 기준이지 통계적 유의성을 대신하지 않는다. 어떤 방식도 통과하지 못하면 기존 선형 방식을 “승자”로 선언하지 않고 전체 전략을 연구용으로 유지한다.

## 7. Deployment gate 재설계

현재 코드는 `accuracy_lift_vs_naive > 0`을 실전 후보 조건으로 사용한다. SVM은 ranking signal을 활용하는 전략이므로 이 조건은 목적과 맞지 않는다.

향후 gate는 다음처럼 분리한다.

### Signal gate

- AUC > 0.5
- Rank IC > 0
- inner/outer fold에서 신호 방향이 안정적
- calibration 붕괴 여부 보고

### Portfolio gate

- Same Exposure 대비 active 성과가 사전 선택 기준 통과
- 비용 민감도 통과
- bootstrap 신뢰구간과 worst-fold 결과 공개

### Operational gate

- 데이터 최신성·결측·schema 검사 통과
- score와 비중이 범위 안에 있음
- 사양과 모델 hash가 동결된 버전과 일치

Accuracy와 naive accuracy는 경고용 진단으로 유지한다. Gate를 바꾸더라도 historical 결과만으로 바로 실전 사용 가능 상태로 바꾸지 않고 forward shadow 기간을 요구한다.

### Score와 비중의 의미 분리

원본 선형 방식에서는 `EWS 63`과 `주식 63%`가 같지만 percentile이나 구간형에서는 달라진다. 따라서 결과와 화면에서 다음을 별도 필드로 유지한다.

- `raw_ews`: 모델이 산출한 0~100 시장 신호
- `allocation_policy`: linear/fixed-bin/expanding-percentile
- `target_stock_weight`: 정책이 변환한 목표 비중
- `executed_stock_weight`: 한 달 lag와 제약을 적용한 실제 비중

Percentile 결과를 다시 EWS라고 부르지 않는다. 그래야 모델 신호 품질과 사용자의 위험성향·배분정책을 독립적으로 평가할 수 있다.

## 8. 회귀 모델 연구 — 비중 함수 검증 후 별도 단계

분류 target은 `+0.1%`와 `+30%`, `-0.1%`와 `-25%`를 같은 class로 처리하므로 정보 손실이 있다. 비중 변환 실험을 완료한 뒤 아래를 별도 실험으로 진행한다.

### Target

```text
Forward 3M KOSPI return
```

### 모델 우선순위

1. ElasticNet
2. 선형 SVR 또는 제한된 RBF SVR
3. Gradient Boosting/Random Forest는 표본 수 대비 복잡도가 높으므로 마지막 탐색 후보

### 규칙

- target winsorization/scaling은 train fold에서만 적합
- 현재와 동일한 3개월 purge와 expanding outer folds 사용
- RMSE만 보지 않고 Rank IC, Pearson IC, 방향 hit rate, 실제 portfolio active 성과 평가
- classification과 regression을 historical Test에서 반복 비교해 승자를 고르지 않음
- 회귀 모델 추가는 별도의 multiple-comparison 항목으로 기록

## 9. 추가로 수정할 부분

### P0 — 구현 전에 해결

1. **KOSPI200으로 연구 기준을 우선 통일**  
   원본 자료의 후보지표, 조합 DB 화면, ETF 설명은 KOSPI200을 중심으로 한다. 현재 `^KS200`도 이 의도와 가깝다. 따라서 원본 충실도 트랙의 신호 target/benchmark 명칭은 KOSPI200으로 통일하는 것이 우선안이다. 지수 가격수익률과 실제 ETF 수익률은 분리한다. 모델 신호 연구에는 검증된 KOSPI200 지수 계열을 쓰고, 투자 백테스트에는 배당·보수·tracking difference·거래비용을 반영할 수 있는 총수익지수 또는 실제 추종상품 adjusted return을 별도로 사용한다. KRX는 KOSPI200을 ETP·인덱스펀드 등의 대표 기초지수로 설명하고 KOSPI200 TR/NTR 지수 산출도 명시한다. [KRX 지수사업](https://open.krx.co.kr/contents/OPN/02/02010100/OPN02010100.jsp), [KRX ETF 개요](https://global.krx.co.kr/contents/GLB/02/0201/0201030100/GLB0201030100.jsp)

2. **현재 Test의 상태 변경**  
   이미 본 `2020~2026` 구간을 untouched Test라고 부르지 않고 `research_holdout`으로 기록한다. 새 사양은 freeze 이후 forward shadow로 최종 확인한다.

3. **재현 가능한 데이터 다운로드**  
   yfinance 다중 헤더가 밀려 저장된 전력이 있으므로 다운로드 직후 OHLCV schema, 행 수, 날짜, 고가≥저가, 거래량 존재 여부를 검증한다. `yfinance`를 requirements에 고정하고 source ticker와 다운로드 시각을 manifest에 저장한다.

4. **실행 산출물 격리**  
   현재 `results/`는 부분 실행 시 과거 파일과 새 파일이 섞일 수 있다. 향후 `results/<run_id>/`에 원자적으로 저장하고 config, 데이터 기간, feature 목록, git/hash, 모델 seed를 manifest로 남긴다.

### P1 — 비중 실험과 함께 해결

5. **Final SVM과 feature-selection 목적 불일치**  
   현재 단일지표·greedy 조합 선택은 Logistic walk-forward AUC로 수행하고 최종 모델은 SVM이다. 최종 검증에서는 전체 선택 과정을 model-aware inner fold 안에서 수행하거나, 모델과 무관한 Rank IC 기반 사전선택 규칙을 사전 고정한다.

6. **Unknown 경제 그룹 정리**  
   `USEPUINDXD`, `WLEMUINDXD` 등이 `unknown`으로 자동 등록되어 그룹별 최대 1개 제한을 우회한다. 각 series를 명시적 `risk` 또는 적절한 경제 그룹에 배정하고 metadata 검사를 추가한다.

7. **그룹 제한의 의미 명확화**  
   “그룹별 최대 1개”는 다양성을 보장하지 않는다. 현재처럼 inflation/liquidity 두 개만 선택될 수 있다. 약한 그룹을 억지로 넣지는 않되, 선택된 그룹 수와 누락 그룹을 결과에 출력한다. 필요하면 그룹별 대표를 먼저 뽑는 2단계 구조를 별도 비교한다.

8. **Config 중복 제거**  
   `MIN_STOCK_WEIGHT`, `MAX_STOCK_WEIGHT`, `TRANSACTION_COST_BPS`가 중복 정의돼 있다. 하나의 authoritative 설정만 남기고 실행 manifest에 직렬화한다.

9. **Calibration 진단 추가**  
   calibrated SVM probability를 선형 비중에 쓰려면 Brier뿐 아니라 calibration slope/intercept와 reliability를 확인해야 한다. Percentile 방식이 선택돼도 drift 감시용으로 유지한다.

10. **원본의 비선형 단일지표 분석 복원**  
    원본은 단일지표를 ML 모델에 입력해 비선형 관계까지 확인한다고 설명한다. 현재 single-factor screening은 Logistic으로 고정돼 있어 이 부분을 재현하지 못한다. 여러 알고리즘을 동시에 시험하지 말고, 사전 고정된 저복잡도 비선형 단일변수 모델 한 개를 추가하거나 최종 SVM과 동일한 model-aware OOS screening을 inner fold 안에서 수행한다. 선형 Logistic 결과는 해석용 진단으로 남긴다.

11. **원본 지표와 proxy의 트랙 분리**  
    `1 Month return Skew in korea stock market universe`, dispersion/correlation, Advance/Decline, McClellan은 단일 지수 OHLCV가 아니라 구성종목 패널이 필요한 원본 지표다. 현재 만든 일별 지수 왜도와 risk-appetite proxy는 `proxy` 트랙에만 두고 원본 재현 점수표에서는 제외한다. 정확한 지표가 확보되면 proxy와 나란히 비교하되 자동 대체하지 않는다.

12. **경제적 검토를 사전 등록된 절차로 추가**  
    원본은 성능 상위 조합을 사람이 검토하고 경제적 의미가 없으면 다음 후보로 넘긴다. 이 절차는 필요하지만 holdout을 본 뒤 개입하면 cherry-picking이 된다. Validation까지만 본 상태에서 `economic_channel`, 예상 신호 방향, 발표 lag, 중복 정보, 거부 사유를 기록하고, 승인된 사양을 freeze한 뒤 research holdout으로 보낸다.

### P2 — 구조적 신뢰도 개선

13. **2,573개 후보의 다중검정 완화**  
    그룹별/원자료별 후보 수를 먼저 제한하고, 상관관계가 높은 변형들의 family 단위 선택을 검토한다. Validation을 수백 번 재사용하는 현재 greedy 탐색의 optimistic bias를 outer fold 재실행으로 측정한다.

14. **Point-in-time 거시 데이터**  
    현재 availability lag는 발표 지연만 근사하며 FRED 수정치를 완전히 막지 못한다. 중요한 series는 ALFRED vintage 또는 당시 공개값으로 교체하는 장기 계획을 둔다.

15. **Cash return convention 점검**  
    단순 `연율/12` proxy 대신 사용 중인 3개월 금리의 quote convention과 월복리 변환을 문서화하고, 결과 차이를 민감도 분석한다.

16. **통계 독립성 보고**  
    3개월 forward target의 월별 관측은 중첩된다. 일반 월수와 함께 effective sample, block-bootstrap 구간, 비중첩 결과를 출력한다.

17. **실전 모니터링 규칙**  
    forward shadow에서 score 분포 drift, feature 결측, calibration drift, turnover 급증, active drawdown 한도를 감시하고 중단 조건을 사전 정의한다.

18. **HP filter 미래참조 차단**  
    원본 Factor Factory에는 HP-filter 변형이 있다. 전체 표본에 일반 양방향 HP filter를 한 번 적용하면 미래 값이 과거 trend에 들어갈 수 있으므로 그대로 추가하지 않는다. 도입하려면 매 시점 과거 데이터만으로 다시 적합하는 one-sided/real-time 방식의 truncate 불변성 테스트를 통과해야 한다. 비용과 불안정성이 크면 제외한다.

19. **완전조합 탐색을 제한된 후보군에서만 수행**  
    원본은 최종 후보의 모든 조합을 자동 평가한다고 설명하지만 현재 2,573개 전체에 적용할 수 없고 과최적화도 심해진다. 원자료·변형 family·경제 그룹을 거쳐 8~12개 정도로 사전 축소한 작은 후보군에서만 조합 크기를 제한해 exhaustive search하고, 그 밖에는 현재 greedy/beam search를 nested validation 안에서 사용한다.

20. **최소 24개월 설정을 그대로 복제하지 않음**  
    원본의 최소 24개월은 동작 시작 조건이지 현재 표본에서 안전한 통계 기준으로 보지 않는다. 현재 60/84개월 최소 학습기간을 기본으로 유지하고, 24개월은 원본 재현 민감도 결과로만 보고한다.

## 10. 예정 산출물

구현 단계에서 다음을 생성한다.

- `position_sizing_comparison.csv`: 세 방식의 동일기간 성과
- `position_sizing_fold_results.csv`: outer fold별 active 성과
- `position_sizing_monthly.csv`: score, percentile, 목표/실제 비중, turnover, 비용
- `position_sizing_bootstrap.csv`: Sharpe 차이와 active return 신뢰구간
- `calibration_diagnostics.csv` 및 reliability plot
- `experiment_manifest.json`: 기간, 설정, 데이터·모델 hash, 선택 결과
- `research_holdout_report.csv`: 선택에 사용하지 않는 2020~2026 진단
- 단위/통합 테스트

## 11. 테스트 계획

### 단위 테스트

- 선형 방식이 현재 비중과 정확히 일치
- 구간형 경계 `35/50/65` 처리
- percentile이 미래 score를 전혀 참조하지 않음
- 과거 데이터를 특정 시점에서 잘라도 이전 percentile/비중이 동일
- 최소 history, tie, 상수 score, NaN 처리
- 모든 비중이 20~80% 범위
- score `t`가 수익률 `t+1`에만 적용
- 거래비용과 첫 매수 turnover 계산

### 통합 테스트

- 모든 전략의 `Start`, `End`, `Months`가 동일
- 각 방식의 Same Exposure 평균 비중이 해당 동적 전략과 일치
- 같은 입력·seed에서 동일 산출물 생성
- Research 단계에서 holdout 날짜를 읽으려 하면 실패
- 부분 실행 실패 시 기존 성공 run 산출물을 덮어쓰지 않음
- 현재 선형 baseline의 주요 수치가 허용 오차 안에서 재현

## 12. 구현 순서

1. 현재 결과와 설정을 immutable baseline run으로 보존
2. 연구 target은 KOSPI200, 투자성과는 총수익/추종상품 return으로 역할을 확정
3. 기간 역할, exact/proxy 구분, run manifest 도입
4. 원본 지표 정의와 아직 모르는 DB 컬럼·목적함수를 문서화
5. 변형 family registry와 causal/no-lookahead 검사를 먼저 작성
6. 원본 충실도용 비선형 단일지표 OOS screening 설계
7. 작은 최종 후보군의 제한된 조합 탐색과 경제적 검토 로그 도입
8. 비중 변환 모듈과 세 방식 단위 테스트 작성
9. 백테스트가 raw EWS와 외부 목표 비중 series를 구분해 받도록 인터페이스 분리
10. Pre-2020 purged outer walk-forward OOS score 생성
11. 세 방식의 동일기간·동일비용 평가와 bootstrap 수행
12. 사전 기준으로 한 방식을 선택하고 사양 freeze
13. Historical research holdout은 한 번 보고하되 선택에는 사용하지 않음
14. Deployment gate를 signal/portfolio/operational gate로 분리
15. Forward shadow 시작
16. 비중 방식이 충분히 검증된 뒤 회귀 모델 연구 착수

## 13. 완료 판정

다음 조건을 모두 충족해야 이번 개선 구현이 완료된 것으로 본다.

- 세 비중 함수가 명세와 단위 테스트를 통과
- 원본 선형 기준선과 개선 실험 결과가 별도 트랙으로 저장됨
- raw EWS와 allocation policy/target/executed weight가 분리됨
- percentile no-lookahead 검사가 truncate 재계산으로 증명됨
- 세 방식과 모든 benchmark가 완전 동일 기간으로 평가됨
- Same Exposure 대비 active 성과와 비용·불확실성이 모두 보고됨
- 함수 선택 과정이 2020~2026 결과를 사용하지 않았음이 manifest로 증명됨
- 선택 기준을 통과하지 못하면 연구용 상태를 유지
- regression은 sizing 실험과 분리된 run으로만 수행
- KOSPI200 지수 신호와 실제 투자수익률의 역할이 분리됨
- 원본 exact 지표와 단일지수 proxy가 혼합되지 않음
- HP filter를 포함한 모든 변형이 truncate no-lookahead 검사를 통과하거나 제외됨
- 데이터 source, feature selection, unknown group, config 중복 문제가 정리됨
- 사람의 경제적 검토가 holdout 이전에 사유와 함께 기록됨
- forward shadow용 freeze 사양과 모니터링 규칙이 생성됨

## 14. 피드백 반영표

| 피드백 | Plan 반영 |
|---|---|
| 분류보다 ranking signal이 의미 있음 | AUC/IC와 portfolio gate를 primary로 전환 |
| 현재 Sharpe 개선이 너무 작음 | Same Exposure 대비 active 성과·신뢰구간으로 판정 |
| EWS 숫자를 그대로 비중으로 쓰지 말 것 | 선형/구간형/percentile 3방식 비교 |
| 35/50/65 구간형 | 경계와 비중을 고정 사양으로 테스트 |
| expanding percentile | 과거 OOS score만 사용하는 no-lookahead 사양 |
| 회귀 target 검토 | sizing 검증 후 ElasticNet/SVR 중심 별도 단계 |
| Factor 추가보다 비중 함수 우선 | 구현 순서 4~8에서 sizing을 최우선 처리 |
| Sharpe +0.1~0.2 수준 필요 | 연구 선택 gate를 median ΔSharpe ≥ 0.10으로 사전 고정 |

## 15. 원본 기술자료와 현재 구현의 차이

### 원본에서 확인된 사항

| 원본 설계 | 현재 구현 | 판단 |
|---|---|---|
| EWS 수치를 KOSPI ETF 비중으로 직접 사용 | `score/100`, 20~80% clip | 원본 기준선으로 유지 |
| 사용자의 위험성향에 따라 최소/최대·threshold 변경 | 최소 20%, 최대 80% 고정 | 모델 신호와 allocation policy로 분리 |
| Factor Factory → 통계 → ML → 전망/백테스트 | 동일한 큰 흐름 | 유지 |
| 원지표당 약 70개 변형, 전체 1만개 이상 | FRED 원지표당 171개, 총 2,573개 | 개수 확대보다 family 통제 필요 |
| 이동평균 변화와 HP-filter 변형 | MA/EWMA/z/vol/slope, HP 없음 | causal HP만 선택적 검토 |
| 단일변수 ML로 비선형 관계 검증 | Logistic 단일지표 screening | 저복잡도 비선형 OOS screening 보완 |
| 최소 24개월 후 rolling/expanding 예측 | 60/84개월, 과거 전체 expanding 학습 | 현재 보수적 최소기간 유지 |
| 3개월 전망 | 3개월 forward target | 유지 |
| 최종 단일지표 후보의 조합 자동 탐색 | 상위 40개 greedy forward | 작은 후보군 exhaustive + nested search |
| 사람이 경제적 의미 검토 | 그룹 최대 1개 제한만 있음 | holdout 전 review log 추가 |
| 조합과 성과를 DB에 저장 | CSV/JSON을 동일 results 폴더에 저장 | run별 immutable manifest로 개선 |

### 아직 원본 자료만으로 확정할 수 없는 사항

- 원본 EWS의 정확한 target 정의가 `3개월 상승확률`인지, 기대수익률인지, 다른 목적함수인지
- 조합 DB의 `mp`, `k200`, `btm` 컬럼 정의와 최종 ranking 공식
- `REC UP RATIO`, `RISK APPETITE INDEX` 등 내부 지표의 정확한 산식
- HP-filter의 당시 실시간 계산 방식과 revision 처리
- 백테스트가 가격지수, 총수익지수, 실제 ETF 중 무엇을 사용했는지
- 거래비용, 보수, tracking difference, 현금금리 처리 방식

따라서 현재 시스템을 “원본 완전 복제”라고 부르지 않는다. 관련 발표 페이지나 지표 정의가 추가로 확보되면 manifest의 `spec_source`와 함께 사양을 갱신한다.

## 16. 최적 권고 구조

### Track O — Original-reference

원본 철학을 재현하는 기준선이다.

- KOSPI200 중심 target
- 원본과 정의가 일치하는 exact 지표만 사용
- 사전 등록한 변형 family
- 한 개의 고정된 저복잡도 비선형 단일지표 ML
- expanding 학습, 3개월 전망
- 작은 후보군의 제한된 조합 탐색
- raw EWS를 비중으로 직접 쓰는 linear allocation
- validation 단계의 경제적 의미 검토

### Track R — Robust-research

과최적화와 포트폴리오 변환 문제를 줄이는 개선 트랙이다.

- 경제 그룹·원자료·변형 family 제한
- purged nested/outer walk-forward
- exact 지표와 proxy 지표의 별도 실험
- linear/fixed-bin/expanding-percentile allocation 비교
- Same Exposure active 성과, 비용, bootstrap 평가
- model-aware feature selection과 deployment gate
- 이후 별도 regression 연구

### 최종 판단 규칙

1. Track O는 항상 원본 기준선으로 남긴다.
2. Track R은 Pre-2020 outer validation에서 사전 기준을 통과해야만 후보가 된다.
3. 이미 본 2020~2026 결과로 O/R 또는 allocation 방식을 선택하지 않는다.
4. raw signal의 AUC/IC와 allocation의 active Sharpe를 별개로 판정한다.
5. 실전 후보 승격은 freeze 이후 forward shadow만으로 결정한다.

### 원본 후보지표 확보 우선순위

**Tier 1 — 현재 데이터로 정의를 맞출 수 있는 지표**

- Consumer Sentiment, CPI YoY, M2 YoY, WTI
- US Term Spread, US High Yield Spread, Recession Probability, VIX
- Equity-market/Economic-policy uncertainty 계열

**Tier 2 — 외부 시계열을 추가하면 계산 가능한 지표**

- KOSPI200−World spread, US Dollar Index, AUD/CHF
- KOSPI200 구성종목 패널 기반 1개월 횡단면 왜도
- 구성종목 dispersion/correlation, Advance/Decline, McClellan
- EPS growth, EPR, BPR, EPS estimate dispersion

**Tier 3 — 원본 내부 산식이 필요한 지표**

- REC UP RATIO US/KR
- 원본 RISK APPETITE INDEX
- 내부 뉴스·투자·심리 지표

Tier 3는 산식 확보 전 임의 proxy로 대체하지 않는다. Tier 2의 구성종목 지표도 현재 단일지수 proxy와 같은 지표로 취급하지 않는다.
