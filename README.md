# EWSProject

거시경제·시장 데이터를 이용해 주식시장의 위험 상태를 추정하는 **연구용 Early Warning System(EWS)** 입니다. KOSPI200, S&P 500, NASDAQ-100을 지원하며 다음 작업을 한 번의 파이프라인에서 수행합니다.

- FRED/ALFRED 및 시장 데이터 통합
- 시점 일치(point-in-time)·발표 지연·데이터 신선도 감사
- Logistic Champion과 SVM·소형 MLP Challenger 검증
- purged walk-forward 및 nested out-of-sample 평가
- EWS 점수, 주식/현금 목표 비중, 백테스트 생성
- 신호·포트폴리오·운영·배포 게이트 판정

> 이 프로젝트의 출력은 연구 결과이며 투자 권유가 아닙니다. `operational_gate=True`는 연구 운영을 허용할 수 있다는 뜻일 뿐입니다. 실제 자본 사용 가능 여부는 반드시 `strict_operational_gate`와 `deployment_eligible`을 별도로 확인해야 합니다.

## 지원 시장

| `--market` | 예측 대상 가격지수 | 투자 가능 수익률 |
|---|---|---|
| `kospi200` | KOSPI200 (`^KS200`) | marcap KOSPI top-200 price proxy |
| `sp500` | S&P 500 (`^GSPC`) | SPY adjusted close |
| `nasdaq100` | NASDAQ-100 (`^NDX`) | QQQ adjusted close |

KOSPI200이 기본값입니다.

## 1. 설치

프로젝트 루트에서 실행해야 합니다. 경로 계산이 현재 작업 디렉터리를 기준으로 이루어집니다.

Python 3.10 이상이 필요하며 Python 3.11 사용을 권장합니다. Windows PowerShell 기준 설치 예시는 다음과 같습니다.

```powershell
cd D:\Programming\python_example\EWS_Project

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONIOENCODING = 'utf-8'
```

주요 의존성은 `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyarrow`, `matplotlib`, `yfinance`입니다.

## 2. 데이터 준비

`Data/`는 Git에 포함되지 않습니다. 새로 받은 저장소에서는 아래 데이터를 준비해야 합니다.

### 2.1 FRED 원자료

`raw_series_catalog.csv`에서 활성화된 FRED 시계열을 내려받습니다. API 키는 필요하지 않습니다.

```powershell
python download_raw_series.py --workers 8
```

기존 파일을 다시 내려받으려면 `--refresh`를 추가합니다. 실행 보고서는 `Data/DERIVED/raw_series_download_report.csv`에 저장됩니다.

### 2.2 ALFRED 시점자료

과거 각 월말에 실제로 관측 가능했던 vintage를 내려받습니다.

```powershell
python download_alfred_vintages.py
```

이 작업은 요청 수가 많아 오래 걸릴 수 있습니다. 필요하면 `--start`, `--end`, `--workers`, `--request-interval`을 조정할 수 있습니다. ALFRED 자료가 없더라도 연구 waiver 아래에서 일부 실행은 가능하지만, point-in-time 운영 게이트와 strict deployment 판정에는 필요합니다.

### 2.3 시장 가격과 투자 가능 자산

분석할 시장마다 가격지수 OHLCV와 배당·분할 조정 가격을 함께 내려받습니다.

```powershell
python download_market_data.py --market kospi200
python download_market_data.py --market sp500
python download_market_data.py --market nasdaq100
```

한 시장만 분석한다면 해당 명령만 실행하면 됩니다. 기본 저장 위치는 다음과 같습니다.

| 시장 | 신호/타깃 데이터 | 투자 가능 수익률 데이터 |
|---|---|---|
| KOSPI200 | `Data/MARKET/KOSPI.csv` | `Data/DERIVED/marcap_kospi200_proxy.csv` |
| S&P 500 | `Data/MARKET/SP500.csv` | `Data/MARKET/SPY_adjusted.csv` |
| NASDAQ-100 | `Data/MARKET/NASDAQ100.csv` | `Data/MARKET/QQQ_adjusted.csv` |

각 파일 옆에는 다운로드 범위, 원천, SHA-256 등이 기록된 `*.metadata.json`이 생성됩니다.

### 2.4 한국 종목 패널과 market breadth

`download_market_data.py`는 한국 개별 종목 패널을 내려받지 않습니다. 다음 파일을 별도로 준비해야 합니다.

```text
Data/MARKET/KOREA_STOCK_PRICE.csv
```

필수 열은 아래와 같습니다.

```text
Code,Close,ChangesRatio,Volume,Amount,Marcap,Stocks,Market,observation_date
```

- `Market` 값 중 `KOSPI` 행을 사용합니다.
- `observation_date`는 파싱 가능한 날짜여야 합니다.
- 대용량 파일을 청크 단위로 처리하므로 날짜 순으로 정렬된 원본을 권장합니다.
- 이 패널은 역사적 KOSPI200 구성종목이 아니라 제공된 KOSPI 상장종목 universe로 해석됩니다.

패널을 준비한 뒤 breadth 캐시를 생성합니다.

```powershell
python build_market_breadth.py
```

결과는 `Data/DERIVED/korea_market_breadth.parquet`와 대응 metadata에 저장됩니다. 원본 패널이 변경되면 반드시 이 명령을 다시 실행해야 합니다. 이 캐시는 세 시장 파이프라인이 공통으로 사용합니다.

## 3. 빠른 동작 확인

먼저 `--quick`으로 데이터 연결과 전체 코드 경로를 확인합니다.

```powershell
python run_pipeline.py `
  --market kospi200 `
  --quick `
  --run-id ews_kospi200_smoke_YYYYMMDD
```

`--quick`은 반복 nested feature 탐색을 생략합니다. 결과 manifest에 `quick_smoke_test=true`가 기록되며, 이 결과를 연구 결론이나 배포 승인에 사용하면 안 됩니다.

원자료가 50개 미만인 상태에서 연결만 시험해야 할 때는 다음 옵션을 사용할 수 있습니다.

```powershell
python run_pipeline.py `
  --quick `
  --allow-partial-raw-universe `
  --run-id ews_partial_smoke_YYYYMMDD
```

`--allow-partial-raw-universe` 역시 스모크 테스트 전용입니다.

## 4. 정식 실행

`--quick` 없이 고유한 run ID로 실행합니다.

```powershell
python run_pipeline.py `
  --market kospi200 `
  --run-id ews_kospi200_full_YYYYMMDD
```

다른 시장은 `--market`만 변경합니다.

```powershell
python run_pipeline.py --market sp500 --run-id ews_sp500_full_YYYYMMDD
python run_pipeline.py --market nasdaq100 --run-id ews_nasdaq100_full_YYYYMMDD
```

정식 실행은 각 outer fold에서 선별과 조합 탐색을 반복하므로 시간이 오래 걸릴 수 있습니다. run ID를 생략하면 `ews_YYYYMMDD_HHMMSS` 형식으로 자동 생성됩니다. 기존 `runs/<run-id>`는 덮어쓰지 않으므로 중단 후 재실행할 때도 새 ID를 사용해야 합니다.

### 사용자 지정 시장 파일

신호 가격지수 파일을 바꾸려면 OHLCV CSV와 metadata를 함께 지정합니다.

```powershell
python run_pipeline.py `
  --market kospi200 `
  --market-file Data\MARKET\my_kospi200.csv `
  --market-metadata-file Data\MARKET\my_kospi200.metadata.json `
  --run-id ews_custom_signal_YYYYMMDD
```

투자 가능 수익률 파일을 바꾸려면 날짜 열과 `adjusted_close`, `adj_close`, `Adj Close`, `total_return_index` 중 하나가 있는 CSV를 사용합니다.

```powershell
python run_pipeline.py `
  --market kospi200 `
  --investable-market-file Data\MARKET\my_total_return.csv `
  --run-id ews_custom_return_YYYYMMDD
```

KOSPI200 기본 백테스트는 `Data/MARKET/KOREA_STOCK_PRICE.csv`의 marcap
패널에서 각 월말 KOSPI 보통주 시가총액 상위 200종목을 고른 뒤 다음 달에
적용하는 가격수익 프록시를 사용합니다. 공식 역사적 KOSPI200 편입 이력이
없는 근사치이므로 공식 지수와 동일하다고 간주하지 않으며 strict
investable-return gate도 통과하지 않습니다. 캐시는 다음 명령으로 갱신합니다.

```powershell
python build_marcap_kospi200.py
```

분류 AUC와 IC는 3개월 후 정답이 완성된 달까지만 계산합니다. 포트폴리오
백테스트는 이후 score-only 신호도 당시 이용 가능한 정보만으로 생성하여 최신
완료 월까지 계산합니다.

단순 종가(`close`)만 있는 파일은 strict investable-return gate를 통과하지 않습니다.

## 5. 결과 확인

모든 결과는 `runs/<run-id>/`에 저장됩니다. 우선 아래 파일을 확인하면 됩니다.

| 파일 | 내용 |
|---|---|
| `latest_ews.json` | 최신 Logistic EWS, 상태, 목표/실행 비중, 배포 상태 |
| `latest_mlp_ews.json` | 최신 MLP Challenger 점수와 검증 상태 |
| `ews_dashboard.png` | EWS·시장·성과 종합 대시보드 |
| `current_allocation.png` | 최신 주식/현금 배분 |
| `model_comparison.csv`, `model_comparison.png` | Logistic·SVM·MLP 비교 |
| `backtest.csv`, `performance_comparison.csv` | 월별 백테스트와 성과 요약 |
| `selected_features.json` | Logistic Champion의 최종 Factor |
| `deployment_gates.csv` | 신호·포트폴리오·운영·strict deployment 판정 |
| `deployment_blockers.csv` | 배포 차단 사유와 필요한 조치 |
| `experiment_manifest.json` | 코드·데이터 hash, 패키지 버전, 설정, 실행 상태 |

`latest_ews.json`의 주요 필드는 다음과 같습니다.

- `ews`: 0~100 범위의 최신 위험선호 점수
- `signal_state`: `Risk-Off`(<35), `Neutral`(35 이상 65 미만), `Risk-On`(65 이상)
- `target_stock_weight`: 모델이 제안한 다음 실행 주식 비중
- `executed_stock_weight`: gate와 fallback을 적용한 실제 연구상 비중
- `deployment_eligible`: strict 운영 조건까지 충족했는지 여부
- `forward_shadow_eligible`: 신규 월별 관측을 shadow로 추적할 수 있는지 여부

점수 하나만 보지 말고 `deployment_gates.csv`, `deployment_blockers.csv`, `experiment_manifest.json`을 함께 확인하십시오.

## 6. HTML 보고서 만들기

단일 실행 보고서:

```powershell
python generate_ews_html_report.py `
  --run-dir runs\ews_kospi200_full_YYYYMMDD `
  --output reports\ews_kospi200_full_YYYYMMDD.html
```

세 시장 비교 보고서:

```powershell
python generate_multi_market_ews_report.py `
  --kospi-run runs\ews_kospi200_full_YYYYMMDD `
  --sp500-run runs\ews_sp500_full_YYYYMMDD `
  --nasdaq-run runs\ews_nasdaq100_full_YYYYMMDD `
  --output reports\ews_multi_market_YYYYMMDD.html
```

보고서는 외부 JavaScript나 별도 이미지 없이 표와 SVG를 포함한 독립형 HTML로 생성됩니다.

## 7. 테스트

```powershell
python -m unittest discover -s tests -v
```

CLI 옵션은 각 스크립트의 `--help`로 확인할 수 있습니다.

```powershell
python run_pipeline.py --help
python download_market_data.py --help
```

## 8. 추가 연구·운영 도구

| 작업 | 명령 |
|---|---|
| 회귀 연구 | `python run_regression_research.py --classification-run runs\<run-id> --run-id <new-id>` |
| benchmark 배분 연구 | `python run_benchmark_allocation_research.py --parent-run runs\<run-id> --run-id <new-id>` |
| 세 시장 동일 MLP 사양 공정 비교 | `python run_universal_mlp_fairness.py --kospi-run ... --sp500-run ... --nasdaq-run ... --output-dir runs\<new-id>` |
| 사람 경제성 검토 등록 | `python approve_economic_review.py --completed-review <completed.csv>` |
| 동결 MLP 재점수화 | `python score_frozen_mlp.py --spec-run-dir runs\<freeze-id> --data-run-dir runs\<data-id>` |
| forward shadow 검사 | `python shadow_monitor.py --run-dir runs\<freeze-id>` |

경제성 검토는 프로그램이 자동 승인하지 않습니다. 생성된 체크리스트를 사람이 작성한 뒤 별도 파일로 등록해야 합니다. 동결 사양의 월별 추적과 shadow ledger append 절차는 [RUNBOOK.md](RUNBOOK.md)를 따르십시오.

## 9. 자주 발생하는 문제

### `FileExistsError` 또는 run 디렉터리 생성 실패

run은 불변 산출물로 취급됩니다. 기존 ID를 재사용하지 말고 새 `--run-id`를 지정하십시오.

### `Market-breadth cache missing`

`Data/MARKET/KOREA_STOCK_PRICE.csv`를 확인한 뒤 `python build_market_breadth.py`를 실행하십시오.

### `Korean stock panel changed after the breadth cache was built`

종목 패널이 바뀌었으므로 breadth 캐시를 다시 생성해야 합니다.

### 원자료 개수가 부족하다는 오류

`python download_raw_series.py --workers 8`을 다시 실행하고 `Data/DERIVED/raw_series_download_report.csv`의 `failed` 항목을 확인하십시오. 부분 universe 허용 옵션은 스모크 테스트에서만 사용합니다.

### 한글 로그가 깨짐

PowerShell에서 다음을 설정한 뒤 다시 실행하십시오.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
```

### Yahoo Finance/FRED 다운로드 실패

네트워크 연결, 프록시/방화벽, ticker 응답 여부를 확인한 뒤 다시 시도하십시오. FRED 다운로드는 실패한 시계열을 보고서에 남기며 하나라도 실패하면 비정상 종료합니다.

## 프로젝트 구조

```text
EWS_Project/
├─ src/                         # 데이터·Factor·모델·검증·백테스트 모듈
├─ tests/                       # 단위 및 재현성 테스트
├─ Data/                        # 원자료와 파생 캐시(Git 제외)
├─ runs/                        # 불변 실행별 산출물(Git 제외)
├─ reports/                     # HTML 보고서(Git 제외)
├─ raw_series_catalog.csv       # FRED/파생 원자료 레지스트리
├─ run_pipeline.py              # 메인 EWS 파이프라인
└─ requirements.txt
```

더 자세한 연구 설계, gate 정의, MLP shadow 운영은 [RUNBOOK.md](RUNBOOK.md), 원자료 확장 절차는 [RAW_SERIES_RUNBOOK.md](RAW_SERIES_RUNBOOK.md)를 참고하십시오.
