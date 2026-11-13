# Capstone — xArm 회차별 하중 이상 판정

한 번의 No.3 운반에서 측정한 전류·전압, ADXL345 가속도, 6개 서보 위치로 정상 여부와 하중 유형을 판정합니다. 출력은 `normal`, `A121`, `B82`, `C182`입니다. 사이클 종료 후 판정하며, 미래 고장 시점이나 잔여 수명을 예측하지 않습니다.

**v1 결과:** 사용자가 학습·평가를 실행했고, 선택 모델은 24특징 로지스틱(C=10, 임계값 0.09)이었습니다. 시험 하중 60회 중 59회를 탐지했지만 정상 오탐은 13/30회였습니다. 균형 점수는 0.775로 기존 로지스틱 0.40 기준의 0.850보다 낮아, 전체 성능 개선을 확인하지 못했습니다.

[아키텍처와 기존 방식과의 차이](docs/ARCHITECTURE.md) · [결과 분석과 그림](experiments/v1/ANALYSIS_KO.md) · [전체 v1 결과](experiments/v1/) · [구현 검증](docs/VALIDATION.json)

기존 데이터·특징을 활용해 검증과 선택 절차를 새로 구성한 실험이며, 새로운 학습 알고리즘의 발명을 주장하지 않습니다. `prepare`, `evaluate`, `predict`와 패키지 import는 학습하지 않으며, 사용자가 `train`을 실행할 때만 모델을 적합합니다.

## 현재 컴퓨터에서 학습 실행

코드와 준비된 데이터는 `/media/robin/shMin/Capstone`에 있습니다. 이 드라이브는 exFAT이므로 가상환경의 심볼릭 링크를 지원하는 홈 디렉터리에 Python 환경을 구성했습니다.

```bash
cd /media/robin/shMin/Capstone
source /home/robin/.cache/capstone-venv/bin/activate
python -m capstone train --data data/prepared --out runs/v2
```

학습은 기본 24개/확장 27개 특징과 로지스틱 3개/부스팅 2개 설정의 총 10개 후보를 날짜별 3분할로 비교합니다. 개발 검증에서 선택한 모델을 개발 자료 전체로 적합하고, 비교용 로지스틱도 저장합니다. 최종 시험 자료는 학습 명령에서 읽지 않습니다.

학습이 끝난 뒤 다음 명령으로 고정 시험셋을 평가합니다. 기존 V4 PCA와 정상 확장 로지스틱의 두 임계값도 같은 회차에 적용합니다.

```bash
python -m capstone evaluate \
  --model runs/v2/model.joblib \
  --data data/prepared \
  --out runs/v2/evaluation
```

특징 CSV에 대한 예측만 필요할 때는 다음 명령을 사용합니다.

```bash
python -m capstone predict \
  --model runs/v2/model.joblib \
  --features data/prepared/test.csv \
  --out runs/v2/test_predictions.csv
```

기존 출력 폴더나 예측 파일은 덮어쓰지 않습니다. v1은 실행을 마쳤으므로 위 명령은 새 실행 `runs/v2`를 사용합니다. 이후에도 새 경로를 지정합니다. 새 학습 결과는 기존 v1과 별도의 실험으로 기록합니다.

## 새 환경에서 준비

Python 3.13.15와 `requirements.lock`을 사용합니다. 다음 예시는 Linux와 uv 기준입니다.

```bash
cd /media/robin/shMin/Capstone
uv venv --python 3.13.15 "$HOME/.cache/capstone-venv"
uv pip install --python "$HOME/.cache/capstone-venv/bin/python" -r requirements.lock
source "$HOME/.cache/capstone-venv/bin/activate"
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
  https://github.com/wplso/xArm-model-training.git data/upstream
python -m capstone prepare --source data/upstream --out data/prepared
```

현재 컴퓨터에는 참고 저장소가 `/home/robin/xArm-model-training`에도 있습니다. 이를 사용해 준비를 다시 검증하려면 새 출력 경로를 지정합니다.

```bash
python -m capstone prepare \
  --source /home/robin/xArm-model-training \
  --out data/prepared_recheck
```

`prepare`는 고정 Git 커밋에서 필요한 LFS 파일만 받아 크기와 SHA-256을 확인합니다. 검증된 다운로드를 `data/source`에 캐시하며, 같은 캐시를 사용하는 재실행에서는 다시 다운로드하지 않습니다. 원본·비교 자료는 약 620 MB입니다. 전체 Git LFS 저장소를 내려받을 필요는 없습니다.

## 데이터와 검증 범위

출처는 [wplso/xArm-model-training](https://github.com/wplso/xArm-model-training)의 `bolt-known-load`, 커밋 `2fdaa693e3aaafe3a221f5a775a12bc340f4d562`입니다. `V4_V51_보완원시`와 `볼트_하중_보완원시`의 원시 목록에서 볼트 장착·품질 승인 회차를 선택하고 `study_only` 자료를 제외했습니다. 원본의 `role`은 보존하고 Capstone의 `split`을 별도로 추가합니다.

| 날짜 | 정상 | A121 | B82 | C182 | 역할 |
|---|---:|---:|---:|---:|---|
| 2026-11-06 | 479 | 30 | 30 | 30 | 개발 |
| 2026-11-10 | 80 | 0 | 0 | 0 | 개발: 평상·재가동 정상 |
| 2026-11-11 | 330 | 30 | 30 | 30 | 개발 |
| 2026-11-12 | 30 | 0 | 30 | 30 | 고정 시험 |
| 합계 | 919 | 60 | 90 | 90 | 1,159회 / 42세션 |

개발 1,069회는 하루 전체를 검증에 두고 다른 두 날짜로 학습하는 세 분할을 사용합니다. 같은 날짜의 연속 회차, 세션, 연결 그룹을 섞지 않습니다. 대치·표준화도 각 학습 분할에서만 적합합니다.

11월 12일의 90회는 모델·임계값 선택에서 분리합니다. 이 자료는 과거 연구에서 이미 분석됐으므로 결과는 **회고 평가**입니다. A121이 시험 자료에 없어 A121의 성능은 개발 날짜 간 검증 결과만 제공합니다. 개발 후보·임계값 선택에 사용한 OOF 점수도 최종 독립 성능으로 해석하지 않습니다.

## 특징과 모델

기본 24개 특징은 원본 V4와 같은 순서·수식입니다.

- 전류 평균·RMS·표준편차·절댓값 P95, 전압 평균·표준편차: 6개
- 가속도 각 축 표준편차, 평균 제거 벡터 RMS·피크: 5개
- 서보 ID1–6 위치 범위·표준편차: 12개
- 그리퍼 ID1의 동작 3–5초 위치 IQR: 1개

확장 27개는 여기에 S3 `[3,4)`, S4 `[4,5)`, S5 `[5,6.5)`초의 평균 전류를 더합니다. 시작은 No.3 명령 쓰기 완료 시각입니다. 전체 통계는 No.3 종료 경계를 포함하고 원복을 제외하며, 구간 통계는 오른쪽 경계를 제외합니다. 보간하지 않고 원시 표본을 사용합니다.

표준편차는 모집단 정의, P95는 nearest-rank, IQR은 선형 분위수입니다. IQR은 서로 다른 위치 표본 4개와 최대 간격 0.75초를 요구합니다. S3/S4는 8개, S5는 12개 이상의 표본과 최대 간격 0.25초, 연속 시퀀스를 요구합니다.

후보와 seed `20261113`은 [실험 설정](capstone/experiment.json)에 고정했습니다. 두 모델 모두 클래스 균형 가중치를 사용합니다.

| 모델 | 설정 |
|---|---|
| 로지스틱 회귀 | C=0.1/1/10, lbfgs, 최대 반복 2,000, 학습 중앙값 대치와 StandardScaler |
| HistGradientBoosting | 잎 수 7/15, 학습률 0.05, 200회, 최소 잎 표본 10, L2=1, 자동 조기 종료 해제, 학습 중앙값 대치 |

이상 점수는 `1 - P(normal)`입니다. 점수가 임계값을 **초과**하면 경보가 나고, A121/B82/C182 중 가장 높은 점수를 유형으로 출력합니다. 이 값들은 실제 고장 발생 확률로 보정된 수치가 아닙니다.

임계값은 0–1을 0.01 간격으로 비교합니다. 자기 날짜를 학습하지 않은 예측(OOF)을 사용하여 다음 목적함수를 최대화합니다.

```text
J = 0.5 × 날짜별 정상 판정률 평균
  + 0.5 × (A121/B82/C182 각각의 날짜별 탐지율 평균)의 평균
```

동점이면 낮은 오탐률 → 로지스틱 → 기본 24개 → 더 강한 규제 → 높은 임계값 순서로 선택합니다. 시험셋에는 A가 없으므로 평가 J의 하중 평균은 B와 C만 사용하며, 개발 J와 직접 같은 구성의 점수로 비교하지 않습니다.

## 입력·출력 계약

`predict`의 CSV는 고유하고 비어 있지 않은 `cycle_id`와 모델 특징 열을 포함합니다. 열 순서는 달라도 됩니다. 라벨·날짜·질량·세션 정보는 예측 입력에 사용하지 않습니다.

출력은 다음 열을 포함합니다.

```text
cycle_id, predicted_label, anomaly_score, threshold, alarm,
quality_status, quality_reason, model_version,
p_normal, p_A121, p_B82, p_C182
```

ID1 IQR과 확장 구간 전류의 허용된 결측만 학습 중앙값으로 대치합니다. 필수 특징 누락, 무한값, 숫자가 아닌 값, 원시 품질 실패는 `invalid_data`로 표시합니다. 이 경우 점수·클래스·경보는 비워 두며 정상으로 간주하지 않습니다. 중복 ID와 잘못된 파일 스키마는 명령 오류로 처리합니다.

모델 번들에는 전처리·분류기, 특징 순서, 임계값, 설정, 학습 회차·날짜·그룹, 데이터·코드 해시를 저장합니다. 평가에는 신뢰할 수 있는 이 프로젝트와 고정 출처의 joblib 번들을 사용합니다.

## 결과 파일

| 위치 | 내용 |
|---|---|
| `data/prepared/` | 원본 목록, 특징, 개발/시험 CSV, 품질, 분할, 특징 재현 오차, 출처 해시 |
| `runs/v1/model.joblib` | 개발 검증으로 선택한 모델 |
| `runs/v1/baseline.joblib` | 같은 분할·임계값 절차의 기본 24특징 C=1 로지스틱 |
| `runs/v1/oof_predictions.csv` | 후보별 날짜 제외 예측·점수·선택 임계값의 판정 |
| `runs/v1/candidates.csv` | 후보별 최적 개발 점수·임계값 |
| `runs/v1/threshold_sweep.csv` | 모든 후보의 101개 임계값 비교 |
| `runs/v1/metadata.json` | 선택 근거, 제외 수, 실행 환경, 입력·코드·모델 해시 |
| `runs/v1/evaluation/REPORT_KO.md` | 최종 회고 평가 요약 |
| `runs/v1/evaluation/`의 CSV/JSON/PNG | 회차별 예측, 정상 오탐·유형별 탐지·AUROC/AP·Macro-F1, 날짜·세션별 분모, 혼동행렬과 그림 |

평가는 선택 모델, 동일 조건 로지스틱, 기존 V4 PCA, 기존 정상 확장 로지스틱의 저장 임계값과 0.40을 비교합니다. 원본 로지스틱의 90회 점수와 두 임계값의 경보도 저장 기록과 대조합니다. 입력이 판정 불가이면 제외 분모와 전체 대비 검출 비율을 함께 기록합니다.

## 학습하지 않는 테스트

```bash
python -m unittest discover -s tests -v
```

테스트에서는 실제 모델의 `fit` 호출을 차단합니다. 인공 추론 객체로 예측·저장·CLI·평가 보고서를 검증하고, 준비된 자료가 있으면 원본 특징과 기존 동결 모델의 점수 재현도 확인합니다. 모델 학습과 학습 후 실제 성능 검증은 위의 명시적 `train` / `evaluate` 실행으로 진행합니다.

코드·설정·검증 요약은 Git으로 관리합니다. 대용량 원시 데이터와 개인 실행 산출물인 `data/`, `runs/`는 기본적으로 제외했습니다. 검토한 v1의 모델, metadata, config, 예측, 분할표, 평가 그림과 분석은 `experiments/v1/`에 한 묶음으로 공개했습니다.

학습을 다시 실행하지 않고 기존 결과를 보존하고 분석 그림을 생성하려면 다음 명령을 사용합니다. 출력 스냅샷 경로는 새 경로여야 합니다. 원래 사용자가 생성한 평가 디렉터리 이름은 `evaluatio`입니다.

```bash
python scripts/archive_run.py \
  --run runs/v1 --evaluation runs/v1/evaluatio \
  --prepared data/prepared --out experiments/v1_recheck
```

아키텍처 SVG/PNG는 Graphviz로 다시 렌더링할 수 있습니다.

```bash
dot -Tsvg docs/architecture.dot -o docs/architecture.svg
dot -Tpng -Gdpi=140 docs/architecture.dot -o docs/architecture.png
```
