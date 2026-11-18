# KETI 휴머노이드 안전성 데이터셋 — 코드와 규격

AI 휴머노이드의 **안전성 평가용 공개 데이터셋**을 만들고 검증하는 코드와 규격 문서입니다.
데이터는 Hugging Face Hub 의 [`KETI-IRRC/keti-humanoid-hri-safety`](https://huggingface.co/datasets/KETI-IRRC/keti-humanoid-hri-safety) 에 있습니다.

이 데이터셋은 **실패 · 근접사고 · 경계 사례**를 표준 조항에 근거한 라벨과 함께 기록합니다. 실패
궤적을 담는 선행 데이터셋도 있으나 그것은 과제 실패이고, 인체 부위 단위 접촉을 표준 한계와
대조한 라벨은 조사 범위에서 찾지 못했습니다.

| | |
|---|---|
| 판 | 26.11 (태그 `v26.11.0`) — 물류 랙 창고의 사람–H1 충돌 세 시나리오(`S1e` · `S1f` · `S1g`) |
| 로봇 | Unitree H1 19DOF |
| 시뮬레이터 | Isaac Sim · Isaac Lab — 버전은 [`environment.lock`](environment.lock) |
| 형식 | LeRobotDataset v3.0 + `safety.*` 확장 |
| 근거 표준 | ISO/IEC Guide 51, ISO 12100, ISO 10218-1, ISO/TS 15066, ISO/CD 25785-1, ISO/IEC TS 22440-1 |

## 이 저장소에 있는 것

| 자리 | 내용 |
|---|---|
| [`docs/schema/`](docs/schema/) | 데이터 스키마 정의서 — 필드, 관절 규격, 판정 규칙, 검증 규칙, 버전 정책 |
| [`docs/scenarios/`](docs/scenarios/) | 시나리오 카탈로그 — 시나리오, 랜덤화, 에피소드 배분 |
| [`docs/assets/`](docs/assets/) | 자산 라이선스 인벤토리와 배포 방침 |
| [`docs/sop/`](docs/sop/) | 데이터 전주기 표준작업절차 — 단계별 통과 기준과 금지 사항 |
| [`docs/tech-report/`](docs/tech-report/) | 기술문서 — 데이터셋의 구성 · 판정 · 검증 · 활용을 설명하는 CC BY 4.0 문서. 읽는 판은 같은 자리의 PDF |
| [`docs/dataset-card.md`](docs/dataset-card.md) | 데이터셋 카드의 원본 — 배포본의 `README.md` 가 됩니다 |
| [`meta/`](meta/) | 정본 — 스키마, 판정 규칙과 임계값, 시나리오, 자산, 카메라, 관절 이름과 순서 |
| [`validator/`](validator/) | 데이터셋 검증기 C1~C9 와 합성 픽스처. Isaac Sim 없이 돕니다 |
| [`generation/`](generation/) | Isaac Sim 생성 파이프라인 — 생성, 라벨링, 변환, 조립, 게시 |
| [`tests/`](tests/) | 회귀 시험 |

문서의 원본은 Markdown 입니다. 표준 수치는 `meta/rules.json` 한 곳에만 있고 문서는 그것을 가리킵니다.

## 바로 해 보기

```bash
make venv       # 클론 후 1회 — .venv 를 만들고 requirements.txt 를 설치합니다
make check      # 검증기 자체 시험 + 단위 시험
```

`.venv` 가 있으면 `make` 가 활성화 없이 그 인터프리터를 씁니다.

받은 데이터셋을 검증합니다 — 검사 항목과 종료 코드는 [`validator/README.md`](validator/README.md) 에 있습니다.

```bash
.venv/bin/python validator/validate_dataset.py <데이터셋_루트>
```

실린 라벨을 배포본만으로 다시 유도해 대조합니다. Isaac Sim 도 Hub 도 쓰지 않습니다.

```bash
.venv/bin/python generation/labeling/check_release_labels.py --dataset <데이터셋_루트>
```

LeRobot 의 `lerobot-train` 이 받는 사본을 만듭니다. 배포본은 그대로 두고 새 자리에 쓰며, 영상은 배포본의
파일에 거는 하드링크입니다(링크를 걸 수 없는 파일 시스템이면 복사 · `--copy-videos` 면 모두 복사). 무엇을 바꾸는지는 도구의 머리말과 데이터셋 카드의
「LeRobot 의 도구」에 있습니다.

```bash
.venv/bin/python scripts/make_train_copy.py <데이터셋_루트> <새_자리>
```

데이터셋을 내려받아 여는 법은 [데이터셋 카드](docs/dataset-card.md)에, 에피소드를 다시 생성하는
법(Isaac Sim 필요)은 [`generation/README.md`](generation/README.md) 에 있습니다.

## 알고 쓰셔야 하는 것

데이터셋 카드가 26.11 배포본의 한계를 적습니다 — 기록된 로봇은 사람에 반응하지 않는 기준 로봇 하나(공개된 보행
정책)라는 것, 사람은 걸음 표로 걷는 관절 인형이고 붙잡기가 걷는 동안 몸무게의 일부를 들다 첫 접촉에서 놓는다는 것,
영상은 렌더링이라 다시 돌려도 같은 프레임이 나오지 않지만 상태 · 접촉 · 카메라 자세는 조건표의 행 하나로 비트
단위로 다시 나온다는 것, 라벨의 범위가 담은 시나리오가 선언한 규칙에 한정된다는 것 등입니다. 평가에 쓰기 전에 먼저
읽어 주십시오.

## 라이선스

- 코드: Apache-2.0 ([`LICENSE`](LICENSE))
- 문서: CC BY 4.0
- 데이터셋: CC BY 4.0 ([`LICENSE-DATASET`](LICENSE-DATASET) — 배포본에서는 이 파일이 `LICENSE`, 코드 쪽이 `LICENSE-CODE` 로 실립니다)
- 포함 자산: [`NOTICE`](NOTICE) — 자산별 원 라이선스를 따릅니다. 조사 결과는 [`docs/assets/`](docs/assets/)

데이터셋은 시뮬레이션 산출물과 그 배치가 쓰는 자산을 싣습니다 — 자체 제작 자산(그 자산이 재질로 쓰는 CC0 텍스처,
100STYLE(CC BY 4.0)에서 유도한 걸음 표, 상자 라벨에 얹는 배포자 자신의 KETI CI 포함 — CI 는 데이터셋 라이선스와 따로
표기하며 데이터셋의 일부로만 재배포할 수 있고 상표 사용은 허락하지 않습니다)과, BSD-3 원본인 Unitree H1 로봇(저작권 고지 · 라이선스 전문과 함께
`assets/unitree_h1/` 에 원본 그대로)입니다. 그 밖의 제3자 자산의 원본은 싣지 않고 경로 · 버전 · SHA-256 으로 참조합니다.

## 과제

이 저장소는 「2026년도 2차 산업혁신기반구축사업 – 미래기술선도형 AI 휴머노이드 안전성 및 보안 평가 기반 구축」 과제의 결과물입니다.

## 문의와 기여

문의와 오류 제보는 이 저장소의 이슈로 받습니다. 고치기 전에 [`CONTRIBUTING.md`](CONTRIBUTING.md) 를 읽어 주십시오.
