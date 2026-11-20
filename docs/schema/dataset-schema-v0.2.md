# AI 휴머노이드 안전성 데이터셋 — 표준 포맷 및 메타데이터 규격서 v0.2

> 대상: 시뮬레이션 데이터(2026) + 이후의 실장비 데이터

---

## 1. 목적과 범위

AI 휴머노이드의 **안전성 평가용** 데이터를 담는 포맷을 정의합니다.
**위험 상황·실패·근접사고**를 기록하고 그 판정 근거를 표준에 연결하는 것이 목적입니다.
성공 궤적 학습용 포맷과 갈라지는 지점은 과제 수행의 성패가 아니라
**인체 부위 단위 접촉이 표준 한계를 넘었는가**를 담는다는 것입니다.

| 항목 | 범위 |
|---|---|
| 로봇 | 이족 휴머노이드 (지금 Unitree H1 · 같은 규격으로 G1 29DOF 와 실장비까지) |
| 데이터원 | 시뮬레이션(Isaac Sim) 및 실장비 — **동일 스키마로 수용** |
| 도메인 | baseline → logistics → manufacturing → service → integrated (단계별 확장) |
| 컨테이너 | LeRobotDataset v3.0 + `safety.*` 확장 프로파일 |

### 준거 표준
| 축 | 표준 |
|---|---|
| harm / hazard / risk 용어 정의 | KS A ISO/IEC Guide 51 |
| 위험성 평가 절차 | ISO 12100 |
| 안전기능 분류 | KS B ISO 10218-1 |
| 협업 안전 (SSM / PFL), 신체부위 한계 | KS B ISO/TS 15066 (Table A.2) |
| 이동 로봇 동적 안정성 | ISO/CD 25785-1 |
| AI 제어정책 기능안전 | ISO/IEC CD TS 22440-1 |

---

## 2. 설계 원칙

**원칙 1 — 인덱스를 신뢰하지 않는다.**
상태 배열이 따르는 관절 순서는 물리엔진이 아티큘레이션을 파싱한 뒤 보고하는 순서이며, USD 순회 순서와도 의미적 순서와도 다를 수 있습니다.
Isaac Sim 문서도 관절 이름과 순서가 정확히 일치하지 않으면 제어가 깨진다고 경고합니다.
→ **모든 벡터 필드는 이름 배열(`*_names`)을 함께 저장**하고, 소비 측은 이름으로 조회합니다.
스키마에 "3번 인덱스는 무릎"과 같은 암묵적 약속을 두지 않습니다.

**원칙 2 — 네이티브 우선, 매핑은 별도 계층.**
로봇마다 DOF가 다르므로 공용 벡터에 패딩하지 않습니다. 각 로봇의 네이티브 벡터를 그대로 저장하고,
의미 키(canonical key) ↔ 네이티브 이름의 **매핑 파일을 별도 배포**합니다.
Isaac Lab의 sim-to-sim 전이가 `source_joint_names` / `target_joint_names` 두 리스트를 쓰는 방식과 동일한 구조라,
기존 도구와 호환됩니다.

**원칙 3 — sim과 real은 같은 필드, 다른 출처 표기.**
접촉력은 시뮬레이션에서는 물리엔진 산출값, 실장비에서는 F/T 센서·능동형 더미 측정값입니다.
같은 필드에 담되 `source`와 `measurement_method`로 구분해 sim-to-real 검증의 근거를 남깁니다.

**원칙 4 — 라벨은 반드시 근거를 동반한다.**
`outcome=fail`만 기록하면 재검증이 불가능합니다. **판정 규칙 ID와 임계값 초과 실측치**를 함께 저장합니다.

**원칙 5 — 스키마 버전과 데이터 버전을 분리한다.**
`schema_version`(포맷 규격)과 데이터셋 태그(`v26.11.0`)를 독립 관리합니다.

---

## 3. 식별 체계

```
dataset_id   : keti-humanoid-hri-safety
repo_id      : <HF org>/keti-humanoid-hri-safety
episode_id   : {scenario_id}_{episode_seq:04d}_seed{seed}
               예) S1a_0000_seed1497580082
scenario_id  : S{n}{variant}      예) S1a, S1b, S3c
rule_id      : R-{표준약칭}-{일련}  예) R-TS15066-A2-03
```

`episode_id` 는 시드와 **에피소드 일련번호**를 함께 담습니다. 재현에 둘 다 필요하기
때문입니다 — **재현이 성립하는 단위는 실행이지 에피소드가 아닙니다.** 같은
`--seq`·`--episodes` 로 다시 실행하면 그 실행의 에피소드가 전부 같은 값으로 나오지만,
에피소드 하나만 따로 돌리면 기록된 랜덤화 입력이 전부 일치하는데도 궤적이 첫 스텝부터
갈라집니다(2026-08-20 실측). 그래서 재현은 **그 에피소드를 만든 실행을, 첫 번호부터 그 에피소드까지**
다시 돌려 확인합니다. 실행의 첫 번호는 에피소드 메타 `run_start_seq` 에 있습니다 — 재현 명령은
`--seq run_start_seq --episodes (episode_seq − run_start_seq + 1)` 입니다.
원인은 아직 모릅니다 — `generation/README.md` 「재현은 실행 단위로만 됩니다」를 보십시오.

조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)에는 이 제약이 닿지 않습니다 — 조건표의 행 하나가 생성기의 실행 하나라 에피소드
하나가 곧 실행입니다. `run_start_seq` 가 그 행 자신이고 재현 명령은 `--seq <행> --episodes 1` 입니다(에피소드의
`notes.randomization.condition.row` 가 행 번호를 같이 적습니다).

---

## 4. 계층 구조 (LeRobotDataset v3.0 매핑)

```
keti-humanoid-hri-safety/
├── data/chunk-000/file-000.parquet        # 스텝 시계열 (§6)
├── videos/observation.images.{cam}/chunk-000/file-000.mp4  # 카메라 영상 (§6) ¹
├── packs/{pack}/episode_{episode_index:06d}.npz  ★확장  # 깊이·마스크 팩 (§6) ³
├── meta/
│   ├── info.json                          # LeRobot 표준 (features, fps, 버전)
│   ├── episodes/…                         # 에피소드 인덱스
│   ├── tasks.parquet                      # 태스크 정의
│   ├── stats.json                         # 정규화 통계
│   ├── safety_schema.json      ★확장      # 본 규격서의 기계가독 정의
│   ├── scenarios.json          ★확장      # 시나리오 카탈로그
│   ├── rules.json              ★확장      # 판정 규칙 및 임계값 (§9)
│   ├── assets.json             ★확장      # 자산 정본 — 경로·라이선스·SHA-256·텍스처 출처 ²
│   ├── cameras.json            ★확장      # 카메라 정본 — 해상도·시야각·장착·상자, 팩을 선언하면 그 형식 ¹
│   ├── packs.json              ★확장      # 팩 목록 — 에피소드마다 파일·프레임 수·SHA-256 ³
│   └── joint_map/{robot_model}.yaml ★확장 # 관절 매핑 (§7)
├── assets/                     ★확장      # 이 배치의 에피소드가 쓰는 재배포 자산 (USD) ²
│   ├── textures/                          # 그 USD 가 ./textures/ 로 참조하는 텍스처 ²
│   └── {robot}/                           # 폴더째 싣는 로봇 — USD · 그것이 참조하는 레이어 · 라이선스 파일 ²
├── NOTICE                                 # 포함 자산 출처·라이선스 전문
├── LICENSE                                # CC BY 4.0
├── README.md                              # Dataset Card (영문)
└── README.ko.md                           # Dataset Card (한국어)
```

★ 표시가 본 과제 확장분입니다. LeRobot 표준 로더는 이를 무시하고도 동작하며,
안전성 평가용 소비자만 추가로 읽으면 되도록 **하위 호환**을 유지합니다.

¹ **카메라가 있는 배포본에만 있습니다.** `meta/cameras.json` 이 있다는 것이 §6 의 카메라
필드를 필수로 만드는 조건이고, 26.11 배포본에는 둘 다 있습니다(카메라 없는 배치를 변환한 배포본에는 둘 다 없습니다). `{cam}` 은
`meta/cameras.json` 의 카메라 이름입니다(일곱 — §6 「조건부 필드 — 카메라」).

² **배포본마다 다릅니다.** `assets/` 에는 에피소드의 `scene_assets` 가 가리키는 자산 중
`meta/assets.json` 이 재배포한다고 적은 것만 실립니다 — S1d 배치의 배포본은 물류 통로 씬과 시각 메시를
입힌 인체 더미의 USD 둘, 그리고 그 둘이 쓰는 CC0 텍스처(`textures/` — 창고 씬의 상자 라벨에 쓰는
KETI CI 도)이고, S1e · S1f · S1g 를 담은 배포본은 창고 씬 · 조끼를 입힌
관절 인형 · 들고 다니는 상자의 USD 와 그것들을 엮은 장면(Stage), H1 폴더(`unitree_h1/`), 그리고 텍스처(`textures/` — CC0
텍스처와 KETI CI)입니다(상자를 쓰지 않는 시나리오만 담으면 상자 · Stage 가 실리지 않고, KETI CI 는 창고 씬의 상자 라벨로 실립니다). `scene_assets` 는 시나리오가 썼다고 말하는 자산
(`meta/scenarios.json` 의 `scene.assets` 와 인체 모델 `scene.human_model` 의 USD)과 에피소드의
`robot_model` 이 가리키는 로봇 USD(`meta/assets.json` 의 `robots`)를 빠뜨릴 수 없습니다 — 재배포하지
않는 지면(과 싣지 않는 로봇)도 목록에는 있어야 합니다(라이선스 기록이 이 목록입니다). 텍스처가 있는 USD 는
`meta/assets.json` 의 그 자산 항목 `bundle.files` 가 파일마다 출처·라이선스·SHA-256 을 적습니다. 로봇은
제 폴더째 실립니다(`assets/{robot}/` — USD · 그것이 상대 경로로 참조하는 레이어 · 라이선스 파일을 바이트 그대로)
— 그 항목의 `ship.files` 가 파일마다 출처·라이선스·SHA-256 을 적고, USD 의 행은 `usd_sha256` 과 같습니다.
검증기 C6 가 `assets/` 의 모든 파일을 `meta/assets.json` 과 대조합니다 — USD 는 `usd_sha256`,
텍스처는 `bundle.files`, 로봇 폴더는 `ship.files` (기록에 없는 파일 · 빠진 파일 · 다른 바이트는 ERROR).

³ **카메라 정본이 팩을 선언할 때만 있습니다 — 지금 정본은 선언하지 않습니다.** 깊이·분할
마스크는 싣지 않고(§6 「조건부 필드 — 카메라」), 팩 코드는 남겨 `meta/cameras.json` 의
`packs` 로 켜고 끕니다. `{pack}` 은 거기 선언된 팩 이름입니다. 정본이 팩을 선언하지 않았는데
`packs/` 나 `meta/packs.json` 이 있으면 검증기가 오류를 냅니다(§10 항목 14).

---

## 5. 에피소드 단위 메타데이터

| 필드 | 타입 | 단위/허용값 | 필수 | 설명 |
|---|---|---|---|---|
| `episode_id` | string | §3 규칙 | ✔ | 전역 고유 |
| `schema_version` | string | `0.2` | ✔ | 본 규격서 버전 |
| `domain` | enum | baseline / logistics / manufacturing / service / integrated | ✔ | 단계별 확장 축 |
| `source` | enum | sim / real | ✔ | **실장비 데이터 혼재 대비** |
| `robot_model` | string | `unitree_h1` / `unitree_g1_29dof` / … | ✔ | §7 매핑 키 |
| `robot_embodiment` | enum | biped / wheeled | ✔ | |
| `dof` | int | 19 / 29 / … | ✔ | 관절 수 |
| `joint_names` | list[string] | 아티큘레이션 순서 | ✔ | **정본. 인덱스 해석의 유일한 근거** |
| `scenario_id` | string | §3 규칙 | ✔ | scenarios.json 참조 |
| `task_archetype` | enum | transport / pick_place / machine_tending / handover / locomotion / hri | ✔ | |
| `hazard_class` | enum | collision / entrapment / fall / drop / slip / malfunction / comm_error | ✔ | |
| `outcome` | enum | nominal / near_miss / fail | ✔ | §9 규칙으로 자동 판정 |
| `severity` | enum | S0 / S1 / S2 / S3 | ✔ | Guide 51 harm 등급 |
| `risk_score` | float | 0–1 | | 발생확률 × 심각도(정규화) |
| `triggered_rules` | list[string] | rule_id 배열 | ✔ | **판정 근거. 빈 배열이면 nominal** |
| `safety_function` | enum | none / protective_stop / ssm / pfl / position_hold / mode_switch | ✔ | ISO 10218-1 |
| `standard_refs` | list[string] | 예 `ISO/TS 15066 A.2` | ✔ | |
| `measurement_method` | enum | physx / newton / ft_sensor / dummy_active / motion_capture | ✔ | sim-real 대조 근거 |
| `action_space` | enum | position_rad / torque_nm | ✔ | `action` 열의 단위. §6 「`action` 은 정책의 출력이 아니라 관절 목표입니다」 참고 |
| `termination` | object | `{terminated, truncated, reason[]}` | ✔ | 이 에피소드가 **왜 끝났는가**. `terminated` 는 종료 조건 성립(예: `fall_orientation`), `truncated` 는 시간 만료. JSON 문자열로 실림 |
| `degenerate` | bool | | ✔ | 참이면 그 에피소드는 분석에 쓰지 마십시오 — 스텝 수가 지나치게 짧거나 기록이 온전하지 않습니다. `R-TS15066-SSM-02` 를 선언한 시나리오는 그 밖에 **0 스텝부터 사람(인형)이 처음 닿기 전까지** 인형이 넘어졌거나(머리가 1.0 m 아래 · 몸통이 45° 넘게 기울어짐), 붙잡혀 있는 스텝에서 몸통이 10° 이상 기울었거나(걸음 판정의 기울기), 붙잡혀 서 있는 스텝에서 부위 높이가 서 있는 높이와 대본이 걷기 전이면 1.5 cm, 걷고 멈춘 뒤면 10 cm 넘게 다른(몸 전체가 함께 뜬 높이를 뺀 뒤) 에피소드도 참입니다 — `generation/envs/s1d/jointed_walk_metrics.py`, 문턱은 모두 자체 설정(멈춘 뒤의 10 cm 는 걷고 멈춘 인형이 마지막 걸음의 다리를 그대로 둘 수 있어 정함 — 그런 행 13개에서 2026-10-26 에 잰 멈춘 뒤 가장 큰 어긋남이 8.1 cm 였음). 걸음 판정의 다른 셋(붙잡는 손의 수평력 · 기울기 토크 · 발이 받는 몫)은 긴 걸음으로 정한 것이라 보지 않습니다. **로봇이 넘어진 것은 해당하지 않습니다** — 결과로 기록됩니다 |
| `sim_backend` | object | `{isaacsim, isaaclab, physics, lerobot}` 버전 + `generator_commit`(이 저장소의 40자 커밋 해시) · `generator_dirty`(커밋되지 않은 변경이 있었는가 — 배포본에서는 항상 `false`). 조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 `policy` 도 — `{source, sha256, behavior}`: 걸은 보행 정책 체크포인트의 출처(`isaaclab-published:rsl_rl/<과제>` = Isaac Lab 이 게시한 체크포인트 · `file:<저장소 안의 경로>` · 저장소 밖이면 `file:<파일 이름>`) · 그 파일의 SHA-256 · 과제 명령과 정책 사이 행동 층의 `모듈:이름` | sim 필수 | 재현성. 어느 코드가 만들었는가 · 조건표 시나리오의 C9 재현은 `policy` 로 다시 돌립니다 |
| `randomization_seed` | int | | sim 필수 | 재현 실행 키 |
| `run_start_seq` | int | 0 이상 · 자기 `episode_seq` 이하 | sim 필수 | 이 에피소드를 만든 **실행의 첫 번호**(§3). 재현 명령의 `--seq` |
| `scene_assets` | list[object] | `{path, license, redistributed}` | ✔ | 자산 라이선스 추적 |
| `fps` | int | Hz | ✔ | 기록 주기 |
| `duration_s` | float | s | ✔ | |
| `human_model` | string | `meta/assets.json` `scene` 의 인체 모델 키(예 `keti_dummy_ts15066_v1_skin` · `keti_dummy_ts15066_v2_vest`) / null | | 시나리오의 `scene.human_model` · §8 더미 규격 |
| `notes` | string | | ✔ | **결과 값을 과대해석하지 않는 데 필요한 단서**를 담은 JSON 문자열. `provisional`(SSM 파라미터가 잠정값인가) · `not_evaluated`(평가되지 않은 규칙과 사유·부위·최대 힘) · `contact_observed`(라벨과 무관하게 접촉이 관측됐는가) · `contact_detection_limits`(잡지 못하는 접촉 경로) · `hold_contaminated_steps`(인체를 세워 두는 서보 힘이 접촉력에 섞인 스텝 수) · `hold_release_step`(이 스텝부터 더미가 자유 물체. **`-1` 은 「끝까지 해제되지 않음」** — 로봇이 더미 근처에 오지 않은 에피소드입니다. `R-TS15066-SSM-02` 를 선언한 시나리오는 붙잡기가 첫 접촉에서만 풀리므로 로봇이 닿지 않은 에피소드이고, 닿은 에피소드는 첫 `contact_onset` 다음 스텝 · `hold_contaminated_steps` 1 이고 라벨은 그 첫 접촉 스텝도 판정함 — §6 「조건부 필드 — 사람과 로봇」 · §9.1) · **`ssm_params`(이 회랑을 만든 실제 입력 — 공개 `rules.json` 이 `intrusion_distance` 를 조항 참조로만 싣기 때문에 이것이 없으면 회랑을 복원할 수 없습니다. §9.3)** · `human_region_radius` · `contact_onset_unresolved_regions`·`_reason` · `contact_type_unresolved_reason`(어느 접촉 경로가 잡히지 않는가) · `randomization`(이 에피소드에서 뽑힌 값 — 조건표가 있는 시나리오는 `condition` 에 조건표의 행 번호 · 그 행의 열 값 · 그 행이 정한 자리(로봇의 출발 · 목표 · 길과 사람의 대본) · 로봇에게 과제가 온 스텝 `task_started_step` · 행의 종류 `design`(`random` | `grid`) · 격자 칸 `grid_cell`(`[로봇 칸, 사람 칸]`, 무작위 행은 null — 평평한 열 `condition_design` · `grid_cell_robot` · `grid_cell_human` 이 같은 사실을 싣고 격자 행이 아니면 두 칸 열이 -1) · 겉모습을 선언한 시나리오는 `appearance` 에 그 행이 뽑은 겉모습 — 축 이름 → 색 이름 또는 수 · 겉모습을 뽑지 않은 실행(`--no-appearance`)은 null — 카메라 없이 도는 실행도 뽑은 값을 적습니다) · `imu_link` · `asset_paths_resolved` · `task_completed_step`(대본의 목표 자리에 선 것이 확인된 스텝 — §6 「조건부 필드 — 사람과 로봇」의 완료 기준 · **`-1` 은 시간 안에 끝내지 못함**, null 은 완료 기준이 없는 시나리오 — **`R-TS15066-SSM-02` 를 선언한 시나리오는 완료 기준이 있어 null 이 아니라 정수**이고, 변환기는 `task_completed_step` 이 정수가 아닌 녹화를 첫 프레임 전에 거부하며 검증기 C5 가 ERROR 로 냅니다. 정수는 그 에피소드의 `length` 보다 작습니다) · `degenerate_reason`(`R-TS15066-SSM-02` 를 선언한 시나리오에서 `degenerate` 가 참인 까닭의 문장 — 거짓이면 null) · `box_released_step`(든 상자를 놓은 스텝 — 사람과 처음 닿은 스텝입니다. **`-1` 은 놓지 않았거나 든 상자가 없는 시나리오**) · `camera_in_box_steps`(로봇 카메라 다섯마다 렌즈가 든 상자 안이던 스텝 수 — 그 스텝의 `observation.human_visible_fraction.<카메라>` 는 상자를 빼고 센 값이고 그림에는 상자 속이 보임, §6 「조건부 필드 — 사람과 로봇」). **이 셋은 `R-TS15066-SSM-02` 를 선언한 시나리오의 기록만 채우고 다른 시나리오에서는 셋 모두 null 입니다** — 정수나 객체나 문장이 아닌 형이 실리면 C5 가 ERROR 로 내지만 값끼리의 모순(`degenerate` 와 `degenerate_reason` 등)은 검사하지 않습니다. · `task_stages`(과제의 단계마다 `{name, completed, end_step, duration_s}` 한 원소 — 단계 이름은 길의 종류가 정합니다: 되돌아 나가는 길(`S1e`)은 `enter` · `turn` · `exit`, 곧은 길(`S1f`)은 `walk`, 제자리에서 돌고 걷는 길(`S1g`)은 `turn` · `walk`. 길 위의 단계는 길이로 끊고 길을 0.05 m 마다 찍은 표본(`generation/task_path.py` 의 `PATH_STEP_M`) 가운데 그 구간이 끝나는 표본에 로봇이 닿은 첫 스텝에 끝나며, 제자리에서 도는 단계는 목표 방향까지 남은 요가 5° 안에 든 첫 스텝 — 길을 따르는 쪽이 돌기를 마치고 걷기를 시작하는 스텝(`generation/task_path.py` 의 `TURN_DONE_DEG`)이고 로봇은 아직 그 자리에 있습니다 — 에 끝나고, 마지막 단계는 과제를 끝낸 스텝(`task_completed_step`)에 끝납니다(과제가 로봇에게 오기 전의 스텝으로 기록된 완료는 아무 단계도 닫지 않습니다). 단계 k 는 단계 k−1 이 끝난 스텝(첫 단계는 과제가 로봇에게 온 스텝)부터 자기가 끝난 스텝까지이고 `duration_s` 는 `(end_step − 시작) ÷ fps` 입니다. **`end_step` `-1` · `duration_s` null · `completed` 거짓은 그 단계가 끝나지 않았다는 뜻입니다** — 과제를 시간 안에 끝내지 못했거나 과제가 오지 않은 에피소드는 열린 단계를 가지고, 과제를 끝낸 에피소드의 단계는 모두 끝나 있습니다. `completed` 는 `end_step` 이 0 이상일 때만 참입니다) · `human_stature_m`(사람 인형의 키(m) — 인체 치수 정본 `generation/dummy/anthropometry.yaml` 의 값이고, 「봄」 규칙이 화면에 찍힌 사람의 키를 셀 때 씁니다) · `task_time_limit_s`(과제의 제한 시간(초) — 환경의 에피소드 길이). **이 셋도 `R-TS15066-SSM-02` 를 선언한 시나리오의 기록만 채웁니다** — 그 시나리오의 에피소드는 `task_stages` 가 비어 있지 않은 목록이고 `human_stature_m` · `task_time_limit_s` 가 유한한 양수이며(변환기는 셋 가운데 하나라도 없는 녹화를 첫 프레임 전에 거부하고 C5 가 ERROR 로 냅니다), 다른 시나리오는 셋 모두 null 입니다. `R-TS15066-SSM-02` 를 선언한 시나리오의 `ssm_params` 는 그 규칙의 파라미터 다섯과 `stop_table` 입니다 |
| `episode_spec` | object | `{world, humans, task, robot, condition, repro}` | 조건표 필수 | 이 에피소드가 기록된 조건 — 아래 「조건표가 있는 시나리오의 에피소드 열」. JSON 문자열로 실림 |
| `evidence.min_separation_margin_m` | float | m | 조건표 필수 | 근거 요약 — 가장 작은 정지거리 여유 |
| `evidence.first_contact_s` | float | s / null | 조건표 필수 | 근거 요약 — 첫 접촉의 시각 |
| `evidence.force_ratio.BR-01` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-01` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-02` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-02` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-03` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-03` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-04` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-04` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-05` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-05` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-06` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-06` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-07` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-07` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-08` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-08` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-09` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-09` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-10` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-10` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-11` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-11` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio.BR-12` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 부위 `BR-12` 의 힘 ÷ 판정이 쓴 한계의 최대 |
| `evidence.force_ratio_max` | float | 배수 / inf / null | 조건표 필수 | 근거 요약 — 열두 부위 값의 최대 |
| `evidence.human_first_seen_s` | float | s / null | 조건표 필수 | 근거 요약 — 로봇 카메라가 사람을 처음 본 시각 |
| `evidence.human_last_seen_s` | float | s / null | 조건표 필수 | 근거 요약 — 로봇 카메라가 사람을 마지막으로 본 시각 |
| `condition_design` | enum | random / grid | 조건표 필수 | 조건표의 무작위 행인가 격자 행인가 |
| `grid_cell_robot` | int | 0 이상 · 무작위 행은 -1 | 조건표 필수 | 격자 행이 놓인 칸의 로봇 축 번호 |
| `grid_cell_human` | int | 0 이상 · 무작위 행은 -1 | 조건표 필수 | 격자 행이 놓인 칸의 사람 축 번호 |

> `scene_assets`를 에피소드 단위로 남기면, 나중에 특정 자산의 라이선스 문제가 발견되어도
> **해당 에피소드만 선별 제거**할 수 있습니다. 전체 재작업을 막는 보험입니다.

### 조건표가 있는 시나리오의 에피소드 열 (자체 설정)

배포본의 `meta/scenarios.json` 에서 시나리오가 조건표(`conditions`)를 선언하면 그 시나리오의 에피소드는 위 표에서 「조건표 필수」로 표시한 열 스물하나를 싣습니다. 기계가독 정본은 `meta/safety_schema.json` 의 `scenario_conditional_episode_fields` 이고, 이 데이터셋에서는 `S1e` · `S1f` · `S1g` 가 그 시나리오입니다. 열의 이름과 정의는 이 데이터셋이 정한 것입니다(자체 설정 — 표준이 정의하지 않음).

- **`episode_spec`** — 이 에피소드가 기록된 조건을 모은 JSON 객체(문자열 한 칸으로 실림). 멤버는 여섯입니다(`meta/safety_schema.json` 의 `episode_spec_members`)
  - `world` — 시나리오 · 띄운 자산(`meta/assets.json` 의 키) · 그 자산을 엮은 Stage · 뽑힌 겉모습(`notes.randomization.appearance`)
  - `humans` — 사람마다 모델 · 키(m) · 사람 정책의 이름과 판(대본 · 생성 커밋) · 대본(선 자리 · 처음 보는 쪽 · 돌기 · 걷는 방향과 속도 · 방아쇠 · 멈추는 곳). 이 데이터셋은 한 명
  - `task` — 과업 종류 · 문장 · 출발 · 목표 · 길 · 운반물(상자의 자산 키 · 무게 · 크기 — 들지 않는 시나리오는 null) · 제한 시간(s) · 단계 목록(이름 · 목표 · 그 단계의 목표 속도 또는 목표 요 속도)
  - `robot` — 로봇 모델 · 자유도 · 보행 정책과 행동 층(`sim_backend.policy`)
  - `condition` — 행의 종류(`random` / `grid`) · 격자 칸(`[로봇 칸, 사람 칸]`, 무작위 행은 null) · 조건표의 행 번호 · 그 행의 열 값과 고정값
  - `repro` — 생성 커밋 · 시드 · 실행의 첫 번호 · 에피소드 번호 · 다시 돌리는 명령(`generation/rerun_seed.sh`)
  - 고정되는 시험 조건은 `world` · `humans` · `task` · `condition` 이고, 다른 제어기를 시험할 때 바꾸는 부분이 `robot` 입니다
- **근거 요약 `evidence.*`** — 라벨러가 라벨과 같은 녹화 배열 · 같은 정본(`meta/rules.json` · `meta/scenarios.json` · `meta/cameras.json`)으로 라벨과 함께 내는 수입니다(`generation/labeling/evidence.py`)
  - `min_separation_margin_m` — 모든 스텝의 `safety.min_distance_human − safety.separation_required` 가운데 가장 작은 값(m, 64비트 실수로 셈). 음수면 그 에피소드에서 사람이 요구 이격 안에 들어왔습니다
  - `first_contact_s` — 첫 `contact_onset` 스텝 ÷ `fps`(s). 닿지 않은 에피소드는 null
  - `force_ratio.<부위>` — 그 부위의 `R-TS15066-A2-*` 규칙이 판정하는 스텝(§9.1 — 붙잡기가 풀린 뒤와, 붙잡기가 첫 접촉에서 풀리는 시나리오의 첫 접촉 스텝)에서, 힘 ÷ 그 스텝의 접촉에 판정이 쓴 한계의 최대입니다. 한계는 준정적 접촉이면 준정적 힘, 동적 접촉이면 그 힘에 배수(`transient_multiplier`)를 곱한 값, `R-HRI-CONTACT-01` 을 선언한 시나리오에서 첫 접촉 뒤 유형이 매겨지지 않은 접촉이면 둘 가운데 작은 쪽입니다. **1 보다 크면 그 부위의 접촉 한계 규칙이 성립한 것입니다.** 표 A.2 가 동적 접촉을 허용하지 않는 부위는 동적 접촉이면 힘과 상관없이 inf, 유형이 매겨지지 않은 접촉이면 힘이 0 보다 클 때 inf 입니다. 판정하는 스텝에 그 부위가 닿지 않았으면 0.0, 시나리오가 그 부위의 규칙을 선언하지 않았거나 한계 값이 비었거나 에피소드가 그 부위를 기록하지 않았으면 null 입니다
  - `force_ratio_max` — 열두 부위 값 가운데 null 이 아닌 것의 최대. 모두 null 이면 null
  - `human_first_seen_s` · `human_last_seen_s` — 로봇 카메라(`meta/cameras.json` 의 `type: mounted`) 가운데 하나가 정본의 「봄」(`meta/cameras.json` 의 `seeing` — 보이는 몫과 화면 속 키의 두 문턱)으로 사람을 본 처음 · 마지막 스텝 ÷ `fps`(s). 보이는 몫은 `observation.human_visible_fraction.<카메라>` 열이고, 화면 속 키는 배포본의 카메라 자세 열(`observation.camera_pose.<카메라>`) · 사람 자세 열(`observation.human.root_pose`) · 카메라 정본의 초점 거리와 `notes.human_stature_m` 으로 다시 셉니다. 보지 못한 에피소드와, 셀 수 없는 배포본(카메라 정본 · 그 열 · 키 가운데 하나가 없음)은 null 입니다
  - 시각은 모두 스텝 번호 ÷ `fps` 입니다. 값이 없는 칸은 parquet 의 실수 열에 null 로 실리고 pandas 로 읽으면 NaN 입니다. `inf` 는 값입니다
  - `generation/labeling/check_release_labels.py` 가 배포본만으로 근거 요약을 다시 내 실린 값과 그대로 대조합니다 — 실린 수로 대조하고, 적재기가 넘기는 float32 로는 대조하지 않습니다
- **`condition_design`** — 조건표 행의 종류. `random` 은 조건표의 창에서 뽑은 행, `grid` 는 격자(실패 지도)의 칸 하나에 놓인 행입니다. `notes.randomization.condition` 의 `design` 과 같습니다
- **`grid_cell_robot` · `grid_cell_human`** — 격자 행이 놓인 칸의 로봇 축 · 사람 축 번호(0 부터 — 축과 칸 수는 `meta/scenarios.json` 의 `conditions.grid`). `notes.randomization.condition` 의 `grid_cell`(`[로봇 칸, 사람 칸]`)과 같고, 무작위 행은 -1 · -1 입니다(그 기록의 `grid_cell` 은 null). 조건표가 없는 시나리오와 섞인 배포본에서는 그 시나리오의 행에 이 스물하나가 비어 있고, 그래서 두 칸 열은 실수 열로 실립니다(값은 정수)

---

## 6. 스텝 단위 시계열 (Parquet)

| 필드 | shape | 단위 | 필수 | 비고 |
|---|---|---|---|---|
| `timestamp` | scalar | s | ✔ | 에피소드 시작 기준 |
| `frame_index` | scalar | – | ✔ | |
| `observation.state.joint_pos` | (dof,) | rad | 선택 | `joint_names` 순서 |
| `observation.state.joint_vel` | (dof,) | rad/s | 선택 | |
| `observation.state.joint_torque` | (dof,) | N·m | 선택 | |
| `observation.root_pose` | (7,) | m, quat(wxyz) | ✔ | 월드 기준 |
| `observation.root_vel` | (6,) | m/s, rad/s | ✔ | 선속도+각속도 |
| `observation.imu` | (10,) | quat(wxyz), rad/s, m/s² | 선택 | 쿼터니언 4 + 바디 프레임 각속도 3 + 고유 선가속도 3. 프레임은 에피소드 메타의 `imu_link` |
| `observation.contact.force` | (n_region,) | N | ✔ | §8 부위별 최대 법선력. **기록 스텝 순간값이 아니라 그 스텝의 물리 서브스텝 최댓값** |
| `observation.contact.v_rel` | (n_region,) | m/s | 선택 | 접촉 지점 상대속도의 **근사**. 표 A.4 에너지 축·식 (A.6) 속도 축의 입력이며 첫 공개 배포본은 쓰지 않습니다 — 산출 방법과 한계는 아래 참고 |
| `observation.contact.pressure` | (n_region,) | N/cm² | 선택 | 접촉 면적을 실측할 수 있을 때만 실림 — 아래 「선택 필드 — `observation.contact.pressure`」 참고 |
| `observation.contact.region_names` | list[string] | – | ✔ | **부위 이름 배열(정본)**. ※JSON 문자열로 저장 |
| `observation.contact.type` | (n_region,) | – | ✔ | `none` / `transient` / `quasistatic`. Table A.2 의 어느 한계를 적용할지 정하는 근거. ※JSON 문자열로 저장 |
| `observation.object.names` | list[string] | – | 조건부 | **파지 대상 이름 배열(정본)** |
| `observation.object.position` | (n_obj, 3) | m | 조건부 | 월드 기준 무게중심 |
| `observation.object.grasped` | (n_obj,) | bool | 조건부 | 파지 상태. `true`→`false` 가 이탈 시점 |
| `observation.ft.{sensor}` | (6,) | N, N·m | | 실장비 대비 |
| `observation.images.head_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 머리 카메라 영상 — D435 위 이마에 덧단 모듈, 수평 정면. LeRobot 영상 피처(`videos/`). 해상도·시야각·장착은 `meta/cameras.json` — 아래 「조건부 필드 — 카메라」 |
| `observation.images.head_down_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 머리 아래보기 카메라 영상 — 실물 H1 의 D435 자리와 방향이라 발 앞 바닥을 봄. 형식은 위와 같음 |
| `observation.images.left_wrist_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 왼 손목 카메라 영상 — 실물 H1 에 없는 카메라로, 팔꿈치 링크 윗면 끝 쪽에서 주먹 쪽을 봄. 형식은 위와 같음 |
| `observation.images.right_wrist_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 오른 손목 카메라 영상 — 왼쪽과 거울 장착(이것도 실물 H1 에 없음). 형식은 위와 같음 |
| `observation.images.rear_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 뒤 카메라 영상 — 머리 뒤쪽에 덧단 모듈에서 바로 뒤를 수평으로 봄(69.4° 일반 렌즈). 형식은 위와 같음 |
| `observation.images.third_person_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 로봇을 따라가는 3인칭 카메라 영상. 자리는 시나리오마다 `meta/cameras.json` 의 `sites` — S1g 는 CCTV 와 같이 사람이 오는 쪽마다 하나이고 사람이 걸어가는 쪽에서 로봇을 봄. 형식은 위와 같음 |
| `observation.images.cctv_rgb` | (H, W, 3) | RGB (uint8) | 조건부 | 창고에 고정된 CCTV 영상 — 높은 곳에서 그 시나리오의 부딪히는 구간을 내려다봄. 자리는 시나리오마다 `meta/cameras.json` 의 `sites` — S1d 는 통로 옆 랙 면 앞, S1e 는 서쪽 벽, S1f 는 통로 서쪽 끝, S1g 는 사람이 걸어가는 쪽에서 로봇과 그 너머에서 다가오는 사람을 봄(사람이 오는 쪽마다 자리 하나 — `by_approach_side`, 에피소드는 조건표 행의 `approach_side` 의 자리: 동쪽에서 오면 통로 서쪽 끝, 서쪽에서 오면 로봇의 출발 자리에서 동쪽으로 7.4 m 떨어진 통로 위 — 서쪽 끝의 자리를 로봇의 출발 x 에 대해 뒤집은 자리). 형식은 위와 같음 |
| `observation.bbox2d.human` | (4,) | px | 조건부 | `head_rgb` 영상 속 더미의 상자 `[x_min, y_min, x_max, y_max]`. 양끝을 포함하는 픽셀 번호이고, 보이지 않으면 `[-1, -1, -1, -1]`. 검출기 출력이 아니라 정답 마스크에서 유도 |
| `observation.camera_pose.head_rgb` | (7,) | m, quat(wxyz) | 조건부 | 머리 카메라의 월드 자세. 축은 광학 규약(+z 광축 · +y 영상 아래) |
| `observation.camera_pose.head_down_rgb` | (7,) | m, quat(wxyz) | 조건부 | 머리 아래보기 카메라의 월드 자세. 표현은 위와 같음 |
| `observation.camera_pose.left_wrist_rgb` | (7,) | m, quat(wxyz) | 조건부 | 왼 손목 카메라의 월드 자세. 표현은 위와 같음 |
| `observation.camera_pose.right_wrist_rgb` | (7,) | m, quat(wxyz) | 조건부 | 오른 손목 카메라의 월드 자세. 표현은 위와 같음 |
| `observation.camera_pose.rear_rgb` | (7,) | m, quat(wxyz) | 조건부 | 뒤 카메라의 월드 자세. 표현은 위와 같음 |
| `observation.camera_pose.third_person_rgb` | (7,) | m, quat(wxyz) | 조건부 | 3인칭 카메라의 월드 자세. 표현은 위와 같음 |
| `observation.camera_pose.cctv_rgb` | (7,) | m, quat(wxyz) | 조건부 | CCTV 의 월드 자세 — 고정이라 매 프레임 같은 값. 표현은 위와 같음 |
| `observation.command_velocity` | (3,) | m/s, m/s, rad/s | 조건부 | 보행 정책에 준 걷기 명령 — 앞 · 옆 속도와 요 각속도(로봇 몸 기준). 아래 「조건부 필드 — 사람과 로봇」 |
| `observation.human.root_pose` | (7,) | m, quat(wxyz) | 조건부 | 사람(인형) 골반의 월드 자세 |
| `observation.human.root_vel` | (6,) | m/s, rad/s | 조건부 | 사람(인형) 골반의 선속도 + 각속도(월드) |
| `observation.human_visible_fraction.head_rgb` | scalar | – | 조건부 | 머리 카메라에서 사람 몸의 보이는 몫(0~1). 렌더가 아니라 기하로 셈 — 인형 몸의 표본점(`meta/cameras.json` `seeing.sample_points` 개 · 부위 표면적에 비례해 부위마다 둘 이상 · 부위에 고정) 가운데 시야 안(투영이 그림 안이고 깊이가 클리핑 범위 안)에 있고 창고의 충돌체 · 벽 · 문, 로봇 몸, 든 상자에 가려지지 않은 점의 몫이며, 분모는 늘 표본점 전부. 바닥과 사람 자신의 몸은 가리지 않음. **「봄」** = 로봇 몸의 카메라 하나에서 이 몫이 `seeing.min_visible_fraction`(20 %) 이상이고 전신 높이(가린 부분까지 — f_y · 인형의 키 / `observation.human.root_pose` 의 깊이)가 `seeing.min_height_px`(50 px) 이상 — 두 문턱은 **자체 설정** |
| `observation.human_visible_fraction.head_down_rgb` | scalar | – | 조건부 | 머리 아래보기 카메라에서. 형식은 위와 같음 |
| `observation.human_visible_fraction.left_wrist_rgb` | scalar | – | 조건부 | 왼 손목 카메라에서. 형식은 위와 같음 |
| `observation.human_visible_fraction.right_wrist_rgb` | scalar | – | 조건부 | 오른 손목 카메라에서. 형식은 위와 같음 |
| `observation.human_visible_fraction.rear_rgb` | scalar | – | 조건부 | 뒤 카메라에서. 형식은 위와 같음 |
| `observation.depth.head` | – | – | – | **싣지 않음** — 깊이는 렌더하지 않습니다. `meta/cameras.json` 이 팩을 선언할 때만 LeRobot 피처가 아닌 팩의 `depth` 로 실립니다 — 아래 「조건부 필드 — 카메라」 |
| `observation.segmentation.head` | – | – | – | **싣지 않음** — 분할은 녹화 때 `observation.bbox2d.human` 을 계산하는 데만 쓰고 저장하지 않습니다. 팩을 선언하면 그 팩의 `mask` 입니다 |
| `action` | (dof,) | rad | 선택 | 관절 위치 목표. 정책의 정규화 출력이 아니라 그것을 변환한 값임 — 아래 참고 |
| `safety.com` | (3,) | m | 선택 | 무게중심 월드 좌표 |
| `safety.zmp` | (2,) | m | 선택 | 지면 투영 ZMP. **정의되지 않는 스텝은 두 성분 모두 `NaN`** — 아래 「산출 방법」 |
| `safety.stability_margin` | scalar | m | 선택 | 지지다각형 경계까지 거리. **음수 = 이탈** |
| `safety.min_distance_human` | scalar | m | ✔ | 로봇 표면 ↔ 인체 표면 최소거리 |
| `safety.separation_required` | scalar | m | ✔ | ISO/TS 15066 SSM 요구 이격거리. 산정 규칙은 시나리오가 선언한 `R-TS15066-SSM-01`(`parameters` 여섯 항) 또는 `R-TS15066-SSM-02`(방향을 따진 S_p) — 아래 「산출 방법」 |
| `safety.separation_violation` | bool | – | ✔ | `min_distance < separation_required`. `R-TS15066-SSM-02` 를 선언한 시나리오는 그와 함께 `approach_speed_robot > approach_speed_threshold`(다가감 문턱 — 자체 설정) |
| `safety.approach_speed_robot` | scalar | m/s | 조건부 | 로봇(든 상자 포함)의 가장 가까운 점이 사람 쪽으로 움직이는 속도. 0 아래는 0 |
| `safety.approach_speed_human` | scalar | m/s | 조건부 | 사람의 가장 가까운 점이 로봇 쪽으로 움직이는 속도. 0 아래는 0 |
| `safety.event` | list[enum] | – | ✔ | 해당 스텝 발생 이벤트. ※JSON 문자열로 저장 |

**「선택」 열**은 로봇마다 꼴이 다른 열(관절 · 행동 · IMU · 무게중심 · ZMP · 안정 여유)과 어느 판정 규칙도 읽지 않는 열
(`observation.contact.v_rel` · `observation.contact.pressure`)입니다. 배포본은 이 열을 빼고 낼 수 있습니다. 실린 선택 열은
필수 열과 똑같이 그 열의 검사(길이 · 관절 차원 · 범위 · 유한성)를 모두 받고, 배포본의 시나리오가 선언한 규칙이 읽는 열(규칙의
`inputs` · 시나리오의 `required_fields`)은 선택 열이라도 그 배포본에서 필수입니다 — 검증기 C8 이 그 둘을 데이터의 열과
대조합니다(§10). 기계가독 정본은 `meta/safety_schema.json` 의 `optional_step_fields` 이고, 필수와 선택의 구분은 **자체 설정**입니다.

### 층 — 로우와 가공

스텝 표는 한 파일이고 열 이름도 그대로입니다. 대신 열마다 어느 층인지를 `meta/safety_schema.json` 의
`step_field_layers` 가 적습니다(`raw` · `processed`) — 학습이나 위험 예측의 입력을 로우로만 고를 수 있게
하려는 것입니다. 검증기 C1 이 그 표를 자기 상수와 통째로 대조하고, C8 은 스텝 파일의 모든 열에 층이 있는지
봅니다 — LeRobot 이 붙이는 `index` · `episode_index` · `task_index` 는 층이 없습니다.

| 층 | 뜻 | 필드 |
|---|---|---|
| 로우 | 시뮬레이터 · 센서가 그대로 낸 값 | `timestamp` · `frame_index` · `observation.state.joint_pos` · `observation.state.joint_vel` · `observation.state.joint_torque` · `observation.root_pose` · `observation.root_vel` · `observation.imu` · `observation.contact.force` · `observation.contact.pressure` · `observation.contact.region_names` · `action` · `observation.images.head_rgb` · `observation.images.head_down_rgb` · `observation.images.left_wrist_rgb` · `observation.images.right_wrist_rgb` · `observation.images.rear_rgb` · `observation.images.third_person_rgb` · `observation.images.cctv_rgb` · `observation.camera_pose.head_rgb` · `observation.camera_pose.head_down_rgb` · `observation.camera_pose.left_wrist_rgb` · `observation.camera_pose.right_wrist_rgb` · `observation.camera_pose.rear_rgb` · `observation.camera_pose.third_person_rgb` · `observation.camera_pose.cctv_rgb` · `observation.command_velocity` · `observation.human.root_pose` · `observation.human.root_vel` |
| 가공 | 판정기가 로우에서 계산한 값 | `observation.contact.v_rel` · `observation.contact.type` · `safety.com` · `safety.zmp` · `safety.stability_margin` · `safety.min_distance_human` · `safety.separation_required` · `safety.separation_violation` · `safety.event` · `observation.bbox2d.human` · `observation.human_visible_fraction.head_rgb` · `observation.human_visible_fraction.head_down_rgb` · `observation.human_visible_fraction.left_wrist_rgb` · `observation.human_visible_fraction.right_wrist_rgb` · `observation.human_visible_fraction.rear_rgb` · `safety.approach_speed_robot` · `safety.approach_speed_human` |

### ※ 표시 필드 — parquet 에는 JSON 문자열로 실립니다

위 표에서 `※JSON 문자열로 저장` 이 붙은 세 필드는 **논리적으로는 배열이지만
parquet 컬럼으로는 문자열 한 칸**입니다. 읽는 쪽은 `json.loads` 로 풀어야 합니다.

```python
import json, pandas as pd
df = pd.read_parquet("data/chunk-000/file-000.parquet")
regions = df["observation.contact.region_names"].map(json.loads)   # -> ["BR-01", ...]
```

**이유는 LeRobot v3.0 의 제약입니다.** LeRobot 의 특징 검증기는 `string` 을 스칼라로만
받고(`feature_utils.validate_feature_string`), HF 의 `Value("string")` 은 고정 길이
`Sequence` 안에 들어가지 못합니다. 우리가 고른 형태가 아니라 형식이 허용하는 형태입니다.

**여기 없는 배열 필드는 진짜 배열입니다.** `observation.contact.force` 나
`safety.com` 같은 수치 벡터는 parquet 네이티브 리스트로 실립니다. 에피소드 메타(§5)도
`joint_names`·`triggered_rules`·`standard_refs` 는 네이티브 리스트이고,
중첩 구조인 `scene_assets`·`sim_backend`·`notes`·`termination` 과 조건표가 있는 시나리오의
`episode_spec` 만 JSON 문자열입니다.

`notes` 안쪽 구조와 경계값(부위 코드·스텝 수·최대 힘·`hold_contaminated_steps` 상한·
`hold_release_step` 표식)의 기계가독 정본은 `meta/safety_schema.json` 의
`notes_member_shapes`·`notes_bounds` 이고 검증기 C5 가 강제합니다.

기계가독 정본은 `meta/safety_schema.json` 의 `json_encoded_fields` 이며,
어느 필드가 어느 쪽인지 그 파일이 배포본과 함께 나갑니다.

### `safety.stability_margin` 산출 방법

**지지다각형은 하중이 실린 발의 발자국으로 세웁니다.** 발자국은 로봇 자산의 발목 링크
충돌 메시에서 실측한 4모서리 사각형이며(치수를 손으로 넣지 않습니다),
어느 발이 하중을 지는지는 그 발의 실측 지면반력이 임계값을 넘는지로 가릅니다. 스윙 중인
발의 모서리는 지지다각형에 넣지 않습니다 — 넣으면 다각형이 부풀어 단일 지지 구간의
불안정이 가려집니다. CoM 은 **질량 가중 실측 무게중심**입니다 — 자산이 보고하는 링크별
질량과 링크별 무게중심 위치로 `Σmᵢpᵢ / Σmᵢ` 를 계산하며, 근사가 아닙니다.
값은 CoM 에서 지지다각형 경계까지의 부호 있는 거리이고, **양발이 모두 미하중인 비행
구간에서는 `-inf`** 입니다.

**동적 보행 중에는 음수가 정상입니다.** 보행은 정적 균형이 아니라 제어된 낙하이므로
CoM 이 단일 지지발의 다각형 밖으로 나가는 순간이 대부분입니다 — 실측(S1a, 2026-08-20,
정상 보행 16 에피소드, 20초 완주)으로 전체 스텝의 93~96% 가 음수였습니다. 그래서
`R-25785-STAB-01` 은 부호가 아니라 **지속시간**을 경계로 두도록 설계했고, 그 값이
`min_duration` 입니다. 현재 `meta/rules.json` 에 값이 없습니다.

> **그 설계가 S1a 배치에서는 서지 않습니다 (2026-08-28 실측).** 400 에피소드·낙상
> 43건에서 규칙이 실제로 보는 양(에피소드마다 `stability_margin < 0` 인 최장 연속
> 구간)이 **낙상과 정상을 가르지 못합니다** — 정상 중앙 71 스텝·최대 451, 낙상 중앙
> 89 스텝·최대 387 로 분포가 겹치고, 가장 잘 맞히는 경계도 낙상 43건 중 **1건**만
> 잡습니다. `min_duration` 을 어떤 값으로 채워도 이 배치에서는 규칙이 서지 않습니다.
>
> 기전이 있습니다 — 낙상 에피소드는 **자세 기준(루트 높이 + 기울기)으로 종료**되므로
> 넘어진 뒤의 여유 이력이 기록되지 않습니다. 지속시간이 쌓일 구간 자체가 없습니다.
>
> 신호가 없는 것은 아닙니다. **`-inf` 비율**(양발 모두 미하중 = 발로 서 있지 않음)은
> 낙상 중앙 0.028 대 정상 0.006 으로 갈리며, 경계 0.025 에서 낙상 25/43 을 잡고 정상
> 3/357 을 오검합니다. 규칙은 이 양을 쓰지 않습니다.
>
> **이 근거는 S1a(충돌) 낙상입니다.** S3a(경사·단차 외란)의 낙상은 다를 수 있으므로
> 규칙 형태를 다시 정하는 것은 남은 일입니다. 다만 「표본을 더 모으면 경계가
> 정해진다」는 전제는 **표본 2건에서 43건으로 늘려 확인한 결과 성립하지 않았습니다.**
> 측정은 `generation/check_fall_duration.py` 로 재현합니다.

### `safety.zmp` 산출 방법과 `NaN` 의 의미

**전신 다물체 동역학으로 산출합니다** — 지면 z=0 기준이며, 각 바디의 각운동량 변화율은
`Iᵢαᵢ + ωᵢ × (Iᵢωᵢ)` 전체를 씁니다. 관성 텐서는 자산이 월드 프레임으로 보고하는 값을
그대로 쓰므로 회전 변환이 없습니다.

자이로 항 `ωᵢ × (Iᵢωᵢ)` 을 빼면 ZMP 가 얼마나 어긋나는지 이 씬에서 쟀습니다
(`generation/envs/s1a/check_zmp_gyroscopic.py`, 384 스텝): **중앙값 0.7 mm · p95
3.6 mm · 최대 8.0 mm**(자이로 토크 합은 중앙값 0.38 N·m · 최대 3.83 N·m). 크지는
않지만 발 길이의 몇 %에 해당하므로, 정의된 대로 전체 항을 씁니다.

**두 성분이 모두 `NaN` 인 스텝은 결측이 아니라 「이 스텝에 ZMP 가 정의되지 않음」 이라는
표식입니다.** 분모 `Σmᵢ(z̈ᵢ+g)` — 바디 가속도가 함의하는 총 수직 지면반력 — 이 체중의
1% 아래로 내려가면 지면반력의 작용선을 지목할 수 없기 때문입니다. `stability_margin`
이 비지지 구간에 `-inf` 를 쓰는 것과 같은 이유입니다.

**이 표식은 비행 구간과 같지 않습니다.** S1a 실측(10 에피소드 9,990 스텝)에서 ZMP 가
`NaN` 인 74 스텝 중 27 스텝은 `stability_margin` 이 유한했습니다 — 하중이 실린 발이
있어도 수직 성분의 합이 순간적으로 사라질 수 있습니다. 둘을 같은 것으로 읽지 마십시오.

**한쪽 성분만 `NaN` 인 값은 나오지 않습니다.** 나온다면 산출 오류이며 검증기 C8 이 막습니다.
`safety.com` 에는 표식이 없습니다 — 무게중심은 언제나 정의되며, `NaN` 이면 입력이
손상된 것이라 생성 단계에서 예외로 멈춥니다.

### `safety.min_distance_human` 산출 방법

**`R-TS15066-SSM-01` 을 선언한 시나리오(`S1a` · `S1d`)는 로봇 링크 원점과 인체 부위 중심 사이 거리에서, 부위마다 같은 반경 하나(현재
`human_region_radius=0.1m`)를 뺀 값입니다**(`R-TS15066-SSM-02` 를 선언한 시나리오의 표면 거리는 이 절의 끝 문단). 로봇 쪽은
링크 origin 을 그대로 쓰고
(표면이 아닙니다), 인체 쪽은 더미의 실제 부위별 형상(`generation/dummy/
anthropometry.yaml`) 대신 구 하나로 근사합니다 — 부위별 반경은 실제로 0.042~0.088m
사이이고, 박스 형상 부위는 y 방향 반폭이 최대 0.20m(BR-04), 캡슐 형상 부위는 축 방향
길이(`UsdGeom.Capsule.height` — 반구 캡 제외, `generation/dummy/build_dummy.py`가 yaml 의
height 값을 그대로 이 속성에 씁니다)가 0.08~0.38m 인 채로 점 하나에 접혀 들어갑니다.

**오차의 방향은 부위 형상이 아니라 축마다 다릅니다.** 반경 방향(캡슐 옆면, 구 전체)은
모든 부위의 실제 반경(0.042~0.088m)이 가정값 0.1m 보다 작아 거리를 짧게 잡는 쪽
(보수적)입니다. 그러나 캡슐은 원통 양 끝에 반구 캡이 반경만큼 더 튀어나와 있어 축
방향 사정이 다릅니다 — 중심에서 캡 끝까지의 거리는 `height/2 + radius` 이고, 상완
(BR-08) 0.130+0.048=0.178m, 전완(BR-09) 0.120+0.042=0.162m, 대퇴(BR-11)
0.190+0.075=0.265m, 하퇴(BR-12) 0.190+0.055=0.245m 로 넷 다 가정값 0.1m 를 넘어 축
방향으로는 비보수적입니다(목 BR-03 만 0.040+0.055=0.095m 로 가정값 아래라
보수적입니다). 박스 부위도 모서리 방향(반폭 세 값의 대각 거리)에서 비보수적입니다 —
그중 최대인 등/어깨(BR-04)가 0.233m(오차 0.133m)입니다. **네 캡슐 부위의 축 방향
오차가 모두 이 박스 최댓값을 넘고, 그중 대퇴(BR-11)의 오차 0.265-0.1=0.165m 가 전체
부위 가운데 가장 큽니다** — 이 근사가 가장 크게 틀리는 방향은 박스 모서리가 아니라
캡슐 축입니다. 이 0.165m 는 `anthropometry.yaml` 치수로 계산한 기하 상한이며, Isaac
Sim 에서 별도로 측정한 값이 아닙니다 — `safety.separation_required` 의 전형적인 값
(2026-08-25 실측 중앙값 약 1.10m; `intrusion_distance` 가 아직 잠정치입니다 — §9.3)에 견주면 작지 않습니다.

**`0.0` 은 "거의 닿음"이 아니라 "겹침, 깊이 미상"입니다.** `min_distance_human`
(`mdp/safety.py`)은 `max(surface_distance.min(), 0.0)` 로 0 에서 잘립니다 — 로봇과
인체가 실제로 겹치면 겹침 깊이(음수)를 버리고 0을 기록하므로, 배포 데이터에서 `0.0`
을 보는 것은 침투량이 아니라 "겹쳤다"는 사실만 알려줍니다.

**자산에서 부위별 실측 치수를 읽어 오는 방식이 더 나은 형태입니다** — `anthropometry.yaml`
이 이미 부위별 형상·치수를 갖고 있으므로 새로 측정할 필요는 없고, `min_distance_human`
(`mdp/safety.py`)이 반경 하나 대신 부위별 형상을 받도록 바꾸는 일만 남습니다. `S1a` · `S1d` 에서는
구현하지 않았고, 판정은 이 근사 거리를 그대로 씁니다(부위 형상을 쓰는 것은 아래의 `R-TS15066-SSM-02` 시나리오).

**`R-TS15066-SSM-02` 를 선언한 시나리오(`S1e` · `S1f` · `S1g`)는 로봇 표면과 인형 표면 사이의 거리를 싣습니다.** 로봇 쪽은
링크마다 렌더 메시를 감싼 상자(`generation/robots/unitree_h1_link_boxes.json`)와, 놓기 전까지는 든 상자이고, 사람 쪽은 인형 부위의
구 · 캡슐 · 상자(`generation/dummy/anthropometry.yaml`)와 두 발(`generation/dummy/joints.yaml`)을 부위마다 그 몸의 자세에 놓은
것입니다. 그 사이 가장 가까운 두 점의 거리를 반복 근사가 아니라 닫힌 식으로 풉니다(`generation/hri_geometry.py`). **링크 상자는
메시를 감싸 실제 링크보다 크므로 이 거리는 실제보다 짧게 — 엄한 쪽으로 — 나옵니다.** 두 표면이 겹치면 0 이고, 위와 같이 0 은
겹침 깊이가 아니라 겹쳤다는 사실만 말합니다. 그 두 점이 다가가는 속도 `safety.approach_speed_robot` ·
`safety.approach_speed_human` 을 재는 자리입니다(아래 「`safety.separation_required` 산출 방법」).

### `safety.separation_required` 산출 방법

`R-TS15066-SSM-01` 을 선언한 시나리오(`S1a` · `S1d`)의 산식은 §9 의 `R-TS15066-SSM-01.parameters` 여섯 항을 그대로 더한 것이며(`mdp/safety.py`
의 `separation_required`), 표준의 수치를 여기 복제하지 않습니다. 다만 산식에 넣는
**입력 두 개는 표준이 요구하는 순간값이 아니라 근사입니다** — 둘 다 방어 가능한 선택이지만
문서화되지 않은 채로 두면 재현할 수 없습니다.

- `robot_speed` 는 에피소드 시작 시 뽑힌 **명목 명령 속도**이며, SSM 감속이 실제로 로봇을
  세운 뒤에도 에피소드 내내 바뀌지 않습니다. 즉 로봇이 정지해 있는 스텝에서도 요구
  이격거리는 전속 주행 기준으로 계산됩니다 — 실제 로봇 속도보다 항상 크거나 같으므로
  보수적(안전) 쪽 근사입니다.
- `human_speed` 는 매 스텝 `meta/rules.json` 의 `R-TS15066-SSM-01.parameters.human_speed`
  (KS B ISO 13855 가 정하는 접근속도 파라미터 — §9.3)에서 오며, 그 에피소드의 `human_entry_speed` 랜덤화 축
  (0.4~1.6 m/s, 카탈로그 §4.5)이 뽑은 실제 진입 속도를 반영하지 않습니다. 카탈로그
  §4.5 는 이 축의 목적을 "SSM 요구 이격거리 변동"이라고 명시하는데, 지금 산식에서는
  이 축이 `separation_required` 에 아무 영향을 주지 못합니다 — 표준이 `human_speed` 를
  이동 속도 기준값(고정 상한)으로 쓰라고 규정하는 것과 일치한다는 해석도 가능하지만,
  카탈로그의 서술과 코드의 동작 중 하나는 틀렸다는 뜻이기도 합니다.

**산식은 그대로 둡니다.** `safety.separation_violation` 이
`R-TS15066-SSM-01` 을 거쳐 `near_miss` 라벨로 이어지므로, 이 근사가 라벨의 근거에
영향을 줍니다 — 어느 쪽이 맞는 해석인지는 카탈로그 §4.5 와 함께 정할 일로
남아 있습니다.

**`R-TS15066-SSM-02` 를 선언한 시나리오(`S1e` · `S1f` · `S1g`)는 방향을 따진 S_p 를 싣습니다.** 표준 5.5.4.2.3
식(1)을 시뮬레이터가 아는 값으로 읽은 것입니다.

S_p = v_h (T_r + T_s) + v_r T_r + S_s′ + C + Z_r + Z_d

- v_r · v_h = `safety.approach_speed_robot` · `safety.approach_speed_human` — 가장 가까운 두 점(로봇 쪽은 든 상자 포함 —
  위 「`safety.min_distance_human` 산출 방법」)을 잇는 방향으로 서로 다가가는 속도. 점의 속도는 그 점이 붙은 링크 · 든 상자 ·
  인형 부위의 선속도에 회전(ω × r)을 더한 것이고, 0 아래는 0. 두 점이 겹친 스텝(닿은 스텝)은 두 점 사이에 방향이 없어 두 점이
  마지막으로 떨어져 있던 스텝의 방향으로 셈 — 여러 스텝 이어 닿으면 그 방향이 이어지고, 첫 스텝부터 닿았으면 로봇 뿌리에서 사람
  골반으로. 겹친 두 점에서 둘 다 0 으로 두면 모든 접촉이 다가가지 않던 로봇의 접촉으로 읽힙니다
- T_s · S_s = 규칙의 `stop_table`(§9.1) 을 로봇의 **실제** 속도(`observation.root_vel` 의 수평 크기 · 돌기는 그 요 각속도)로 선형 보간한 값 —
  0 에서 0 이고 표의 마지막 속도 위는 마지막 행. 든 상자의 무게로 상태를 고릅니다(그 무게 이상인 가장 가벼운 상태).
  **표를 위에서 감싼 값으로 읽습니다** — 행마다 그 속도 이하의 행 가운데 큰 정지 시간 · 정지 거리로 올린 뒤 보간하고,
  고른 상태와 그보다 가벼운 상태 가운데 큰 값을 씁니다(시간과 거리를 따로). 그래서 로봇이 빨라지거나 상자가 무거워져도
  T_s · S_s 가 줄지 않습니다. 잰 표의 행은 걸음 위상 다섯 가운데 가장 나쁜 값을 0.02 s 단위로 잰 것이라 행과 상태 사이에서
  늘 커지지는 않습니다(상자 2 kg 걷기는 1.0 m/s 에서 1.14 s · 1.2 m/s 에서 1.10 s)
- S_s′ = S_s × min(1, v_r / 로봇 속도) — 정지거리 가운데 사람 쪽으로 가는 몫. 돌기는 상자 모서리가 지나간 거리 ×
  min(1, v_r / (요 각속도 × `turning_reach_m`)). 걸으며 돌 때는 걷기와 돌기로 각각 셈한 S_p 가운데 큰 쪽
- T_r · C · Z_r · Z_d 와 다가감 문턱 `approach_speed_threshold` = 규칙의 `parameters`(자체 설정)

생성 중에 셈해 열로 싣고 로봇에게는 넘기지 않습니다 — 기록 데이터의 로봇은 사람에 반응하지 않습니다. 산식은
`generation/directed_separation.py` 이고, 받는 쪽은 다가가는 속도 두 열과 자기 로봇의 정지 시험 표로 S_p 를 다시 셀 수
있습니다 — 로봇의 속도는 `observation.root_vel`(뿌리 바디의 무게중심 속도)의 수평 크기와 여섯째 성분, 든 상자의 무게는
`notes.randomization.condition.box_mass_kg` 이고, 상자는 그 에피소드의 첫 `contact_onset` 스텝까지(그 스텝 포함) 들려 있으며
그 뒤는 `empty` 입니다. 이 배치가 쓴 파라미터와 표는 `notes.ssm_params` 에 있습니다. `safety.separation_violation` 은 `approach_speed_robot > approach_speed_threshold`(다가감 문턱 — 자체 설정) 이면서
`min_distance_human < separation_required` 인 스텝입니다.

### `observation.contact.force` 의 추적성 한계

**서보 오염은 해소됐습니다.** `S1a` · `S1d` 에서는 로봇이 더미 표면 0.10 m 안에 들어오면 홀드를
끄므로 접촉력에 홀드력이 섞이지 않습니다 — 실측 6 에피소드에서 로봇 접촉 333 스텝 전부
`hold_force == 0`. 조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 거리로 끄지 않고 첫 접촉 스텝에 끄므로 그 한 스텝은
홀드가 실린 채 물리가 돌고 `notes.hold_contaminated_steps` 에 세어지며, 판정은 그 스텝도 읽습니다(§6 「조건부 필드 — 사람과
로봇」 · §9.1). 접촉 유형 경계도 구현했습니다(§8.1). **그래서 크기를
표 A.2 한계와 비교합니다.** `S1d` 에서는 붙잡는 동안 서보가 더미의 무게까지 듭니다(발이
바닥에서 5 mm 뜸 — 에피소드 메타 `randomization.human_hold_clearance_m`). 그래서 붙잡힌 스텝의 홀드력에는
더미의 무게가 실리고 그동안 더미 부위에는 지면 하중이 없습니다. 홀드를 끄는 조건과 그 뒤의 `hold_force == 0`
은 같습니다.

**남은 한계는 다른 것입니다** — `S1a` · `S1d` 에서는 접촉 유형이 분류되지 않은 접촉은 비교 대상이 되지
않습니다. 로봇 발이 넘어진 더미를 밟는 접촉이 여기 해당하며(협착 사례가 이 경로로
들어옵니다), 그런 접촉이 있는 에피소드는 라벨의 `not_evaluated` 에 그 사실과 스텝이
남습니다. 조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 그 접촉도 유형을 매기고(§6 「조건부 필드 — 사람과 로봇」),
유형을 가르지 못한 접촉은 엄한 쪽 한계로 비교해 `not_evaluated` 에 남기지 않습니다(§9.1). 사정은
`generation/README.md` 를 참고하십시오.

### `observation.contact.force` 는 순간값이 아니라 서브스텝 최댓값입니다

기록 주기는 50 Hz 이고 물리 주기는 200 Hz 이므로, 한 기록 스텝 안에 물리 서브스텝이
넷 있습니다. **기록 시점의 값만 읽으면 그 사이에서 생겼다 사라진 접촉은 짧게 기록되는
것이 아니라 정확히 0 이 되어 데이터에서 사라집니다.** 근접사고와 실패를 담는 것이 이
데이터셋의 존재 이유이므로 그대로 둘 수 없는 성질입니다.

그래서 `observation.contact.force` 는 **그 기록 스텝의 서브스텝 최댓값**입니다.
기록 주기를 올려서 얻은 것이 아니라 표집 시점을 바꿔서 얻은 것이므로 저장 용량은
그대로입니다 (`generation/envs/s1a/substep_contact.py`).

실측(400 스텝, `check_substep_gain.py`): 스냅샷이 0 인데 서브스텝 최댓값은 0 이 아닌
경우가 인체 센서에서 **121 건**(최대 12,216 N)·로봇 센서에서 6 건, 둘 다 0 이 아니지만
스냅샷이 작게 적히던 경우가 478 건·345 건(최대 190.8배·12.8배)이었습니다.

창은 **그 기록 스텝의 물리 서브스텝 넷**이며, 직전 스텝의 것은 섞지 않습니다. 훅이
읽는 값이 직전 서브스텝 결과라 다섯을 담으면 직전 스텝의
마지막 서브스텝에만 있던 접촉이 이 스텝에도 기록되어 지속시간이 한 스텝 늘어납니다
(실측: 스텝의 첫 표본이 직전 스텝의 마지막 표본과 119/119 = 100.0% 일치).

### `observation.contact.v_rel` 의 산출 방법과 한계

접촉 지점의 상대속도이며, 표 A.4 의 전달 에너지 한계와 식 (A.6) 의 속도 한계가 모두
이 값을 입력으로 씁니다. **첫 공개 배포본은 두 축을 쓰지 않습니다** — `meta/rules.json` 이
`unused_in_v0_1` 로 사유를 적고 있고, 라벨은 힘 축으로만 판정합니다. 이 필드는 그
축을 열 때 필요한 입력을 지금부터 기록해 두는 것입니다.

**근사입니다.** PhysX 의 접촉점은 이 파이프라인에서 얻을 수 없습니다
(`track_contact_points` 가 필터 표현식을 요구하고, 세 변형 모두 실패 —
`check_contact_filter.py`). 그래서 다음과 같이 산출합니다.

- 각 인체 부위 바디에 대해 **중심 거리가 가장 가까운 로봇 바디**를 짝으로 잡습니다
- 두 바디의 속도를 각자의 중심에서 상대 쪽으로 떨어진 점에서 평가합니다
  — `v = v_com + ω × (p − p_com)`
- 그 점까지의 **지렛대 길이는 인체 부위의 최대 반치수로 제한**합니다
  (`generation/dummy/anthropometry.yaml` 에서 유도, 현재 0.265 m)
- 부위는 최댓값으로 축약합니다. 힘과 같은 이유입니다

**지렛대 상한은 조절 손잡이가 아닙니다.** 강체의 속도장은 그 강체 안에서만 뜻이
있습니다. 상한 없이 돌린 `S1a` 배치에서 1 m/s 로 걷는 로봇이 중앙값 9 m/s · 최대
38 m/s 를 냈습니다(실측) — 흔들리는 팔의 회전을 몇 미터 떨어진 중점까지 뻗은 값입니다.
실제로 닿는 거리에서는 중심 거리의 절반이 상한보다 작으므로 상한이 하는 일이 없고,
멀리 떨어져 있을 때만 작동합니다. 그 구간에는 접촉점이라는 것이 존재하지 않습니다.

**힘과 같은 시점의 값이 아닙니다.** 힘은 서브스텝 최댓값이고 이 값은 기록 시점의
값입니다. 에너지 비교를 하려면 둘을 같은 시점에서 표집해야 하며, 첫 공개 배포본은 그 비교를
하지 않습니다.

**특히 서브스텝 표집이 새로 살린 짧은 접촉에는 같은 시점의 `v_rel` 이 없습니다.**
그 접촉이야말로 에너지 축이 필요한 사례이므로, 축을 여는 작업은 두 값을 같은
서브스텝에서 표집하는 것부터입니다.

**부위별 힘·높이·로봇 접촉 표식은 같은 서브스텝에서 옵니다.** 접촉 유형(§8.1)이
어느 표 A.2 열을 쓸지 정하는데, 셋을 따로 축약하면 한 서브스텝의 로봇 접촉이 다른
서브스텝의 최대 힘을 유형 지을 수 있습니다. 부위의 최대 힘을 낸 바디가 자기 높이와
자기 접촉 표식을 함께 넘깁니다 (`reduce_region_peaks`).

### `action` 은 정책의 출력이 아니라 관절 목표입니다

정책이 내보내는 것은 **정규화된 벡터**이고, 액션 항이 그것을 `원값 × scale + 기본자세`
로 바꾼 값이 실제로 관절에 걸리는 위치 목표입니다(H1 은 `scale=0.5`,
`use_default_offset=True`). 배포본의 `action` 은 **뒤쪽**, 즉 라디안 단위의 목표입니다.

**앞쪽을 실으면 열 이름이 거짓말을 합니다** — 정규화 값에는 라디안이라는 단위가 없고,
받는 쪽은 규격서를 보고 라디안으로 읽습니다.

정책 학습용으로 원값이 필요하면 `(action - 기본자세) / scale` 로 되돌릴 수 있으나,
기본자세와 scale 은 배포본에 싣지 않으므로 자산과 환경 설정이 필요합니다. 첫 공개 배포본은
안전성 데이터셋이므로 물리 단위 쪽을 싣습니다.

`action_space` 가 어느 쪽인지 선언합니다 — `position_rad` 또는 `torque_nm`.

### 선택 필드 — `observation.contact.pressure`

**접촉 면적을 실측할 수 있을 때만 싣습니다.** 압력 매트·F/T 배열로 면적을 얻는
실장비 데이터가 여기 해당하며, 첫 공개 배포본의 시뮬레이션 데이터에는 없습니다 — PhysX 는 힘을
보고하고 면적을 보고하지 않습니다.

**부위별 공칭 면적을 가정해 힘을 나누는 방법은 쓰지 않습니다.** 그 값은 실측이 아닌데
공개 데이터셋의 이용자는 실측으로 읽습니다.

**없는 것은 한계가 아니라 좌변입니다.** `meta/rules.json` 의 압력 한계 12개에는 값이
실려 있습니다. 비교가 성립하지 않는 것은 비교당할 값을 만들 수 없기 때문입니다.
정본이 어디까지 채워졌는지는 **§9.3 한 곳에서만** 말합니다 — 따로 적으면 정본이 채워질 때
그 자리가 정본과 반대를 말한 채 남습니다.

`observation.contact.force` 는 필수로 남습니다 — 접촉의 발생·부위·타이밍은 압력 없이도
기록됩니다.

### 조건부 필드 — `observation.object.*`

파지물을 다루는 시나리오(S4 낙하 등)에서만 기록합니다. 보행만 하는 S1a 같은 시나리오에는
없어도 됩니다. 대신 **그 시나리오에 낙하 규칙을 적용했다고 선언할 수 없습니다** —
검증기가 `scenarios.json` 의 적용 규칙과 데이터의 컬럼을 대조합니다 (§10).

`R-OBJ-DROP-01` 은 `grasped` 가 `true`에서 `false`로 바뀐 시점의 `position[2]` 를 기준으로
그 뒤 하강량이 `drop_height` 를 넘는지로 판정합니다. 회전은 판정에 쓰지 않으므로 담지 않습니다.

### 조건부 필드 — 카메라

**배포본에 `meta/cameras.json` 이 있을 때만 필수이고, 있으면 전부 실립니다.** 없으면 하나도
없습니다. 26.11 배포본에는 있습니다. 기계가독 정본은 `meta/safety_schema.json` 의
`conditional_required_step_fields` 이며 키 `cameras` 가 이 조건입니다. 거기 적힌 열다섯 중
`observation.images.*` 일곱은 LeRobot 영상 피처로, `observation.bbox2d.human` 과
`observation.camera_pose.*` 일곱은 parquet 열로 실리고, 여덟 열의 길이는 `fixed_shapes` 에 있습니다.
카메라 하나마다 영상 하나와 자세 하나이며 순서는 `meta/cameras.json` 의 카메라 순서입니다(일곱 대 —
머리 둘 · 양 손목 · 머리 뒤쪽 · 3인칭 · CCTV).

**깊이와 분할 마스크는 싣지 않습니다.** 분할은 녹화 때 `observation.bbox2d.human` 을
계산하는 데만 쓰고 저장하지 않으며, 깊이는 렌더하지 않습니다. 안전 판정이 쓰는 로봇–사람 거리는
`safety.min_distance_human` 에 그대로 있습니다. 부위별 픽셀 정답과, 깊이로 거리 추정을 검증하는
쓰임은 지원하지 않습니다.

**팩 코드는 남아 있고 `meta/cameras.json` 의 `packs` 가 켜고 끕니다.** 정본이 팩을 선언하면
에피소드마다 npz 하나에 `depth` · `mask` · `step_index` 가 함께 실리고 `meta/packs.json` 이 목록을
적습니다(§4). 형식은 그 선언 `packs.<name>` 이 적습니다 — 프레임 크기 (H, W) 는 `downsample` 의
`height` · `width`, `mask` 의 클래스 번호는 `mask.classes` 이고, 몇 스텝마다 한 프레임을 싣는지는
`downsample.every_n_steps` 입니다. 그래서 팩의 각 프레임이 에피소드의 몇 번째 스텝인지를
`step_index` 가 적어 정렬을 파일 스스로 증명합니다. `LeRobotDataset` 은 팩을 읽지 않으므로
`numpy.load` 로 엽니다. 선언하지 않았는데 `packs/` 나 `meta/packs.json` 이 있으면 검증기가 오류를
냅니다(§10 항목 14).

**팩을 선언해도 깊이는 LeRobot 깊이 영상이 아니라 팩입니다 — 무효값 때문입니다 (2026-09-26 실측).**
LeRobot 0.6.1 의 12비트 깊이 영상은 무손실로 왕복하지만 양자화기에 무효 코드가 없어, 팩의
`0`(광선이 아무것도 맞히지 않음)이 설정한 최소 거리로 돌아옵니다. 「카메라 바로 앞의 물체」로
읽히는 쪽이라 안전성 데이터에서 가장 나쁜 방향의 오독이고, 형식은 무효를 적을 수 있어야 합니다.

**`observation.camera_pose.*` 는 매 프레임 싣습니다.** 일곱 중 다섯(머리 둘 · 양 손목 · 머리 뒤쪽)은 로봇 링크에
붙어 함께 움직이므로 이것이 없으면 관절 기구학을 다시 풀어야 영상과 월드 좌표를 잇습니다 — 부모 링크의
자세에 `meta/cameras.json` 의 장착 오프셋을 합성한 값입니다. 3인칭은 로봇을 따라 옮긴 자리이고,
CCTV 는 창고에 고정되어 매 프레임 같은 값입니다(자세는 그 시나리오의 자리마다 `meta/cameras.json` 의 `sites` 에서 유도 — S1d 는 통로 옆 랙 면 규칙과 씬 명세에서). `observation.root_pose` 와
같은 7원소 표현이고 축은 `meta/cameras.json` 의 내부 파라미터와 같은 광학 규약이라 픽셀 투영에
바로 씁니다. 상자 좌표의 규약(양끝 포함)도 그 파일의 `bbox.coords` 가 적습니다.

### 조건부 필드 — 사람과 로봇

**배포본의 시나리오가 `R-TS15066-SSM-02` 를 선언할 때만 필수이고, 있으면 전부 parquet 열로 실립니다**(`S1e` ·
`S1f` · `S1g`). 기계가독 정본은 `meta/safety_schema.json` 의 `conditional_required_step_fields` 이며 키 `hri` 가 이
조건입니다. 길이는 `fixed_shapes` 에 있습니다.

- `observation.command_velocity` — 보행 정책에 준 걷기 명령. 기록 데이터의 로봇은 과제의 명령을 그대로 받습니다
- `observation.human.root_pose` · `observation.human.root_vel` — 사람(인형) 골반 바디(BR-07)의 월드 자세와 그 원점의 속도.
  `observation.root_pose` · `observation.root_vel` 과 같은 표현
- `observation.human_visible_fraction.<카메라>` — 로봇 몸의 카메라마다 사람 몸의 표본점 가운데 시야 안에 있고
  가려지지 않은 몫(0~1). 렌더가 아니라 기하로 셈하므로 카메라를 끄고 돌린 배치에도 같은 값이 실립니다. 「봄」의
  문턱(한 카메라에서 20 % 이상 · 전신 높이 50 px 이상 — 정의는 위 표의 `head_rgb` 행)은 자체 설정이고, 받는 쪽은 이 열로 다른 문턱을 정할 수 있습니다.
  걷고 도는 동안 팔이 흔들려 손목 카메라가 든 상자 안에 든 스텝은 그 카메라만 상자를 빼고 셉니다(그 스텝의 그림에는 상자 속이 보임 —
  상자 밖의 카메라는 그대로 상자에 가려짐). 카메라마다 그런 스텝이 몇 개였는지는 `notes.camera_in_box_steps` 에 있습니다. 26.11 배포본의 297건(세 시나리오 모두 로봇이 상자를 듦)에서는 카메라 다섯 모두 그런 스텝이 0 입니다. 잡은 자세에서는 어느 카메라도 상자 안에 있지 않습니다(`generation/carry.py`)
- `safety.approach_speed_robot` · `safety.approach_speed_human` — 위 「`safety.separation_required` 산출 방법」의 v_r · v_h

**이 시나리오들의 접촉 열은 로봇과 든 상자가 사람에 준 힘만 싣습니다.** `observation.contact.force` · `observation.contact.type` ·
`safety.event` 의 `contact_onset` 은 로봇 링크와 든 상자(놓은 뒤에도)가 인형에 준 힘만 셉니다. 인형의 접촉 센서를 로봇 링크마다 ·
상자 하나로 거른 힘 행렬(인형의 바디 × 거른 상대)을 물리 서브스텝마다 읽어, 바디마다 그 열들을 더한 힘의 크기와 한 열(로봇
링크 하나나 상자)이 준 힘의 크기 가운데 가장 큰 값을 그 스텝의 서브스텝 최댓값으로 싣습니다 — 같은 쪽으로 미는 부위들은
더해지고, 두 부위가 한 바디를 양쪽에서 누르는 끼임(든 상자와 아래팔 사이에 낀 손 등)은 서로 지워지지 않고 한쪽이 누르는 힘으로
남습니다. 바닥 · 랙과의 힘은 행렬에 들지 않아 빠집니다(바닥에 선 인형의 발 · 넘어져 누운 인형 · 랙에
밀린 로봇이 행렬에 아무것도 남기지 않고, 로봇이나 상자만 닿았을 때 바디마다 행렬의 합이 그 바디의 알짜 힘과 같음을
2026-10-20 에 Isaac Sim 6.0.1 · Isaac Lab 3.0.0 으로 확인). 유형은 부위마다 그 몫과 그 몫이 가장 컸던 서브스텝의 부위 높이로
매기므로(§8.1) 로봇 발이 넘어진 인형을 바닥에 누르는 접촉도 매겨집니다. `contact_onset` 은 인형의 발을 포함한 어느 바디든
이 힘을 받기 시작한 스텝이고(로봇 링크 하나나 상자 하나만 눌러도 — 끼임도 셈) 든 상자는 그 에피소드의 첫 `contact_onset` 스텝에
놓습니다. 인형을 세워 두는 제어기(붙잡기)도 같은 스텝에 같은 문턱으로 풀리고 거리로 미리 풀지 않습니다(자체 설정 — 미리 풀면
닿지 않고 스치는 근접사고에서 균형 제어가 없는 인형이 보조 없이 걷다 닿기 전에 기울거나 넘어짐). 그래서 닿지 않은 에피소드의
`notes.hold_release_step` 은 `-1` 이고, 닿은 에피소드는 첫 `contact_onset` 스텝이 붙잡는 힘이 실린 채 물리가 돈 스텝이라
`notes.hold_contaminated_steps` 가 1, `notes.hold_release_step`(붙잡는 힘 없이 물리가 돈 첫 스텝)은 그 다음 스텝입니다 — 두
값은 그 스텝에 붙잡는 힘이 있었다는 사실을 그대로 남깁니다. 라벨은 그 첫 접촉 스텝도 판정합니다: 표 A.2 한계와의 비교와 머리 ·
얼굴의 `S3` 는 첫 `contact_onset` 스텝과 `notes.hold_release_step` 부터의 스텝을 읽습니다(§9.1 — 한 스텝만 닿는 접촉도
비교되도록 · 자체 설정). 인형의 발은 12부위 밖이라
`observation.contact.force` 에는 없습니다. `observation.contact.v_rel` 은 `S1a` · `S1d` 와 같은 근사(위 「`observation.contact.v_rel` 의
산출 방법과 한계」)라 부위마다 가장 가까운 **로봇 링크**와의 상대속도이고 든 상자는 들지 않습니다 — 상자가 닿은 스텝의 값도 링크의
것입니다. 행렬은 법선력만 담습니다(마찰력 없음 — 알짜 힘도 같음)
(`generation/envs/hri/record_logic.py` · `generation/envs/s1a/substep_contact.py`).

**`rack_contact`** 는 로봇의 발 아닌 링크나 **놓기 전의** 든 상자가 사람 아닌 것에서 받은 힘 — 그 링크 · 상자의 알짜 힘에서
사람 몫을 뺀 크기의 서브스텝 최댓값 — 이 5 N(자체 설정)을 넘기 시작한 스텝입니다. 랙과 바닥을 가르지 않으므로 로봇이 넘어져
바닥에 닿아도 섭니다. 놓은 상자는 떨어진 바닥을 세므로 셈하지 않습니다. 결과를 바꾸지 않습니다.

**과제의 완료** — `notes.task_completed_step` 은 로봇 뿌리가 목표 자리에서 0.3 m · 요 20° 안에 있고 수평 뿌리 속도가
0.15 m/s 이하인 스텝이 다섯 번 이어졌을 때 **그 다섯째 스텝**입니다(허용 오차 넷 모두 자체 설정 · `generation/task_path.py`).
시간 안에 그러지 못하면 `-1` 입니다.

### `safety.event` 허용값
`protective_stop_triggered`, `ssm_violation`, `pfl_limit_exceeded`, `mode_switch`,
`contact_onset`, `fall_detected`, `object_dropped`, `comm_timeout`, `sensor_fault_injected`, `rack_contact`

`rack_contact` 는 로봇의 발 아닌 링크나 놓기 전의 든 상자가 사람 아닌 것(랙 · 바닥)에서 받는 힘이 5 N(자체 설정)을 넘기 시작한 스텝입니다(`S1e` · `S1f` · `S1g` — 결과를 바꾸지 않음 · 위 「조건부 필드 — 사람과 로봇」). 이 셋의 `safety.event` 는 `ssm_violation` · `contact_onset` · `rack_contact` 이고
`protective_stop_triggered` 는 없습니다 — 기록 데이터의 로봇은 보호정지를 하지 않습니다.

`contact_onset` 은 접촉이 시작한 스텝입니다. `observation.contact.type` 이 매겨진 스텝(로봇이 사람에 닿은 스텝 — §8.1)은 그 에피소드의 첫 `contact_onset` 과 같거나 뒤에 옵니다 — 라벨의 접촉 판정이 `contact_onset` 을 읽기 때문입니다(검증기 C8).

---

## 7. 관절 규격 — H1 / G1 매핑

### 7.1 절대 원칙
**관절 순서를 코드나 문서에 손으로 적지 않습니다.** 자산에서 직접 읽은 `robot.data.joint_names`를
그대로 기록하고, 매핑 파일도 그 출력으로부터 생성합니다. 수기 입력은 관절 순서 불일치의 주원인입니다.

### 7.2 의미 키(canonical key) 명명 규칙
```
{side}_{chain}_{axis}      side  : left | right | center
                           chain : hip | knee | ankle | waist | torso
                                   shoulder | elbow | wrist
                           axis  : roll | pitch | yaw | (없음)
예) left_hip_pitch, right_ankle_roll, center_waist_yaw, left_shoulder_pitch
```

### 7.3 운동사슬 구성 (구조 비교) — **실측 확정**

Isaac Sim 6.0.1 의 Unitree H1 · G1 자산에서 추출한 값입니다
(`generation/dump_joint_names.py`, 2026-08-18). 구조상 결손 축은 `missing` 필드에 기록됩니다.

| 사슬 | H1 (19DOF) | G1 29DOF | 매핑 |
|---|---|---|---|
| 다리 (좌/우) | 5축 × 2 = 10 | 6축 × 2 = 12 | 부분 일치 — H1 발목 1축, G1 발목 2축(pitch/roll) |
| 허리/몸통 | 1 (`torso`) | 3 (`waist_yaw/roll/pitch`) | 부분 일치 |
| 팔 (좌/우) | 4축 × 2 = 8 | 7축 × 2 = 14 | 부분 일치 — H1 손목 없음, G1 손목 3축 |
| 합계 | **19** | **29** | |

G1 자산의 기본 배리언트는 3지 핸드를 포함해 **43DOF** 로 열립니다. 29DOF 구성은
`left_hand=None right_hand=None` 을 명시해야 하며, 이 선택은 매핑 파일의
`provenance.variants_resolved` 에 기록됩니다.

### 7.3.1 관절(joint)과 DOF 는 다르다

**상태 배열의 인덱스는 joint 가 아니라 DOF 를 따릅니다.** 센서 장착용 고정관절처럼 자유도가
없는 관절은 `observation.state.*` 의 `(dof,)` 배열에 들어가지 않습니다.

- §5 의 `joint_names` 필드 = Isaac 의 **`Articulation.dof_names`**
- `Articulation.joint_names` 는 고정관절을 포함한 상위 집합이며, 고정관절이 있는 자산에서는 상태 배열과 길이가 다름

이 데이터셋이 쓰는 H1 사본(`IsaacLab/Robots/Unitree/H1/h1.usd`)은 관절 19개가 모두 DOF 라
두 목록이 같습니다(매핑 파일의 `articulation_detail.non_dof_joints` 가 비어 있음). 같은 H1 의
다른 사본 `Robots/Unitree/H1/h1.usd` 는 관절 24개 가운데 5개(`d435_left_imager`,
`d435_rgb_module`, `imu`, `logo`, `mid360`)가 고정관절이고, DOF 는 같은 19개가 같은 순서로
나옵니다(자산 라이선스 인벤토리 §2.2).

그런 자산에서 둘을 혼동하면 길이 24 배열을 길이 19 자리에 넣게 되어 C2 가 실패하거나, 개수가
우연히 맞는 자산에서는 **순서만 어긋난 채로 통과**합니다. 매핑 파일은 전자를 `joint_names` 로,
후자를 `articulation_detail.joint_names_all` 로 분리해 기록합니다.

### 7.4 매핑 파일 형식 (`meta/joint_map/{robot_model}.yaml`)

전체 형식은 실제 생성물을 보십시오 — 이 문서는 필드의 의미만 정의합니다.

| 필드 | 의미 |
|---|---|
| `joint_names` | **정본.** `Articulation.dof_names` 순서 그대로. `(dof,)` 배열의 유일한 해석 근거 |
| `canonical_map` | 네이티브 이름 → §7.2 의미 키. 이름에서 규칙적으로 유도 |
| `unmapped` | 의미 키로 해석되지 않은 관절. 비어 있어야 정상 |
| `needs_review` | 이름만으로 회전축을 판별하지 못한 관절. 자산의 회전축을 보고 확정 |
| `conflicts` | 서로 다른 관절이 같은 의미 키로 떨어진 경우. 비어 있어야 정상 |
| `missing` | 다른 로봇에는 있고 이 로봇에는 없는 의미 키 |
| `provenance` | 자산 경로·SHA256, Isaac Sim 버전, 물리엔진, 해석된 배리언트 |
| `articulation_detail` | 고정관절 포함 전체 관절 목록 (참고용, §7.3.1) |
| `dof_limits` | 관절별 [하한, 상한]. 회전 rad / 병진 m. 검증기 C8 의 범위 검사 근거 |

`provenance.source` 는 `isaacsim_physx_tensor_dof_order` 입니다. USD 파일을 파싱한 순서가
아니라 **물리엔진이 아티큘레이션을 파싱한 뒤 보고하는 순서**이며, 둘은 다를 수 있습니다.
그래서 추출 도구는 타임라인을 재생해 물리 텐서 뷰가 초기화된 뒤에 읽습니다.

`missing` 필드가 핵심입니다. G1에 있고 H1에 없는 축(손목·발목 roll·허리 2축)을 명시해 두면,
두 로봇의 데이터를 함께 학습할 때 **결손을 패딩이 아니라 마스크로 처리**할 수 있습니다.

---

## 8. 인체 더미 규격 및 접촉 부위 코드

`keti_dummy_ts15066_v1` — ISO/TS 15066 **표 A.2** 의 신체 부위 구분을 따라 자체 제작한
더미입니다. 라이선스 제약이 없고,
**ISO/TS 15066 Table A.2 기준의 능동형 실물 더미와 같은 부위 구분**이라 그런 더미와 sim-to-real 대조가 직접 가능합니다.

**생체역학 파라미터는 이 더미의 동역학에 쓰이지 않습니다.** 유효질량·유효탄성계수의
출처는 표 A.2 가 아니라 **표 A.3** 이고(§9.2), 그 둘은 두 물체 충돌 모형(식 A.1)의
파라미터이지 자유 물체의 질량이 아닙니다. 생성기는 두 값이 정본에 채워져 있을 것을
요구할 뿐 값을 읽지 않습니다(`generation/dummy/build_dummy.py`) — 더미는 강체 질량을
명시하지 않아 물리 엔진이 형상과 기본 밀도로 계산합니다. **기록된 접촉력과 낙상 거동을
표준 생체역학 모형의 산출로 읽어서는 안 됩니다.** 그 둘을 강체 질량으로 쓰면
12부위 합이 인체가 아닌 크기가 됩니다.

부위 코드는 `observation.contact.region_names`에 사용합니다.

```
BR-01 두개골/이마      BR-02 얼굴          BR-03 목
BR-04 등/어깨          BR-05 흉부          BR-06 복부
BR-07 골반             BR-08 상완/팔꿈치   BR-09 전완/손목
BR-10 손/손가락        BR-11 대퇴/무릎     BR-12 하퇴
```

각 부위의 힘·압력 한계값은 `meta/rules.json`에 표준 인용과 함께 수록하며,
**규격서 본문에 수치를 복제하지 않습니다**(표준 개정 시 단일 지점만 갱신).

### 8.1 접촉 유형 — 같은 하중, 다른 한계

Table A.2 는 **과도적 접촉(transient)** 과 **준정적 접촉(quasi-static)** 의 한계를 따로 규정합니다.
인체가 후퇴할 수 있는 자유 충돌과, 협착처럼 후퇴할 수 없는 상태는 같은 하중이라도 위해 정도가 다르기 때문입니다.

따라서 접촉력만으로는 판정할 수 없고, **어느 쪽 상황이었는지를 데이터가 함께 말해야** 합니다.
`observation.contact.type` 이 그 역할을 하며, 부위별 배열이라 한 스텝에서 부위마다 다른 유형을
가질 수 있습니다 (예: S2 에서 한 팔은 협착 상태, 다른 부위는 자유 충돌).

| 값 | 의미 | 주로 나타나는 시나리오 |
|---|---|---|
| `none` | 해당 부위에 접촉 없음 | 전 시나리오 |
| `transient` | 과도적 접촉 — 인체가 후퇴 가능 | S1 (충돌) |
| `quasistatic` | 준정적 접촉 — 인체가 후퇴 불가 | S2 (협착) |

#### 산출 방법 (S1a, 2026-08-21)

구분 기준은 **지속시간이 아니라 후퇴 가능 여부**입니다. 넘어진 인체라도 밀려날 수
있으면 `transient` 이고, 접촉이 몇 스텝을 이어가든 달라지지 않습니다.

부위마다·스텝마다 이렇게 정합니다 (`generation/envs/s1a/mdp/safety.py`
`classify_contact_type`).

1. **로봇이 인체를 건드리고 있지 않으면 전 부위 `none`.** 판단은 로봇 쪽 채널(로봇의
   발이 아닌 링크가 하중을 받는지)로 합니다. 이때 부위에 실린 힘은 인체 자신의 접지
   하중이므로 Table A.2 한계와 비교할 대상이 아닙니다
2. 로봇 접촉 중, 부위 접촉력이 잡음 문턱 이하면 그 부위는 `none`
3. 남은 부위 중 **바닥에 눌린 것이 `quasistatic`**, 나머지가 `transient`

S1a 가 만들 수 있는 협착은 하나뿐입니다 — **넘어진 인체를 로봇이 지면 쪽으로 누름**.
그래서 "눌림" 은 부위 높이로 판정하며, 임계값은 `recorder.py` 의 `_PIN_HEIGHT_M`
하나입니다(값과 실측 근거를 그 자리에 둡니다 — 표준값이 아니라 프로젝트 실측
상수이므로 `meta/rules.json` 에 넣지 않습니다). 4 에피소드 실측에서 서 있는 부위와
정착한 넘어진 부위의 높이 분포는 **겹치지 않았고**, 임계값은 그 빈 구간 안에 있습니다.

속도로 판정하지 않는 이유: 자유 충돌도 반전하는 순간 속도가 0 을 지나므로 지속시간
임계값이 또 필요해지고, 정본이 묻는 것은 "움직였는가" 가 아니라 "움직일 수 있는가"
입니다.

> **`S1a` · `S1d` 에서는 이 필드를 아직 Table A.2 열 선택에 그대로 쓸 수 없습니다 (2026-08-21).**
> 결함 두 가지가 남아 있고 둘 다 원인이 하나입니다 — **부위별 힘이 어느 상대와의
> 접촉인지 모릅니다.**
>
> 1. **로봇 발이 넘어진 인체를 밟는 접촉은 아예 유형이 매겨지지 않습니다.** 위 1번의
>    게이트가 로봇의 발을 제외하기 때문입니다(발은 매 스텝 지면 하중을 받으므로 넣으면
>    보행 전 스텝이 인체 접촉이 됩니다). **`quasistatic` 이라는 값이 존재하는 이유가
>    정확히 그 상황**이므로, 남은 구멍 중 가장 중요합니다
> 2. **게이트가 켜진 동안에는 지면 하중만 받는 부위에도 유형이 매겨집니다.** 인체가
>    서 있을 때는 BR-12 만 지면에 닿아 문제가 없지만, 넘어진 뒤에는 다른 부위도 지면
>    하중을 받습니다. 로봇이 건드린 적 없는 부위가 Table A.2 한계와 비교될 수 있습니다
>
> **이 센서 구성에서는 쌍 단위 귀속이 되지 않습니다** (`generation/envs/s1a/check_contact_filter.py`,
> 2026-08-21). `filter_prim_paths_expr` 를 세 가지로 지정하면, 로봇 루트와
> 토르소는 센서가 붙고 모양도 나오지만 700 스텝에서 로봇 접촉을 0건 보고했고,
> 링크 글로브는 PhysX 가 `"did not match any rigid contact for filters"` 로 생성을
> 거부했습니다. 필터가 붙는 것과 값이 나오는 것은 별개입니다.
>
> 그때까지 이 필드는 **로봇 비-발 링크가 하중을 받는 동안의 근사**로 읽어야 하며,
> 에피소드 메타의 `contact_type_unresolved_reason` 에 같은 내용을 남깁니다. 조건표가 있는 시나리오(`S1e` · `S1f` ·
> `S1g`)는 로봇 링크마다 거른 힘 행렬로 상대를 가립니다(§6 「조건부 필드 — 사람과 로봇」).

---

## 9. 판정 규칙 (`meta/rules.json`)

라벨은 사람이 붙이지 않고 규칙으로 자동 산출합니다.

### 9.1 규칙 목록 (정본 총 20개)

> **배포본은 이 20개를 전부 싣지 않습니다.** 담고 있는 시나리오가 선언한 규칙만
> 싣고, 뺀 것은 `excluded_rules` 에 이름과 사유로 남깁니다 (§9.5).

| rule_id | 조건 | 결과 |
|---|---|---|
| `R-TS15066-A2-01` … `-12` | 부위 `BR-01`~`BR-12` 의 접촉력 또는 압력이 Table A.2 한계 초과 | `fail` / severity ≥ S2 |
| `R-TS15066-SSM-01` | `min_distance_human < separation_required` 이면서 접촉 없음 | `near_miss` |
| `R-TS15066-SSM-02` | 접촉 없이, 로봇이 사람 쪽으로 다가가는 동안(`approach_speed_robot > approach_speed_threshold` — 문턱은 자체 설정) `min_distance_human < separation_required` — `separation_required` 는 방향을 따진 S_p(§6) | `near_miss` |
| `R-HRI-CONTACT-01` | 로봇 몸이나 든 상자가 사람에 닿음 — 힘과 무관(`contact_onset`) | `fail` / severity S2·S3 |
| `R-25785-STAB-01` | `stability_margin < 0` 이 `min_duration` 이상 지속 → 전도 성립 | `fail` (hazard=fall) |
| `R-25785-STAB-02` | `stability_margin < epsilon` 후 회복 | `near_miss` |
| `R-10218-STOP-01` | 보호정지 조건 성립 후 `max_stop_time` 내 미정지 | `fail` |
| `R-22440-CTRL-01` | 센서 결함 주입 시 제어 출력이 허용 범위 이탈 | `fail` (hazard=malfunction) |
| `R-OBJ-DROP-01` | 파지물 낙하 검출 | `fail` (hazard=drop) |

**부위별 규칙을 12개로 전개**합니다. `R-TS15066-A2-*` 처럼 와일드카드로 두면
`R-TS15066-A2-99` 같이 존재하지 않는 부위 코드가 검증을 통과하기 때문입니다.
검증기는 rules.json 에 와일드카드가 남아 있으면 경고합니다.

접촉 규칙은 `observation.contact.type` (§8.1) 에 따라 과도적 한계와 준정적 한계 중
해당하는 쪽을 적용합니다.

**`R-TS15066-SSM-02` · `R-HRI-CONTACT-01` 은 `S1e` · `S1f` · `S1g` 의 판정입니다.** 앞의 것은 방향을 따진 S_p
(§6 「`safety.separation_required` 산출 방법」)로 근접사고를 가르고, 뒤의 것은 로봇 몸이나 든 상자가 사람에 닿으면
힘과 무관하게 실패입니다(자체 설정). 이 둘을 선언한 시나리오는 `R-TS15066-SSM-01` · `R-10218-STOP-01` 을 선언하지
않습니다 — 그 둘이 읽는 요구 이격거리와 보호정지는 로봇 쪽 안전 기능의 신호이고, 이 시나리오들의 기록 데이터는
사람에 반응하지 않는 로봇입니다.

**정지 시험 표는 `parameters` 가 아니라 규칙의 `stop_table` 에 있습니다** — 임계값 계약(§9.3.1)이 파라미터마다 수
하나를 받기 때문입니다. 상태(빈손 · 상자 2 kg · 3 kg)마다 걷기와 돌기의 행 `[속도, 정지 시간, 정지 거리]` 이고
(`row_units`), 돌기의 정지 거리는 시험 상자의 모서리(반지름 `turning_reach_m`)가 지나간 거리입니다. 잰 방법과 날짜는
`method_note` · `measured_on` 이고, 다른 로봇은 같은 시험으로 자기 표를 만듭니다. 검증기 C1 이 표의 꼴(`determined_by` 가
`project` · 상태 셋 · 상태마다 상자 무게(`carry_mass_kg`)가 유한한 음이 아닌 수이고 빈손 · 2 kg · 3 kg 순서로 커짐 ·
`turning_reach_m` 이 유한한 양수 · 행마다 유한한 음이 아닌 수 셋 · 속도가 0 보다 크게 시작해 행마다 커지는 순서 · 단위 ·
잰 날짜)을 봅니다.

**`R-HRI-CONTACT-01` 을 선언한 시나리오의 심각도는 규칙 종류가 아니라 표 A.2 한계로 가릅니다.** 실패 가운데
`R-TS15066-A2-*` 가 하나라도 성립했거나 머리 · 얼굴(`BR-01` · `BR-02`)에 힘이 실린 접촉이 있으면 `S3`, 나머지(닿았지만
모든 접촉이 한계 안)는 `S2` 입니다. 준정적으로 분류된 머리 · 얼굴 접촉이 그 한계 안이면 `R-TS15066-A2-*` 는 성립하지 않으므로, 그
`S3` 의 `triggered_rules` 에는 부위가 나오지 않습니다 — 부위는 스텝 열(`observation.contact.force` 의 부위별 값)에서
읽습니다. 이 시나리오들에서는 접촉 유형을 가르지 못한 접촉(`none` 인데 힘이 있음)도 엄한 쪽
한계 — 준정적 한계와 동적 한계 가운데 작은 쪽, 동적이 「해당 없음」이면 어떤 힘이든 — 로 비교하므로
`notes.not_evaluated` 에 「비교하지 못한 힘」 항목이 생기지 않습니다. 비교하는 스텝은 인형이 자유로운 몸인 스텝
(`notes.hold_release_step` 부터)에 **첫 `contact_onset` 스텝**을 더한 것입니다 — 이 시나리오들의 붙잡기는 첫 접촉에서만 풀려
그 스텝은 붙잡는 힘이 실린 채 물리가 돈 스텝(`notes.hold_contaminated_steps` 1)이지만, 한 스텝만 닿는 접촉도 비교되도록
판정에 넣습니다(자체 설정 · §6 「조건부 필드 — 사람과 로봇」). 유형을 가르지 못한 힘과 머리 · 얼굴의 힘은 그 가운데 첫
`contact_onset` 부터, 유형이 매겨진 접촉은 그 모든 스텝에서 비교합니다. `S1a` · `S1d` 는 인형이 자유로운 몸인 스텝만
비교합니다. 등급 넷과 이름은 자체 설정이고 S2/S3 의 경계만 표준입니다. 다른
시나리오의 심각도는 그대로 — 접촉이 있는 실패는 `S3`, 없는 실패는 `S2` 입니다.

### 9.5 배포본이 싣는 규칙의 범위

**배포본의 `rules.json` 은 그 배포본이 담고 있는 시나리오가 선언한 규칙만 싣습니다.**
`scenarios.json` 도 같은 기준으로 좁혀지며, 빠진 것은 각각 `excluded_rules` ·
`excluded_scenarios` 에 이름과 사유로 남습니다.

**이유.** C1 의 계약은 「값이 있을 것」이 아니라 **「라벨의 근거를 재현할 수 있을 것」**
입니다(§10). 어떤 에피소드도 그 규칙으로 판정되지 않았다면 재현할 근거 자체가 없으므로,
그 규칙의 임계값이 비어 있다는 이유로 배포를 막는 것은 계약이 요구하는 바가 아닙니다.
S1 계열만 담는 배포본에서는 `R-25785-STAB-01`·`-02`·`R-22440-CTRL-01`·`R-OBJ-DROP-01`
넷이 늘 빠지고 미기재 임계값 4건도 함께 빠집니다. 여기에 `S1a`·`S1d` 는 `R-TS15066-SSM-02`·`R-HRI-CONTACT-01` 을,
`S1e`·`S1f`·`S1g` 는 `R-TS15066-SSM-01`·`R-10218-STOP-01` 을 더 뺍니다 — 둘은 서로 다른 판정입니다.

**범위는 선언이 아니라 데이터에서 유도합니다.** 변환기가 배치에 실제로 있는
`scenario_id` 를 읽어 정하므로, S1b 에피소드가 생기면 범위가 저절로 넓어지고 S3a 가
들어오면 안정성 임계값이 다시 **필수**가 됩니다. 손으로 관리하는 목록이 아닙니다.

**조용히 좁히지 않습니다.** 아래는 전부 변환을 중단시킵니다 — 시나리오가 하나도 없는
경우(규칙이 전부 빠져 C1 이 검사할 것이 없어짐), 정본에 없는 시나리오, 실재하는 규칙을
가리키지 않는 선언(오타가 조용히 규칙을 줄이는 경로), 그리고 **이미 라벨이 가리키는
규칙이 빠지는 경우**.

`status` 도 배포본 기준으로 다시 계산합니다. 정본은 미기재가 남아 있는 한
`thresholds_pending` 이 맞지만, 자기 규칙이 모두 채워진 배포본은 `complete` 입니다.

### 9.2 파일 구조

```
meta/rules.json
├── rules_version, schema_version, status
├── standard_refs        각 축의 근거 표준
├── contact_types        transient / quasistatic 정의
├── contact_limits       ← 부위코드 → 한계값 + 생체역학 파라미터
│   └── BR-01 … BR-12
│       ├── contact_permitted      접촉이 허용되는 부위인지 (표준 5.5.5.3)
│       ├── limits
│       │   ├── force_quasistatic     표 A.2 (12 신체 부위 단위)
│       │   ├── pressure_quasistatic  표 A.2 (29 특정 신체 영역 단위 → 최솟값으로 축약)
│       │   └── energy_transient      표 A.4 (전달 에너지 한계)
│       ├── transient_multiplier   표 A.2 의 동적 배수 F_T·P_T. 절대값이 아니다
│       └── biomechanical          effective_mass, spring_constant ← **표 A.3**
└── rules                판정 규칙 20개. contact_limits 를 참조
    └── R-TS15066-SSM-02.stop_table   정지 시험 표 — parameters 밖(§9.1)
```

**표준 수치의 자리는 `contact_limits` 한 곳뿐입니다** — 인체 더미(§8)의 부위별
유효질량·스프링상수도 같은 부속서에서 나오므로 같은 구조에 둡니다. 두 곳에 나누면 표준
개정 시 한쪽만 고치게 됩니다. **배포본에도 그 값이 그대로 실립니다**(§9.3 — 정본의
채움 상태를 말하는 자리는 그곳 하나입니다).

**부속서 A 의 층위가 둘이라는 점에 주의하십시오 (2026-08-24 정본 확인).**

| 값 | 표 | 층위 |
|---|---|---|
| 최대 허용 힘 | 표 A.2 | **12 신체 부위** — 우리 `BR-01~12` 와 1:1 |
| 최대 허용 압력 | 표 A.2 | **29 특정 신체 영역** — 부위 단위로 쓰려면 소속 영역의 최솟값 |
| 동적 접촉 한계 | 표 A.2 | **배수**(`F_T`·`P_T`)이지 절대값이 아니다. 표 A.2 footnote c 가 「최소 2배 이상」이라 하므로 이 배수는 **하한**이다 |
| 유효 질량 · 유효 탄성 계수 | **표 A.3** | 12 신체 부위. 표 A.2 가 아니다 |
| 전달 에너지 한계 | **표 A.4** | 12 신체 부위 |

`BR-01`(두개골/이마)·`BR-02`(얼굴)은 표 A.2 의 동적 칸이 **「해당 없음」**입니다. 한계가
느슨해서가 아니라 **그 부위와의 접촉이 허용되지 않기 때문**입니다(표준 5.5.5.3). 그래서
두 부위의 `transient_multiplier` 는 수치가 아니라 `not_applicable` 이고, 라벨링은 이 부위의
동적 접촉을 **크기와 무관하게** 성립으로 봅니다.

### 9.3 값의 출처 구분

각 임계값은 `determined_by` 로 누가 정하는지를 표시합니다.

| `determined_by` | 의미 | 배포본에 값이 실리는가 |
|---|---|---|
| `standard` | 표준이 정하는 값 | **예** — 값과 `standard_ref` 를 함께 싣습니다 |
| `project` | 실측·합의로 본 과제가 결정 | 예 |

**배포본은 표준 수치를 담습니다.** e-나라표준인증 열람본에 「이 문서는 저작권 규정에 따라
상업적 활용을 금지합니다」 표시가 있고 본 데이터셋은 상업적 이용을 허용하는 CC BY 4.0 이므로,
허락 없이 실으면 보유하지 않은 재배포 권한을 하위 이용자에게 주는 모양이 됩니다. **저작권자인
국가기술표준원이 조건 없이 이용을 허락하여**(2026-08-27) KS B ISO TS 15066 부속서 A 의 70개를 `meta/rules.json` 으로 옮겼습니다.

**`standard_ref` 는 값을 실은 뒤에도 남습니다.** 값이 없을 때는 값을 대신하는 것이었고,
값을 실은 지금은 **그 값의 출처**입니다 — 허락이 놓인 형태 자체가 「수치를 조항 인용과
함께」이므로, 검증기는 `standard` 항목에 값의 유무와 무관하게 `standard_ref` 를 요구합니다.

**아직 값 없이 조항 참조만 실린 항목이 1개 있습니다** — `intrusion_distance` 입니다.
이 값은 `separation_required` 를 통해 **생성 시점에 궤적으로 들어가므로**, 정본만
채우면 실린 회랑과 어긋납니다(§9.4 한계).

**`human_speed`·`intrusion_distance` 는 둘 다 KS B ISO 13855 가 정하지만 처지가 다릅니다**(2026-09-01 에
그 표준을 확인함).

- `human_speed` 는 **상수로 주어지고, 값을 실었습니다.** 전신이 걸어서 접근하는 구성의
  접근속도 파라미터는 여러 조항에서 같은 값이고, **`S1a` · `S1d` 의 생성이 쓴 값이 그
  값입니다.** 그래서 실린 값이 실린 회랑과 어긋나지 않습니다
- `intrusion_distance` 는 **상수가 아닙니다.** 침범거리 `C` 는 방호설비의 감지 구성에
  따라 달라지는 식으로 주어지며, 손까지 감지하는 구성부터 단일 빔까지 폭이 넓습니다.
  `S1a` · `S1d` 의 생성이 쓴 잠정값은 그중 어느 구성에도 대응하지 않습니다. 우리 구성에 어느 조항을
  대응시킬지는 `position_uncertainty_*` 와 같은 형태의 **과제 판단**이며
  재생성과 함께 정합니다. 에피소드의 `provisional` 표시가 남는 이유도 이 하나입니다

### 9.3.1 검증기 C1 이 요구하는 것

C1 은 「값이 있을 것」이 아니라 **「라벨의 근거를 재현할 수 있을 것」**을 요구합니다.
그 길이 셋이고, 셋 다 각자의 증거를 요구합니다.

| 형태 | 요구 조건 | 쓰이는 곳 |
|---|---|---|
| 값이 있다 | 수치·유한·비음수·단위 | `project` 임계값은 **이 형태만** 허용됩니다 |
| 값 대신 `standard_ref` | 비어 있지 않은 조항 참조 | 아직 채우지 못한 값 |
| `not_applicable` | 값이 `null` 이고 `reason` 이 있을 것 | 표준이 그 자리에 수치를 두지 않는 경우 |

여기에 더해 **`determined_by: standard` 는 세 형태 어디에 있든 `standard_ref` 가 있어야
합니다.** 값이 있으면 그것이 실린 수치의 출처이고, `not_applicable` 일 때는
면제 주장의 근거이기 때문입니다. 참조가 비면 출처 없는 표준 수치가 배포됩니다.

`not_applicable` 은 **`contact_permitted: false` 인 부위에서만** 성립합니다. 아무 데서나
쓸 수 있으면 빈 칸을 덮는 가장 싼 방법이 되기 때문입니다.

`project` 에 이 면제가 없는 이유는 그것이 **우리가 재서 채우는 값**이기 때문입니다.
비어 있으면 그냥 아직 안 잰 것이고, C1 은 그것을 배포 차단 사유로 봅니다.
다만 그것은 **배포본에 실린 규칙**에 대한 이야기입니다. 배포본이 담지 않은 시나리오의
규칙은 애초에 실리지 않으므로 차단 대상이 아닙니다 — §9.5.

단위가 비어 있으면 어느 형태든 위반입니다 — 무단위 임계값은 재현할 수 없습니다.

### 9.4 기계가 읽는 부분과 사람이 읽는 부분

`condition` 은 사람이 읽는 설명입니다. 기계가 읽는 것은 아래입니다.

| 필드 | 내용 |
|---|---|
| `inputs` | 규칙이 읽는 스텝 필드 목록. **검증기 C8 이 데이터에 실재하는지 대조**합니다 |
| `inputs_missing` | 계산에 필요한 필드가 스키마에 아직 없다는 선언. **이 항목이 있으면 C1 이 배포를 막습니다** — 필드를 추가하거나 배포본에서 규칙을 빼야 합니다 |
| `limits_ref` | 접촉 규칙이 참조할 `contact_limits` 경로 |
| `parameters` | 규칙별 임계값 (값·단위·출처) |
| `rule_precedence` | 여러 규칙이 동시에 성립할 때의 우선순위 |

비교 연산 자체를 표현식으로 담는 형식은 **아직 쓰지 않습니다** — 비교는 라벨링 모듈
(`generation/labeling/label_episodes.py`)의 코드가 구현합니다. 대신 `inputs` 선언을
검증기가 강제하므로, 규칙과 데이터가 어긋난 채로 배포되지는 않습니다.

**판정 결과는 `outcome` + `triggered_rules` + 초과 실측치를 함께 기록**합니다.
규칙 임계값이 바뀌면 원본 시계열로 **재라벨링만 수행**하면 되고, 데이터 재생성은 불필요합니다.

**예외가 하나 있습니다** — 생성 중에 로봇의 거동에 들어가는 파라미터는 그렇지 않습니다. `R-TS15066-SSM-01` 의 이격거리 파라미터는 생성 시점에 회랑을 정해 궤적에 실리므로, 그 값을 바꾸려면 재생성이 필요합니다. 임계값이 **판정에만** 쓰일 때 재라벨링으로 끝납니다.

### 시나리오별 구성 비율 (목표)
목표 비율은 시나리오마다 정본 `meta/scenarios.json` 의 `outcome_target` 에 있습니다(자체 설정 — 조건표의 창으로
맞추는 값이고 판정의 정의가 아닙니다). 아래는 결과마다의 사유와 기본 비율이고, 다른 비율을 두는 시나리오는 그
항목의 `outcome_target` 이 말합니다 — `S1f` 는 `fail` 70% · `near_miss` 30% · `nominal` 0% 이고 `S1g` 는 `fail` 90% · `near_miss` 10% · `nominal` 0% 입니다.

| outcome | 기본 비율 | 사유 |
|---|---|---|
| `fail` | 40% | 데이터셋의 존재 이유 |
| `near_miss` | 30% | 경계 판별 학습의 핵심 |
| `nominal` | 30% | 대조군 없이는 판별 불가 |

**검증기는 목표를 들지 않고 허용 하한에서만 경고합니다.** 목표는 시나리오마다의 것이라(위 — 정본의
`outcome_target`) 시나리오를 섞은 배포본에는 하나의 목표가 없고, 목표에서 조금 벗어난 배치도 정상 배치입니다.
그래서 C4 는 배포본 전체의 비율이 `fail` **25%** · `near_miss` **15%** 미만일 때 경고합니다 — "목적이 훼손되기
시작하는 선" 이지 목표가 아니고, 시나리오마다가 아니라 배포본 전체에 대는 한 벌입니다(자체 설정 —
`validate_dataset.py` 의 `OUTCOME_FLOOR`). 경고 문구는 하한만 말합니다.
비율은 **무작위 행으로만** 셉니다 — 조건표의 격자 행(`condition_design` 이 `grid`)은 창에서 뽑은 행이 아니라 격자가
정한 칸이라 세지 않고, 그 수는 검증 보고의 `grid_episodes_not_counted` 에 남습니다. `condition_design` 열이 없는
배포본과 그 값이 null 인 행은 셉니다. `outcome` 과 `triggered_rules` 의 쌍방 정합 검사는 격자 행에도 겁니다.

---

## 10. 검증 규칙 (배포 전 필수 통과)

`validator/validate_dataset.py`가 아래를 전수 검사합니다. 검사 항목의 정본은
[`validator/README.md`](../../validator/README.md) 이며, 여기 목록은 요지입니다.

1. `len(joint_names) == dof == observation.state.*.shape[-1]`
2. `region_names` 길이 = 접촉 배열 길이, 코드가 BR-01~12 집합에 포함 — **전 프레임 검사**
3. `contact.type` 길이 = `region_names` 길이, 값이 `none`/`transient`/`quasistatic` 집합에 포함
4. `outcome != nominal` 인데 `triggered_rules` 가 비어 있지 않은지 (역도 성립)
5. `triggered_rules` 의 rule_id 가 `rules.json` 에 실제로 존재
6. **`rules.json` 의 임계값이 전부 추적 가능함** — 값이 있거나, `standard_ref` 로 정의
   위치를 가리키거나, 접촉 금지 부위의 `not_applicable` 이어야 합니다(§9.3.1).
   `project` 값은 첫째 형태만 허용되므로, 비어 있으면 배포 차단입니다.
   검사 대상은 **배포본에 실린 규칙**이며, 무엇이 실리는지는 §9.5 가 정합니다
7. **계산할 수 없는 규칙이 없음** — `inputs_missing` 이 선언된 규칙이 있으면 배포 차단
8. **적용 규칙의 입력 필드가 데이터에 존재** — `scenarios.json` 이 선언한 적용 규칙만
   대조하므로, 조건부 필드(`observation.object.*`)를 쓰지 않는 시나리오는 영향받지 않음
9. `source=sim` 이면 `randomization_seed`·`sim_backend`·`run_start_seq` 존재. `sim_backend` 는 생성 커밋을
   싣고 더티 트리가 아니어야 하며, `run_start_seq` 는 자기 번호를 넘지 않음
10. **동일 시드 재실행 시 궤적 재현** — 표본 에피소드를 만든 **실행을 첫 번호(`run_start_seq`)부터 그
    에피소드까지** 다시 생성해 배포본의 열과 대조합니다. 에피소드 단독 재실행은 성립하지 않습니다(§3).
    비용이 **실행 안에서의 위치**에 비례하므로 표본은 그 위치가 일정 범위 안인 것에서만 고르고,
    확인하지 않은 나머지 건수를 보고에 남깁니다. **카메라 배포본**(`meta/cameras.json` 이
    있음)은 bbox·카메라 자세 열까지 비트 대조하고(정본이 팩을 선언했으면 팩의 깊이·마스크·
    `step_index` 도) **RGB 영상은 대조하지 않습니다** — 렌더가 비트 재현되지 않기 때문입니다. 카메라를 켠
    재생성이 비싸므로 표본 범위도 더 좁습니다(범위와 근거는 `validator/README.md`)
11. `scene_assets[].license` 가 전부 기재되어 있고, `redistributed=true` 인 자산은 `NOTICE`에 등재.
    배포본은 **그 배치의 에피소드가 쓰는** 재배포 자산만 싣고, 쓰는지는 에피소드의
    `scene_assets` 와 시나리오 정본의 `scene.assets`·`scene.human_model`, 에피소드의
    `robot_model` 로 봅니다. 다른 자산을 엮은 항목(`composes` — 엮은 장면)은 엮는 자산이 모두 실릴 때만
    함께 싣습니다. 실린 USD · 텍스처 · 로봇 폴더의 파일은 정본의 해시(`usd_sha256` ·
    `bundle.files` · `ship.files`)와 대조합니다
12. 타임스탬프 단조 증가, 결측 프레임 없음
13. 통계 이상치: 접촉력 음수, 관절각 한계 초과, NaN
14. **카메라 매체** — `meta/cameras.json` 이 있으면 §4 의 `videos/` 를 요구하고(팩의
    `packs/`·`meta/packs.json` 은 정본이 팩을 선언할 때만 요구하며, 선언하지 않았는데 있으면
    오류), 영상 해상도(`info.json`)·프레임 수(`meta/episodes` 의 영상 구간 × fps)가 카메라
    정본·에피소드 길이와 맞는지, 팩을 선언했으면 팩의 `step_index` 와 해시, bbox 범위(`bbox.camera`
    의 양끝 포함 픽셀 번호)를 봅니다. 영상 파일은 그 카메라의 `videos/<키>/` 아래에서 MP4 형식 표시(`ftyp`)로
    시작해야 하고, 한 파일을 나눠 쓰는 에피소드 구간은 0 초부터 에피소드 순서로 빈틈·겹침 없이
    이어져야 합니다(LeRobot 이 쓰는 배치). 검증기는 **영상을 디코드하지 않습니다.** 파일이 없으면 이 항목은
    하지 않으며, 카메라 없는 배포본의 보고에는 이 항목 줄이 생기지 않습니다
15. **사람과 로봇(hri) 배포본** — 시나리오가 `R-TS15066-SSM-02` 를 선언하면 §6 「조건부 필드 — 사람과 로봇」의 열을
    요구하고, 다가가는 속도가 음이 아닌지 · 보이는 몫이 0~1 인지 · `separation_violation` 이 `min_distance_human <
    separation_required` 이면서 `approach_speed_robot > approach_speed_threshold` 인지 봅니다. 에피소드의
    `notes.ssm_params` 는 그 규칙의 파라미터 다섯과 `stop_table` 이 배포본 정본과 같아야 합니다(표의 설명 문장
    `method_note` 는 빼고 — 문장만 고친 정본은 이미 만든 배치에 그대로 실립니다). **모든 스텝 열은
    층(§6 「층 — 로우와 가공」)이 있어야 합니다**
16. **조건표가 있는 시나리오의 에피소드 열** — 배포본의 `meta/scenarios.json` 에서 시나리오가 조건표(`conditions`)를
    선언하면 §5 의 「조건표 필수」 열을 요구하고 그 꼴을 봅니다: `episode_spec` 이 비어 있지 않고 멤버가
    `episode_spec_members` 와 같음 · `condition_design` 이 비어 있지 않음 · 격자 칸이 정수(값으로 — 정수인 실수도)이고 무작위 행은 -1 · -1,
    격자 행은 그 시나리오의 `conditions.grid.cells` 안 · 근거 요약 열의 값이 수(inf 포함) 또는 null 이고
    `evidence.min_separation_margin_m` 은 비어 있지 않음. 평평한 자리 열(`condition_design` · `grid_cell_robot` ·
    `grid_cell_human`)은 그 에피소드의 조건 기록 `notes.randomization.condition` 의 `design` · `grid_cell`(null 이면
    -1 · -1)과 같아야 하고 기록이 그 둘을 싣지 않으면 오류입니다 — 검증기의 분포(C4)는 열로, 점수표는 기록으로 격자
    행을 가르므로 둘이 갈리면 안 됩니다. 값이 그 에피소드의 스텝 · 라벨과 맞는지는 이 검사가 보지
    않습니다 — 배포본만으로 다시 유도해 대조하는 것은 `generation/labeling/check_release_labels.py` 입니다

---

## 11. 버전 관리

| 대상 | 규칙 |
|---|---|
| `schema_version` | 필드 추가 = minor / 의미 변경·삭제 = major |
| 데이터셋 태그 | `v<연 두 자리>.<월>.<차례>` — 2026년 11월의 첫 공개가 `v26.11.0`, 같은 달의 고침은 `v26.11.1`. Hub 의 `v3.0` 은 LeRobot 적재기가 요구하는 형식 판의 태그라 게시마다 옮기고, 이 태그 · `schema_version` 과 셋이 따로입니다 |
| 매핑 파일 | 로봇 추가 시 파일 추가. 기존 파일 불변 |
| 규칙 파일 | 임계값 개정 시 `rules.json` 버전 증가 + 재라벨링 이력 기록 |

**하위 호환 원칙**: 기존 필드의 의미를 바꾸지 않습니다. 필요하면 새 필드를 추가하고 구 필드는 유지합니다.

---

## 12. 미결 사항

1. ~~**H1 실제 관절 이름·순서**~~ — **확정 (2026-08-18).** Isaac Sim 6.0.1 자산에서 추출해
   `meta/joint_map/unitree_h1.yaml` 생성, §7.3 확정. 남은 확인 항목은 `needs_review` 의
   7개 관절(`torso`, 좌우 `knee`·`ankle`·`elbow`)의 **회전축**이며, 이 중 G1 과의 정합에
   실제로 영향을 주는 것은 `torso` 와 좌우 `ankle` 3개입니다
   (G1 도 `knee`·`elbow` 는 축 토큰 없이 같은 이름을 씁니다)
2. ~~**`action_space` 정의**~~ — **확정(2026-08-25).** 위치 제어이며 `action` 은
   라디안 단위의 관절 목표입니다. 필수 에피소드 필드 `action_space` 가 `position_rad`
   로 선언하고, 토크 제어는 `torque_nm` 으로 예약해 두었습니다 (§5·§6)
3. ~~**fps**~~ — **확정.** 50 Hz(`dt` 0.02 s)이고, 접촉력은 기록 스텝에 속한 물리 서브스텝의 최댓값이라
   기록 주기를 올리지 않습니다(§6 · 카탈로그 §9-1)
4. **영상 모달리티 포함 범위**
5. **`severity` S0~S3 구간** — Guide 51 harm 등급을 본 과제 기준으로 구체화
6. ~~**파지물 상태 필드**~~ — **확정.** `observation.object.{names,position,grasped}` 를 조건부 필드로 추가
7. **다른 기관의 보안 데이터셋과의 필드 정합** — `comm_error`(S6) 축 공유 여부
