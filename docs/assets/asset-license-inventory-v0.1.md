# 자산 라이선스 인벤토리 v0.1

> **이 문서는 데이터셋 배포본에 동봉됩니다.** `NOTICE` 와 `meta/assets.json` 이
> 라이선스 판정의 근거로 이 파일을 가리키므로, 받는 쪽이 판정을 따라갈 수 있도록
> 함께 싣습니다.
> 대상: Isaac Sim 6.0.1 번들 자산 (`isaacsim_assets/Assets/Isaac/6.0/`)
> 조사일: 2026-08-18 / 상태: **26.11 배포본 사용분 확정**
>
> 본문이 인용하는 `generation/…` 경로는 조사에 쓴 **프로젝트 저장소의 스크립트**이며
> 배포본에는 들어 있지 않습니다. 판정 결과와 그 근거는 이 문서 안에서 닫힙니다.

---

## 0. 결론 먼저

**제3자 자산 원본은 배포하지 않습니다. 산출물과 자체 제작 자산은 배포합니다.** 예외는 넷 —
자체 제작 자산이 재질로 쓰는 **CC0 텍스처**는 그 자산과 함께 싣고, 관절 인형이 USD 안에 싣는 **걸음 표**는
100STYLE(CC BY 4.0)에서 유도한 것이라 출처 · 라이선스 링크 · 바꾼 내용과 함께 실으며(원본 동작 파일은 싣지 않음 — §3.1),
상자의 라벨(창고 씬의 상자와 들고 다니는 상자)에 얹는 **KETI CI** 는 배포자 자신의 CI 라 원본 그대로 싣되 데이터셋 라이선스와 따로 표기하고
(`LicenseRef-KETI-CI` — 데이터셋의 일부로만 재배포할 수 있고, CC BY 4.0 §2(b)(2) 대로 상표 사용은 허락하지 않음 · §3.1), **Unitree H1** 로봇은 자기 라이선스 파일이
있는 BSD-3 경로의 사본이라 저작권 고지 · 라이선스 전문 · 면책 문구와 함께 제 폴더(`assets/unitree_h1/`)에 원본 그대로 싣습니다(§3).

| | |
|---|---|
| 배포 | 시뮬레이션 **산출물** — 궤적 Parquet, 카메라 일곱 대의 RGB 영상(MP4), 메타데이터 (CC BY 4.0). 영상에 나타나는 외형은 §5 의 2번에 있습니다 |
| 배포 | KETI **자체 제작** USD 중 그 배치의 에피소드가 쓰는 것 (CC BY 4.0) — 26.11 배포본(`S1e` · `S1f` · `S1g`): 물류 창고 씬 · 조끼를 입힌 관절 인형 · 들고 다니는 상자. 시각 메시를 입힌 인체 더미(`keti_dummy_ts15066_v1_skin`)는 시나리오 `S1d` 를 담는 배치의 인체 모델이라 이 배포본에는 없습니다. 지면은 파일 없는 절차적 평면이므로 배포 대상이 아닙니다 |
| 배포 | 판정에 쓰는 **표준 임계값 70개** (CC BY 4.0) — §7 에 재배포 근거가 있습니다 |
| 배포 | 자체 제작 자산이 쓰는 **CC0 텍스처** — 원본 그대로이며 파일마다의 출처·SHA256 이 배포본의 `meta/assets.json` 에 실립니다 (§3.1) |
| 배포 | 관절 인형 USD 안의 **걸음 표** — 100STYLE(CC BY 4.0)에서 유도한 표. 출처 · 라이선스 링크 · 바꾼 내용이 배포본의 `meta/assets.json` 과 `NOTICE` 에 실리고, 원본 동작 파일(BVH)은 싣지 않습니다 (§3.1) |
| 배포 | 상자 라벨(창고 씬의 상자와 들고 다니는 상자 — 같은 파일)의 **KETI CI** — 배포자(한국전자기술연구원) 자신의 CI. 공식 PNG 를 바이트 그대로 싣고 출처 · SHA256 이 배포본의 `meta/assets.json` 에 실립니다. 데이터셋 라이선스와 따로 표기하며(`LicenseRef-KETI-CI`) 상표 사용은 허락하지 않습니다(CC BY 4.0 §2(b)(2)) (§3.1) |
| 배포 | **Unitree H1** 로봇 — BSD-3 원본 `IsaacLab/Robots/Unitree/H1/h1.usd` 와 그것이 상대 경로로 참조하는 `instanceable_meshes.usd`, 그 경로의 라이선스 파일 `LICENSE.txt` 를 바이트 그대로 배포본의 `assets/unitree_h1/` 에. 파일마다의 출처 · SHA256 이 배포본의 `meta/assets.json`(`ship`)에, 저작권 고지와 라이선스 전문 · 면책 문구가 `NOTICE` §3 · §4 에 실립니다. 이 파일들은 데이터셋 라이선스가 아니라 BSD-3 을 따릅니다 (§3) |
| 배포 | 자체 제작 자산과 H1 을 **엮은 장면(Stage)** — Isaac Sim 에서 여는 장면 하나(`assets/keti_hri_stage_v1.usd`, CC BY 4.0). 엮는 자산이 모두 실리는 배포본에만 실리고, 시나리오를 돌리는 것은 코드 저장소의 명령입니다 (§3.1) |
| 비배포 | 그 밖의 NVIDIA·제3자 **자산 원본** (`.usd`) — 자기 라이선스 파일이 없는 NVIDIA 자산은 쓰지 않거나 경로·버전·SHA256 으로 참조만 |

이 방침이면 **자산** 쪽에서 라이선스 회신을 기다릴 항목이 없습니다. 재현성은 로봇 밖의 제3자 자산을
동봉하지 않고도 확보됩니다 — 로봇은 배포본에 실려 있고, 실행 환경은 이용자가 Isaac Sim 을 설치하면
같은 것을 얻으며, 우리는 그 버전과 해시를 기록해 두기 때문입니다.

**자산이 아닌 것 하나는 회신을 받았습니다** — 판정에 쓰는 표준 임계값입니다. §7 을 보십시오.

---

## 1. 판정 기준

자산의 제작 주체가 아니라 **자산에 붙은 라이선스 파일이 기준**입니다. 같은 로봇이라도
경로에 따라 라이선스가 다르므로(§2.1), "NVIDIA 배포분이니 자유롭다"는 전제는 성립하지 않습니다.

조사 방법은 자산 트리에서 라이선스·저작권 고지 파일을 전수 검색하고, 파일이 없는 자산군은
상위 계약(NVIDIA Omniverse License Agreement)의 적용을 받는 것으로 판정했습니다.

### 상위 계약의 핵심 조항 (Omniverse License Agreement §2.1)

| 대상 | 조항 요지 |
|---|---|
| Content (NVIDIA 3D 자산) | 본인 제품·서비스에 **편입한 형태로는 배포 가능**하나, **단독 배포는 불가** |
| User generated content | Omniverse 로 **생성한** 영상·스틸·모델 등은 **배포 가능** |

우리가 만드는 것은 후자(시뮬레이션 산출물)이고, 자산 `.usd` 를 데이터셋에 담는 것은
전자의 "단독 배포"에 해당할 소지가 있습니다. §0 방침의 근거입니다. 이 상위 계약은 자기 라이선스
파일이 **없는** 자산의 몫입니다 — 자기 라이선스 파일이 있는 자산은 위 기준대로 그 파일이 정하고,
H1 의 BSD-3 경로가 그렇습니다(§2 · §3).

---

## 2. 자산군별 현황

> **기계가 읽는 정본은 `meta/assets.json` 입니다.** 이 문서는 판정의 *근거*를 적고,
> 그 판정의 결과(경로·라이선스·재배포 가능 여부)는 저 파일 한 곳에만 둡니다. 에피소드
> 메타의 `scene_assets` 와 `robot_embodiment` 가 거기서 나오며, 변환기는 생성 스크립트가
> 실제로 로드한 경로로 그 파일을 조인합니다 — 정본에 없는 경로가 나오면 변환이 멈춥니다.
> 값을 이 문서에서 새로 적지 마십시오. 사본이 둘이 되면 어긋납니다 — 정본은 `meta/assets.json` 하나입니다.


| 자산군 | 라이선스 근거 | 판정 | 사용 |
|---|---|---|---|
| `Robots/Unitree/` | `unitree-LICENSE.txt` = **MPL-2.0** | 재배포 가능, 파일단위 카피레프트 | G1 참조만 (§3) |
| `IsaacLab/Robots/Unitree/` | `LICENSE.txt` = **BSD-3-Clause** | 재배포 가능, 고지 의무만 | **H1 사용 · 배포본에 동봉** (§3) |
| `Samples/Mujoco_Menagerie/unitree_*/` | `LICENSE` = **BSD-3-Clause** | 재배포 가능 | 미사용 |
| `Environments/` | `environment-supplement-LICENSE.txt` | **재배포 불가** — Limited Use Content (non-transferable · non-sublicensable · 수정 금지) | **미사용** |
| `Materials/Textures/Skies/PolyHaven/` | `license.txt` = **CC0 1.0** | 제약 없음 | 미사용 |
| `Props/`, `Sensors/`, `People/`, `Materials/` (기타), `Robots_Multiphysics/` | 라이선스 파일 **없음** → 상위 계약 적용 | 단독 배포 불가 | **미사용** |
| `IsaacLab/ActuatorNets/Unitree/` | `LICENSE.txt` = MIT | 재배포 가능 | 미사용 |

### 2.1 같은 로봇, 다른 라이선스 — 주의 지점

Unitree H1·G1 자산은 **세 경로에 각각 다른 라이선스로** 존재합니다.

| 경로 | 라이선스 | 저작권자 표기 |
|---|---|---|
| `Isaac/Robots/Unitree/` | MPL-2.0 | Unitree Robotics (2018-2019) |
| `Isaac/IsaacLab/Robots/Unitree/` | **BSD-3-Clause** | HangZhou YuShu TECHNOLOGY (2016-2022) |
| `Isaac/Samples/Mujoco_Menagerie/unitree_h1\|g1/` | BSD-3-Clause | HangZhou YuShu TECHNOLOGY (2016-2023) |

**H1 은 BSD-3 경로(`IsaacLab/Robots/Unitree/H1/h1.usd`)를 정본으로 씁니다.**
BSD-3 은 고지 의무만 지므로 CC BY 4.0 데이터셋과 함께 다루기가 가장 단순하고,
Isaac Lab 이 생성에 쓰는 프레임워크라 경로도 자연스럽습니다. 같은 이유로 이 사본을 라이선스
전문과 함께 배포본에 싣습니다(§3).

### 2.2 두 H1 사본의 실측 차이

`generation/dump_joint_names.py` 로 두 사본을 각각 추출해 대조했습니다.

| 항목 | `Robots/Unitree` (MPL-2.0) | `IsaacLab/Robots` (BSD-3) |
|---|---|---|
| DOF 수·이름·**순서** | 19 | 19 — **완전 동일** |
| 전체 관절 수 | 24 (센서 장착 고정관절 5개 포함) | 19 (고정관절 없음) |
| 센서 장착점 | `d435_left_imager`, `d435_rgb_module`, `imu`, `logo`, `mid360` | 없음 |
| `hip_pitch` 한계 | `[-3.14, 2.53]` | `[-1.57, 1.57]` |

관절 순서가 같으므로 데이터 스키마 규격서가 확정한 관절 구성은 사본 선택과 무관하게 유효합니다. 배포본에서는 `meta/joint_map/unitree_h1.yaml` 이 그 정본입니다.
다만 **한계값이 다르므로** 검증기 C8 의 근거는 실제 사용 사본(BSD-3)을 따릅니다.

> BSD-3 사본에는 카메라·LiDAR 장착 프레임이 없습니다. 영상 모달리티를 쓸 경우
> 카메라 프림을 직접 배치해야 하며, 씬을 자체 제작하는 방침이므로 추가 부담은 아닙니다.

### 2.3 Isaac Lab 의 H1 태스크가 기본으로 싣는 파일은 이 표의 어느 것도 아니다(2026-08-20 실측)

> **이 데이터셋의 환경은 그 기본값을 `h1.usd` 로 덮어씁니다 — §2.4.**

`robot.cfg.spawn.usd_path` 로 대조해 보면, Isaac Lab 의 H1 velocity 태스크(`H1FlatEnvCfg` →
S1a 가 상속)가 기본으로 로드하는 것은 위 §2.1~2.2 의 `h1.usd` 가 아니라 같은 BSD-3 경로 아래
**`h1_minimal.usd`** 입니다. 라이선스는 같은 디렉터리이므로 문제가 없지만, **콜라이더
구성이 다릅니다** — 로봇의 20개 바디를 전수 검사한 결과:

| 자산 | 콜라이더 있는 바디 |
|---|---|
| `h1_minimal.usd` (Isaac Lab H1 태스크의 기본값) | **3/20** — `left_ankle_link`, `right_ankle_link`, `torso_link` 뿐 |
| `h1.usd` (§3 이 정본으로 못박은 BSD-3 경로) | **20/20** |

즉 그 기본값으로 데이터를 만들면 로봇의 팔·다리에 콜라이더가 없어 **로봇이 그
부위로 인체를 건드려도 접촉이 원리적으로 생성되지 않습니다** — 부위별 접촉이 이
데이터셋의 존재 이유이므로 직접적인 영향입니다.

### 2.4 `h1.usd` 로 덮어씁니다 (2026-08-20 실측)

덮어쓸 때 따질 것은 하나입니다 — 사전학습 정책(`Isaac-Velocity-Flat-H1-v0`)이
`h1_minimal.usd` 로 학습됐을 가능성. `generation/envs/s1a/check_asset_swap.py` 가
같은 시드로 두 자산을 나란히 돌려 확인했습니다.

| | `h1_minimal.usd` | `h1.usd` |
|---|---|---|
| 에피소드 완주 | 2/2 | 2/2 |
| 평균 전진속도 | +0.961 m/s | +0.972 m/s |
| 평균 전진거리 | +5.72 m | +5.79 m |
| 강제 겹침 시 접촉 보고 바디 | **3/20** (발 외 1: torso) | **13/20** (발 외 12) |

**걸음은 달라지지 않았고**(정책 재학습 불필요), **커버리지는 11개 바디가 늘었습니다** —
양 팔꿈치·양 무릎·좌우 hip 3링크씩·pelvis. 정상 보행 중에는 두 자산 모두 발목 2개만
보고합니다(닿는 것이 지면뿐이므로 당연합니다). 자산 외에 다른 차이는 없습니다:
Isaac Lab 의 `H1_MINIMAL_CFG` 는 `H1_CFG.copy()` 에 `spawn.usd_path` 한 줄만 바꾼
것이고, 두 USD 는 움직이는 관절 19개의 이름·순서와 바디 20개의 이름·질량이 동일합니다.

`generation/envs/s1a/s1a_env_cfg.py` 가 `spawn.usd_path` 를 `h1.usd` 로 덮어씁니다 —
§3 이 정본으로 못박은 경로와 실행 시점 자산이 일치합니다.

---

## 3. 사용 자산 확정

| 자산 | 출처 | 라이선스 | 배포 |
|---|---|---|---|
| Unitree H1 (19DOF) | `IsaacLab/Robots/Unitree/H1/h1.usd` 와 그것이 같은 폴더에서 상대 경로로 참조하는 `instanceable_meshes.usd`, 라이선스 파일 `IsaacLab/Robots/Unitree/LICENSE.txt` | BSD-3-Clause | **포함** — 세 파일을 바이트 그대로 `assets/unitree_h1/` 에(파일마다의 출처 · SHA256 은 `meta/assets.json` 의 `ship`, 받기 · 검증은 `generation/robots/fetch_robot.py`). 저작권 고지와 라이선스 전문 · 면책 문구는 `NOTICE` §3 · §4. 이 로봇을 쓰는 배치의 배포본에만 실립니다 |
| `keti_dummy_ts15066_v1` | KETI 자체 제작 | CC BY 4.0 | **26.11 배포본에는 싣지 않음** — 이 더미의 부위 프리미티브에 관절을 더하고 시각 메시와 조끼를 입힌 인형이 실립니다(§3.1). 이 캡슐 더미만 쓰는 S1a 배치는 배포하지 않습니다 |

H1 의 판정 근거 — 그 폴더에는 자기 라이선스 파일(BSD-3)이 있습니다. BSD-3 은 저작권 고지 · 조건 · 면책 문구를
함께 싣는 한 바이너리 재배포를 허락하고, Omniverse 계약의 단독 배포 금지는 자기 라이선스 파일이 없는 NVIDIA 자산의
몫입니다(§1). 폴더째 싣는 것은 다른 로봇의 같은 이름 파일과 부딪치지 않게 하기 위함이고, USD 가 참조하는 레이어가
같은 폴더에서 상대 경로로 풀립니다. 표준 수치를 USD 에 싣지 않는다는 금지 속성 검사는 우리가 만든 USD(`assets/` 바로
아래)에만 걸리고, 이 폴더는 행마다 고정한 SHA256 으로 대조됩니다 — H1 의 질량은 로봇 자신의 값입니다.

씬을 자체 제작으로 구성하기로 한 판단이 여기서 확인됩니다(S1a 는 지면뿐이고 S1d · S1e · S1f · S1g 는 §3.1 의 물류 창고 씬) — 배포본에서는 `meta/scenarios.json` 이 그 정본입니다.
`Environments/` 를 썼다면 배포가 불가능했습니다.

각 에피소드의 `scene_assets[]` 에 `{path, license, redistributed}` 를 기록하므로,
나중에 특정 자산의 조건이 바뀌어도 **해당 에피소드만 선별 제거**할 수 있습니다.

### 3.1 자체 제작 자산 — 26.11 배포본이 싣는 것

26.11 배포본은 `S1e` · `S1f` · `S1g`(물류 창고의 사람–H1) 배치 297건(시나리오마다 99건)입니다. 배포본은 그 배치의
에피소드가 쓰는(에피소드 메타 `scene_assets`) 재배포 자산만 싣습니다 — 그래서 아래 자산을 쓰지 않는 배치
(S1a)를 변환한 배포본에는 그 자산이 없고, 거꾸로 이 배포본에는 캡슐 더미 `keti_dummy_ts15066_v1` 도,
시나리오 `S1d` 를 담는 배치가 인체 모델로 쓰는 `keti_dummy_ts15066_v1_skin` 도 없습니다. 아래 첫 표는 창고 씬 ·
그 텍스처 · 시각 메시를 입힌 인체 더미를 적고, 이 가운데 입힌 더미만 이 배포본에 싣지 않습니다(이 배포본의 인체 모델은
그 아래 표의 `keti_dummy_ts15066_v2_vest`). `meta/assets.json` 의 `redistributed` 는 첫 표의 두 USD 모두
`true` 입니다(변환기·게시기·검증기 C6 가 에피소드 기준으로 봅니다). 배포본에는 USD 가
`assets/` 에, 텍스처가 USD 의 상대 참조(`./textures/<파일>`) 그대로 `assets/textures/` 에 실리고, 파일마다의
출처·라이선스·SHA256 은 배포본의 `meta/assets.json` 해당 항목 `bundle.files` 에 있습니다 — 배포본에는
lock 파일이 없기 때문입니다.

| 자산 | 출처 | 라이선스 | 배포 |
|---|---|---|---|
| `keti_logistics_aisle_v1` | KETI 자체 제작 — `generation/scene/logistics_aisle.yaml` 을 `generation/scene/build_logistics_scene.py` 가 USD 로 만듦. 선반 상자와 랙 바닥 단의 큰 상자마다 y 를 보는 두 옆면(통로와 줄 사이 틈을 보는 면)에 흰 라벨과 KETI CI(콜라이더 없는 시각 메시 · 크기 · 색 · 띄우는 거리는 자체 설정 — 같은 파일의 `labels`) | CC BY 4.0 | **포함** (`assets/`, 텍스처 · KETI CI 와 함께) |
| 씬 텍스처 5 세트 (콘크리트 — 바닥·벽 아랫단 · 골강판 — 천장·벽·셔터 문 · 도색 강판 · 합판 · 골판지). 지붕 트러스·등기구·바닥 표시선·랙 바닥 단의 팔레트·상자 라벨의 흰 판은 텍스처 없는 단색 재질. 라벨의 KETI CI 는 아래 「들고 다니는 상자」 표의 그 행 | Poly Haven 4 세트 · ambientCG 1 세트. 파일마다 출처 URL·SHA256 은 `generation/scene/textures.lock`, 내려받기·검증은 `generation/scene/fetch_textures.py` | **CC0 1.0** (각 자산 페이지의 표기를 2026-09-22 확인) | `assets/textures/` 에 포함 — 세트마다 세 장(`<세트>_diff_1k.jpg` · `<세트>_nor_gl_1k.jpg` · `<세트>_rough_1k.jpg`, `Cardboard001` 은 `Cardboard001_1K-JPG_Color.jpg` · `Cardboard001_1K-JPG_NormalGL.jpg` · `Cardboard001_1K-JPG_Roughness.jpg`) |
| `keti_dummy_ts15066_v1_skin` | KETI 자체 제작 — `keti_dummy_ts15066_v1` 의 물리 그대로에 `generation/dummy/skin/skin_spec.yaml` 대로 MakeHuman(MPFB2 2.0.17) 이 만든 시각 메시를 `generation/dummy/build_dummy.py --skin` 이 입힘. 얼굴은 합성 | CC BY 4.0 | **26.11 배포본에는 싣지 않음** — 이 배포본의 인체 모델은 아래 `keti_dummy_ts15066_v2_vest` 입니다. 이 자산을 인체 모델로 쓰는 배치의 배포본에는 `assets/` 에, 텍스처와 함께 실립니다 |
| MakeHuman 시스템 자산 6종 (피부 · 작업복 · 안전화 · 머리 · 눈 · 눈썹) 과 텍스처 8장 | MakeHuman 시스템 자산 팩 `makehuman_system_assets_cc0.zip` (files.makehumancommunity.org). 쓴 자산과 근거는 `generation/dummy/skin/SKIN-LICENSE.md`, 텍스처 8장과 팩 파일 16개의 팩 안 경로·SHA256 은 `generation/dummy/skin/textures.lock` — 팩 zip 과 바이트로 대조한 뒤에만 쓰는 표 | **CC0 1.0** (팩 메타데이터와 각 자산 파일 머리글 「explicitly released as CC0 in september 2020」, 2026-09-22 확인) | 텍스처 8장만 `assets/textures/` 에 포함(`brown_eye.png` · `eyebrow001.png` · `male_worksuit01_diffuse.png` · `male_worksuit01_normal.png` · `middleage_lightskinned_male_diffuse2.png` · `shoes01_diffuse.png` · `shoes01_normal.png` · `short02_diffuse.png`) — 팩 파일은 사람을 만든 입력이라 싣지 않음 |

판정 근거는 §1 의 원칙 그대로입니다 — 자산에 붙은 라이선스 표기로 판정합니다. Poly Haven 은 전
자산을 CC0 로 배포하고 각 페이지가 "CC0 License" 를 표기하며, ambientCG 는 "released under the
Creative Commons CC0 license" 를 표기합니다. MakeHuman 은 시스템 자산 팩을 CC0 로 배포하고 MPFB 의
LICENSE.md 가 내보낸 모델에 「어떤 권리도 주장하지 않는다」고 적습니다 — 도구(MPFB)는 GPL 이지만
산출물에는 미치지 않습니다(`SKIN-LICENSE.md`). NVIDIA `Environments/`·`Props/`·`Materials/` 는 §2 의
판정대로 쓰지 않았습니다. 씬 USD 에 표준 수치는 없습니다(치수는 전부 과제가 정한 장면 치수).

**관절 인형(v2) — 그것을 인체 모델로 쓰는 배치가 더 싣는 것.** 위 표의 피부 텍스처 8장은
같고(같은 lock · 같은 행), 아래가 더해집니다. 배포 여부는 위와 같은 규칙입니다 — 그 배치의 에피소드가 참조할 때만. 26.11 배포본이 그 배치입니다.

| 자산 | 출처 | 라이선스 | 배포 |
|---|---|---|---|
| `keti_dummy_ts15066_v2_jointed` | KETI 자체 제작 — `keti_dummy_ts15066_v1` 의 부위 프리미티브에 관절 10 · 발 둘 · 부위 질량(밀도)을 더한 관절 인형(`generation/dummy/joints.yaml` 을 `generation/dummy/build_dummy.py --jointed` 가 USD 로 만듦). 관절 · 가동 범위 · 발 · 부위 질량은 자체 설정 | CC BY 4.0 | **싣지 않음** — 물리 대조의 기준이고, 같은 물리에 피부와 조끼를 입힌 아래 자산이 실립니다 |
| `keti_dummy_ts15066_v2_vest` | KETI 자체 제작 — 관절 인형에 위 표의 MakeHuman 시각 메시와 아래 조끼를 `build_dummy.py --jointed --skin --vest --seams` 가 입힘. 물리는 관절 인형과 같음. 피부는 부위 몸마다 갈라 붙이되 엉뚱한 몸에 든 조각(늘어뜨린 손 옆의 바지 등)은 둘레의 몸으로 옮기고, 관절 이음새 근처의 피부 정점마다 이웃 몸과 섞는 비율을 실어 시나리오 실행이 렌더할 때마다 섞음 — 옮기는 기준 · 관절마다의 반경은 자체 설정(`generation/dummy/skin_seams.yaml`) | CC BY 4.0 | **포함** (`assets/keti_dummy_ts15066_v2_jointed_skin_vest.usd`, 피부 텍스처와 함께) |
| 형광 조끼 | KETI 자체 제작 — 피부의 작업복(MakeHuman `male_worksuit01`, CC0) 가운데 몸통 · 골반 부위(BR-04 · BR-05 · BR-06 · BR-07)에 든 옷의 바깥 둘레를 재서 그 위를 덮듯 고른 판을 아랫단 · 띠 · 진동둘레 · 목둘레 · V넥으로 자른 메시. 텍스처 없이 단색 재질 둘(형광 노랑 · 밝은 회색 반사 띠). 덮는 부위 · 둘레를 재고 고르는 값 · 띄우는 거리 · 아랫단 · 띠 · 진동둘레 · 목둘레 · V넥의 자리 · 색 · 광택은 자체 설정(`generation/dummy/vest.yaml`) | CC BY 4.0 | 조끼 입은 인형 USD 안에 포함 — 따로 싣는 파일 없음 |
| 걸음 표 (100STYLE 에서 유도) | The 100STYLE Dataset - Ian Mason (https://www.ianxmason.com/100style/ · 원본 묶음 https://zenodo.org/records/8127870). 앞으로 걷기(`Neutral_FW`)를 인형 관절로 옮기고(리타깃) 디딤발을 바닥에 맞추고 주기를 평균해 속도별로 정리함 — `generation/dummy/gait/build_gait.py`, 원본 묶음의 해시는 `generation/dummy/gait/gait.lock` | **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/) | 유도한 표만 관절 인형 USD 안(`/KetiDummy/Gait`)에 포함 — 출처 · 라이선스 · 바꾼 내용이 배포본의 `meta/assets.json`(`gait_source`)과 `NOTICE` 에 실립니다. 원본 동작 파일(BVH)은 저장소에도 배포본에도 싣지 않음 |

판정 근거는 같은 원칙입니다 — 100STYLE 은 CC BY 4.0 으로 배포되고(`gait.lock` 이 원본 묶음의 해시와 함께 적은
라이선스), CC BY 4.0 은 출처 · 라이선스 링크 · 바꾼 내용을 밝히면 바꾼 것을 재배포하도록 허락합니다. 조끼는 CC0 작업복의 둘레를 재서 우리가 만든 메시라
데이터셋과 같은 CC BY 4.0 입니다.

**들고 다니는 상자와 상자 라벨의 KETI CI — 그것을 쓰는 배치가 더 싣는 것.** 배포 여부는 위와 같은 규칙입니다 — 그 배치의 에피소드가 참조할 때만. 상자는 그것을 드는 배치에, KETI CI 는 상자를 드는 배치와 창고 씬을 쓰는 배치에 실립니다. 26.11 배포본은 세 시나리오 모두 로봇이 상자를 들어 둘 다 실립니다.

| 자산 | 출처 | 라이선스 | 배포 |
|---|---|---|---|
| `keti_carried_box_v1` | KETI 자체 제작 — `generation/scene/carried_box.py` 가 USD 로 만듦. 단색 골판지 몸에 흰 라벨과 KETI CI 를 네 옆면과 윗면의 가운데에. 기본 크기 · 몸 색 · 라벨과 CI 의 크기는 자체 설정이고, 크기를 바꿔도 라벨은 늘어나지 않음. 질량 · 강체 없음 | CC BY 4.0 | **포함** (`assets/keti_carried_box_v1.usd`, KETI CI 와 함께) |
| KETI CI (영문 대표 CI · 가로형 PNG 584 × 147) | 한국전자기술연구원 누리집 「CI 소개」(https://keti.re.kr/introduce/ci.php)의 내려받기 https://keti.re.kr/file/KETI_CI.zip?a=1 안의 PNG. 압축 파일과 PNG 의 SHA256 은 `generation/scene/keti_ci.json`, 받기 · 검증은 `generation/scene/fetch_keti_ci.py` | **`LicenseRef-KETI-CI`** — 배포자 자신의 CI. 데이터셋 라이선스와 따로 표기하고, 이 데이터셋의 일부로 바꾸지 않은 채 함께 재배포할 수 있음. 상표로 쓰거나 데이터셋과 떼어 따로 쓰거나 배포할 권리는 주지 않음 — CC BY 4.0 §2(b)(2) 도 상표 사용은 허락하지 않음 | 원본 PNG 를 바이트 그대로 `assets/textures/keti_ci_en.png` 에 포함 — 창고 씬(`keti_logistics_aisle_v1`)의 상자 라벨도 같은 파일이라 그 씬을 쓰는 배치(상자를 들지 않는 시나리오 포함)에도 실리고, 둘이 함께 실리면 한 벌 |

판정 근거 — KETI CI 는 배포자 자신의 CI 입니다. 누리집의 저작권 정책은 사전 허가 없는 배포를 금하므로 사용 허락을 받고(2026-10-17)
상자의 라벨로만 싣습니다 — 들고 다니는 상자와 창고 씬의 상자. 데이터셋을 받은 쪽은 이 파일을 데이터셋의 일부로 바꾸지 않은 채 함께 재배포할 수 있고,
데이터셋과 떼어 따로 쓰거나 배포할 권리는 받지 않습니다. CC BY 4.0 은 상표권을 허락하지 않으므로(§2(b)(2)) 이 CI 를 상표로 쓸 권리도
생기지 않습니다. 상자 메시와 재질은 우리가 만든 것이라 데이터셋과 같은 CC BY 4.0 입니다.

**엮은 장면(Stage) — 엮는 자산이 모두 실리는 배치가 더 싣는 것.** 에피소드가 참조하는 파일이
아니라, 그것이 엮는 자산(`meta/assets.json` 의 이 항목의 `composes` — 창고 · H1 · 조끼 입은 관절 인형 · 들고 다니는 상자)이
모두 실리는 배포본에만 함께 실립니다. 상자를 쓰지 않는 시나리오만 담은 배포본에는 실리지 않습니다. 26.11 배포본은 넷이 모두 실려 함께 실립니다.

| 자산 | 출처 | 라이선스 | 배포 |
|---|---|---|---|
| `keti_hri_stage_v1` | KETI 자체 제작 — `generation/scene/build_stage.py` 가 USD 로 만듦. 위의 창고 · 조끼 입은 관절 인형 · 상자와 §3 의 H1 을 배포본에 실리는 이름으로 `assets/` 에서 상대 경로로 참조하고, 로봇 카메라 다섯(머리 앞 둘 · 머리 뒤쪽 · 양 손목) · 자리마다의 3인칭 카메라와 CCTV(`S1g` 는 사람이 오는 쪽마다 하나씩) · 출발 · 목표와 사람 자리의 표식(보이기만 함)을 더함. Isaac Sim 에서 여는 장면이고, 시나리오를 돌리는 것(대본 · 조건표 · 판정 · 기록)은 코드 저장소의 명령. 엮는 자산의 바이트 · 물리 · 텍스처 파일 · 표준 수치는 없음. 놓는 자리와 표식은 자체 설정 | CC BY 4.0 (참조하는 H1 은 §3 의 BSD-3) | **포함** (`assets/keti_hri_stage_v1.usd`) — 엮는 자산이 모두 실릴 때만 |

판정 근거 — 엮은 장면은 우리가 만든 USD 이고 다른 자산의 바이트를 담지 않습니다(참조만). 참조되는 파일은 각자의 행대로
실리고(H1 은 §3 의 BSD-3, 창고 · 인형 · 상자는 위의 CC BY 4.0), 엮은 것은 데이터셋과 같은 CC BY 4.0 입니다.

---

## 4. 재현 절차 (실행 환경을 동봉하지 않고 재현을 보장하는 방법)

데이터셋 이용자는 아래로 동일 자산과 실행 환경을 확보합니다.

1. Isaac Sim **6.0.1** 설치 — 정확한 판은 에피소드 메타의 `sim_backend` 에 있습니다 (`meta/episodes/**.parquet`. `meta/info.json` 이 아닙니다)
2. 로봇 자산: 배포본의 `assets/unitree_h1/` 에 실려 있습니다(`h1.usd` 가 같은 폴더의 `instanceable_meshes.usd` 를 참조). Isaac Sim 의 `Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/H1/h1.usd` 와 같은 바이트입니다
3. 무결성 대조: `meta/assets.json` 의 이 로봇 항목(`ship` — 파일마다 SHA256)과 `meta/joint_map/unitree_h1.yaml` 의 `provenance.usd_sha256`
4. KETI 자체 제작 자산은 데이터셋에 포함되어 있음

관절 이름·순서·한계값은 `meta/joint_map/` 에 이미 기록되어 있으므로, 자산을 구하지 못해도
**데이터 해석 자체는 가능**합니다.

---

## 5. 미결 및 리스크

| # | 항목 | 영향 | 대응 |
|---|---|---|---|
| 1 | **G1 29DOF 무손 구성이 BSD-3 경로에 없음** — `IsaacLab/Robots/Unitree/G1/` 은 23DOF+핸드(37DOF) 구성만 제공. 29DOF 무손은 MPL-2.0 경로에만 존재 | G1 로 옮기면 라이선스 재검토 필요 | 옮길 때 재확인. 데이터 생성에는 사용하지 않으며, 현재는 H1 의 `missing` 산출용 참조로만 씀 |
| 2 | 영상 모달리티 포함 시 렌더링 결과물의 지위 | 상위 계약상 "user generated content" 로 배포 가능하나 자산 외형이 드러남 | **26.11 배포본의 영상에는 외형이 나타납니다** — H1 로봇(BSD-3 경로, §2) · CC0 텍스처와 MakeHuman 메시, 우리가 만든 조끼와 상자 · 상자 라벨의 KETI CI(§3.1). **H1 은 렌더 결과에 더해 BSD-3 원본 파일도** §3 대로 `assets/unitree_h1/` 에 실립니다(라이선스 전문과 함께). CC0 텍스처와 MakeHuman 이 만든 시각 메시는 §3.1 대로 파일로도 실립니다(텍스처는 `assets/textures/`, 메시는 입힌 인형 USD 안). **하늘은 나오지 않습니다** — 씬은 사방이 막힌 실내 창고이고, Isaac Lab 기본 환경이 까는 돔 조명의 하늘(Nucleus 의 `…/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr`, 원본 Poly Haven CC0)은 `S1dEnvCfg` 가 빼며 26.11 배포본의 환경 `HriEnvCfg` 는 그 설정을 물려받습니다(2026-09-29 에 `S1d` 의 영상에서 확인). 조명은 씬 자산 안의 면광원과 텍스처 없는 채움광뿐이라 렌더에 들어가는 외부 조명 이미지는 없습니다(외부 파일이 전혀 없는 것은 아닙니다 — 시뮬레이션은 H1 을 Nucleus 에서 읽습니다) |
| 3 | Isaac Sim 6.0.1-rc.7 은 정식 릴리스가 아님 | 재현성 문서에서 버전 고정 근거가 약함 | 정식 릴리스 확인 후 필요 시 재생성. 관절 맵은 스크립트 재실행으로 갱신 가능 |

---

## 6. "비상업 목적" 논리를 쓰지 않는 이유

배포 라이선스 CC BY 4.0 은 상업적 이용을 허용하므로, 재배포 권한 없는 자산을 포함하면
**보유하지 않은 권리를 하위 이용자에게 부여**하는 결과가 됩니다. 「비상업 목적이니 괜찮다」는
논리는 그래서 성립하지 않습니다.

방어 논리는 "확인하고 허용 범위 내에서 배포했다" 입니다.

---

## 7. 표준 임계값의 재배포 근거

자산은 아니지만 **같은 질문을 받는 항목**이라 여기에 함께 적습니다. `NOTICE` 와
`meta/assets.json` 이 이 문서를 라이선스 판정의 근거로 가리키므로, 배포본에 실린
것 가운데 출처 확인이 필요한 것은 이 문서 하나로 따라갈 수 있어야 합니다.

| 항목 | 내용 |
|---|---|
| 대상 | `meta/rules.json` `contact_limits` 의 **임계값 70개** — 12개 인체 부위의 힘·압력·에너지 한계, 동적 배수, 유효질량, 탄성계수 |
| 출처 | KS B ISO TS 15066:2016 (2022 확인) 부속서 A 표 A.2·A.3·A.4 |
| 근거 | **2026-08-27 국가기술표준원(저작권자)의 이용 허락. 조건·단서 없음** |
| 형태 | 값과 **조항 참조를 함께** 싣습니다. 각 항목의 `standard_ref` 가 어느 표의 어느 값인지 가리킵니다 |
| 배포 조건 | CC BY 4.0 — 데이터셋과 같습니다 |

표준 열람본에는 「이 문서는 저작권 규정에 따라 상업적 활용을 금지합니다」 표시가 있고
본 데이터셋은 상업적 이용을 허용하므로, 값을 싣는 근거는 위의 이용 허락입니다.

과제 정본(저장소의 `meta/rules.json`)에는 **아직 값 없이 조항 참조만 실린 임계값이 1개 있습니다** — 이격거리 산정식의 **침범
거리**입니다. 이유는 허락도, 표준을 못 본 것도 아닙니다. 그 값을 정하는
**KS B ISO 13855** 를 2026-09-01 에 확인했는데, 그 표준이 침범 거리를 **상수로 주지
않습니다** — 방호설비가 무엇을 감지하느냐에 따라 달라지는 값이고, 어느 구성을 본 과제의
계측에 대응시킬지가 판단으로 남았습니다. 그 값은 요구 이격거리를 통해 생성 시점에
궤적으로 들어가므로 정하는 순간 재생성이 따라옵니다.

그 임계값은 `R-TS15066-SSM-01` 의 것이고, 26.11 배포본은 그 규칙을 싣지 않습니다(담은 시나리오가 선언하지 않습니다).
그래서 이 배포본의 `meta/rules.json` 에는 조항 참조만 실린 임계값이 없고, 값이 실린 표준 임계값은 위 「대상」의 70개가
전부입니다.

**같은 표준이 정하는 인체 접근 속도는 과제 정본에 값이 실려 있습니다.** 그 표준은 이
값을 상수로 줍니다. 위 표의 「대상」이 접촉 한계 70개인 것과 별개이며, 출처도
KS B ISO TS 15066 이 아니라 **KS B ISO 13855** 입니다. 이 값도 `R-TS15066-SSM-01` 의 것이라 26.11 배포본의
`meta/rules.json` 에는 없습니다.

**표준 문서 자체나 표 전체를 복제하지는 않습니다.** 판정 체계가 참조하는 임계값만
조항 인용과 함께 옮겨 적은 형태이고, 본문·그림·설명은 옮기지 않습니다.

**그중 v0.1 라벨이 실제로 계산에 쓰는 것은 22개**입니다 — 부위별 준정적 힘 12개와
적용 가능한 동적 배수 10개. 나머지 48개(압력·전달 에너지·유효질량·탄성계수)는
**이 배포본의 판정에 쓰이지 않습니다.** 그래도 싣는 이유는 둘입니다.

- 이용자가 **우리가 쓰지 않은 축으로** 같은 데이터를 다시 판정할 수 있어야 합니다.
  압력은 접촉 면적을, 에너지는 힘과 같은 시점의 상대속도를 요구하는데 v0.1 이 그
  둘을 갖추지 못했을 뿐, 한계값 자체는 같은 표에서 옵니다
- 한 부위의 한계는 힘·압력·에너지가 **한 벌로 정의**됩니다. 쓰는 것만 골라 실으면
  받는 쪽이 어느 축이 왜 빠졌는지 알 수 없습니다

각 항목은 `unused_in_v0_1` 로 「이 배포본에서는 쓰지 않는다」를 스스로 밝힙니다.
