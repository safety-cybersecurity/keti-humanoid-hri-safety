# 생성 파이프라인

Isaac Sim / Isaac Lab 으로 에피소드를 생성하고, 라벨을 매기고, LeRobotDataset v3.0 으로 변환 · 조립해
Hugging Face Hub 에 게시하는 코드입니다.

- **생성**에는 Isaac Sim 과 Isaac Lab 이 필요합니다 — 버전은 저장소 루트의 `environment.lock` 에 있습니다. CI 에서는 돌리지 않습니다
- **라벨링 · 변환 · 조립 · 게시**는 Isaac Sim 없이 돕니다(아래 「환경이 둘로 나뉘어 있습니다」)
- 받은 배포본을 **검증**하는 것은 `validator/` 이고 이 디렉터리 없이 돕니다

처음이면 「환경이 둘로 나뉘어 있습니다」 → 「`envs/s1a/` — 실행과 실측」 → 「S1d — 자산을 만들고 실행하기」 → 「S1e · S1f · S1g — 장면 · 자산 · 카메라를 만들고 확인하기」 → 「재현은 실행
단위로만 됩니다」 순서로 읽으십시오. 받은 배포본에서 라벨을 다시 유도해 보려면 「실린 라벨은 배포본만으로 다시 나옵니다」입니다.

| 자리 | 하는 일 |
|---|---|
| `dump_joint_names.py` | 자산에서 관절 이름 · 순서를 추출 → `meta/joint_map/*.yaml` |
| `capture_environment.py` | 실행 환경의 버전을 수집 → `environment.lock` |
| `dummy/` | ISO/TS 15066 의 부위 구분에 따른 사람 인형(강체 17개 · 관절 인형은 발 둘을 더해 19개 — 그 걸음 표도)과 그 시각 메시(`dummy/skin/`) |
| `scene/` | 물류 통로 씬의 빌더와 배치 측정 도구 |
| `envs/` | Isaac Lab 환경(`s1a/` · `s1d/` · 조건표가 있는 시나리오의 `hri/`)과 실행기 `run_s1a.py` |
| `conditions.py` · `human_script.py` · `task_path.py` · `behavior.py` · `hri_geometry.py` · `carry.py` · `robots/` | 조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)의 조건표와 가드 · 사람 대본 · 과제의 길과 완료 · 행동 층 · 사람과의 거리와 방향을 따진 S_p · 든 상자의 자리 · H1 의 관절 틀과 순기구학 — Isaac 없이 돌고 시험됩니다 |
| `labeling/` | 기록된 물리량 → 판정(`meta/rules.json` 적용)과 그 근거 요약(`labeling/evidence.py`), 그리고 배포본만으로 라벨과 근거 요약을 다시 유도하는 도구 |
| `episode_spec.py` | 조건표 시나리오의 에피소드 명세 `episode_spec`(멤버 여섯 — 사이드카와 정본에서 유도) |
| `convert_lerobot.py` | 녹화 → LeRobotDataset v3.0 |
| `run_batch.py` · `assemble_batch.py` | 배치를 청크로 생성하고 하나로 조립 |
| `publish_dataset.py` | 게시 전 점검 · 업로드 · 왕복 대조 · 태그 |
| `rerun_seed.sh` · `rerun_policy.py` | 검증기 C9 의 재현 훅과, 조건표가 있는 시나리오의 에피소드를 기록 때의 보행 정책 · 행동 층으로 다시 돌릴 인자 |

## `dump_joint_names.py`

자산에서 관절 이름·순서를 추출해 `meta/joint_map/{robot_model}.yaml` 을 생성합니다.
검증기와 달리 Isaac Sim 이 필요하며, 번들 파이썬으로 실행합니다.

```bash
# H1 — G1 과 대조해 missing 까지 채운다
~/isaacsim/python.sh generation/dump_joint_names.py \
    --usd ~/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/Unitree/H1/h1.usd \
    --robot-model unitree_h1 \
    --missing-vs meta/joint_map/unitree_g1_29dof.yaml

# G1 — 기본 배리언트는 3지 핸드 포함 43DOF 이므로 손을 빼야 29DOF 가 된다
~/isaacsim/python.sh generation/dump_joint_names.py \
    --usd ~/isaacsim_assets/Assets/Isaac/6.0/Isaac/Robots/Unitree/G1/g1.usd \
    --robot-model unitree_g1_29dof \
    --variant left_hand=None --variant right_hand=None
```

출력의 `unmapped` · `conflicts` 가 비어 있지 않으면 매핑이 깨진 것이고,
`needs_review` 는 이름만으로 회전축을 판별하지 못한 관절입니다.

**USD 를 파싱하는 것만으로는 부족합니다.** 상태 배열이 따르는 순서는 물리엔진이
아티큘레이션을 파싱한 뒤의 순서이고 USD 순회 순서와 다를 수 있어, 이 도구는 타임라인을
재생해 물리 텐서 뷰가 초기화된 뒤에 읽습니다. 또한 `joint` 와 `DOF` 는 다릅니다
(스키마 §7.3.1) — 이 저장소가 쓰는 H1 사본(`IsaacLab/Robots/Unitree/H1/h1.usd`)은 관절 19개가
모두 DOF 이지만, 다른 사본 `Robots/Unitree/H1/h1.usd` 는 관절 24개 중 5개가 센서용 고정관절이라
DOF 는 19개입니다.

## `dummy/`

`build_dummy.py` 가 `meta/rules.json` 의 `contact_limits` 와 `dummy/anthropometry.yaml` 을
읽어 부위별 강체 17개(부위 12종, 좌우 한 쌍 5종)를 가진 더미 USD 를 만듭니다.
어느 수치도 코드에 넣지 않습니다.

```bash
~/isaacsim/python.sh generation/dummy/build_dummy.py                  # 임계값 없으면 실패(코드 2)
~/isaacsim/python.sh generation/dummy/build_dummy.py --geometry-only  # 형상만. 잠정 표시가 박힘
~/isaacsim/python.sh generation/dummy/check_contact_resolution.py     # 부위 분해 확인
```

임계값이 비어 있으면 만들지 않습니다. 값이 빠진 채로 만든 더미의 접촉 하중은 표준과
아무 관계가 없는데도 관계가 있는 것처럼 보이기 때문입니다. 산출물은 `*.usd` 라 저장소에
담지 않으며, 정본은 스크립트와 두 데이터 파일입니다.

`check_contact_resolution.py` 는 부위마다 프로브 구를 정면에 겹쳐 놓고 시뮬레이션해, 접촉
보고가 BR 코드로 되돌아오는지 확인합니다. 시나리오 카탈로그 §9-2 에 대한 시험이며
**자산 그대로의 아티큘레이션 구성에서 17/17 통과합니다** (2026-08-19).

### 조인트 로컬 프레임은 부모의 배율 안에 있다

조인트의 상대 위치를 월드 기준으로 적으면 **부위가 저작한 자리에 놓이지 않습니다.** 겉으로는
「아티큘레이션 구성에서는 프로브 접촉이 보고되지 않는다」로 보이지만 접촉 보고는 정상입니다.

`physics:localPos0` 은 월드 좌표가 아니라 **body0 의 로컬 프레임** 값입니다. 몸통 부위
(BR-04·05·06·07 등)는 단위 상자를 `xformOp:scale` 로 키워 만들므로 그 로컬 프레임에는
배율이 들어 있고, 월드 기준 상대 위치를 그대로 적으면 물리엔진이 배율만큼 줄여 읽습니다.

| 부위 | 저작한 오프셋 | 부모 배율 | 물리엔진이 읽은 값 |
|---|---|---|---|
| BR-05 ← BR-04 | (0.05, 0, −0.12) | (0.18, 0.4, 0.16) | (0.009, 0, −0.0192) |
| BR-08 ← BR-04 | (0.03, 0.21, −0.14) | (0.18, 0.4, 0.16) | (0.0054, 0.084, −0.0224) |

`attach_stage` 직후에는 저작한 자세 그대로지만, **첫 스텝에 아티큘레이션이 루트에서
정기구학으로 다시 놓이면서** 16개 부위가 몸통 안으로 접힙니다. 정강이(BR-12)는 60 cm 를
올라와 몸통 높이에 겹칩니다. 프로브는 저작 위치 기준으로 배치되므로 대부분 아무것도
만나지 못하고, 몸통 근처 두 개만 엉뚱한 부위를 보고합니다.

그래서 부모의 월드 변환 역으로 옮겨 적습니다(`build_dummy.py`). 회전이 붙어도 같은 식이
성립합니다.

**조인트가 붙었는지는 중력을 켜야 알 수 있습니다.** 접촉 분해 시험은 중력을 끄고 도는데,
그 상태에서는 조인트가 하나도 붙지 않아도 부위가 제자리에 있으므로 "자세가 유지된다" 가
아티큘레이션의 증거가 되지 못합니다. PhysX 가 고정조인트 16개를 전부 거부한 구성에서도
그 시험은 17/17 을 냅니다.
루트만 월드에 고정하고 중력 1초를 돌려 전 부위가 제자리인지 보면 갈립니다 —
2026-08-19 확인 시 최대 이동 0.00 mm 였습니다. **이 확인은 아직 스크립트에 들어 있지
않습니다.** 조인트 구성을 손대면 별도로 한 번 돌려 보십시오.

**자가 충돌은 자산이 아니라 씬 구성 쪽에서 꺼야 합니다.** `UsdPhysics.CollisionGroup` 이
자기 자신을 거르게 하면 그룹 구성원이 아무것과도 충돌하지 않아 로봇이 더미를 그대로
통과합니다(실측). 자산은 `keti:requiresSelfCollisionDisabled` 로 요구사항만 표시하고,
씬을 만드는 쪽이 아티큘레이션의 self-collision 을 끕니다.

같은 BR 코드의 프림이 둘인 부위는 라벨링에서 **합산이 아니라 최대값**으로 모아야 합니다.

빌더가 요구하는 유효질량 · 스프링상수 24건(표 A.3)은 `meta/rules.json` 의
`contact_limits.*.biomechanical` 에 값과 조항 참조가 함께 실려 있습니다. 그래서 `--geometry-only`
없이 그대로 돕니다.

**Isaac Lab 씬에 아티큘레이션으로 올라갑니다** (`generation/envs/s1a/check_scene.py`,
2026-08-19 확인). 고정조인트 16개짜리 0-DOF 아티큘레이션을 Isaac Lab 의 `Articulation` 이 거부하지
않고, 링크 17개·DOF 0개로 올라가 10 스텝을 완주합니다. kinematic 강체 묶음
(`RigidObjectCfg` + `--root kinematic`)으로 내려갈 필요가 없습니다. 자가 충돌이 실제로
꺼졌는지는 위치 변위만으로 판정할 수 없어(고정조인트가 상대 운동을 막으므로 켜져 있어도
똑같이 균일하게 보임) 같은 스크립트가 10 스텝 뒤 부위별 접촉력도 함께 확인합니다 — 접지
부위(BR-12, 하퇴) 외에는 전부 힘이 잡히지 않아야 통과입니다.

## `envs/s1a/` — 실행과 실측

`generation/envs/register.py` 가 `Keti-S1a-v0` 로 등록하고, `generation/envs/run_s1a.py` 가
사전학습 정책(`Isaac-Velocity-Flat-H1-v0`)으로 에피소드를 생성합니다. 정책은 학습하지
않으므로 관찰 배열은 상속받은 그대로 두고 씬만 늘립니다.

```bash
~/isaacsim/python.sh generation/envs/run_s1a.py --episodes 1            # seq 0 부터, 기본 1000 스텝
~/isaacsim/python.sh generation/envs/run_s1a.py --seq 10 --episodes 10 --out .tmp/episodes/S1a
```

에피소드당 `<episode_id>.npz`(스텝 시계열) + `<episode_id>.json`(메타) 두 파일을 냅니다.

### 화면에서 보면서 녹화하기

배치는 헤드리스로 만듭니다. 같은 스크립트에 플래그를 붙이면 보면서 돌리거나 영상을 남깁니다 —
플래그가 없으면 배치를 만드는 경로 그대로입니다.

```bash
~/isaacsim/python.sh generation/envs/run_s1a.py --gui --video --seq 0 --episodes 1
~/isaacsim/python.sh generation/envs/run_s1a.py --video --seq 0 --episodes 3   # 창 없이 mp4 만
```

| 플래그 | 무엇 |
|---|---|
| `--gui` | Kit 창을 열어 보면서 돌립니다. 카메라는 로봇 골반을 따라갑니다 (뒤·옆·위에서 전방을 봄) |
| `--video` | 에피소드마다 npz 옆에 **같은 이름의 mp4** (H.264 · 50 fps · 1280×720). 창이 없어도 됩니다 |
| `--speed` | `--gui` 의 재생 배속 **상한**. 기본 1.0 = 실시간, 0 = 기다리지 않음. 한 스텝의 계산·렌더가 그보다 느리면 그만큼 느려지며, 에피소드 끝에 실제 배속을 인쇄합니다 |
| `--free-camera` | 추적을 끄고 마우스로 돌립니다. 영상도 그 시점을 담습니다 (Kit 캡처가 뷰포트 카메라를 읽으므로) |
| `--compare-with DIR` | 끝난 뒤 생성된 npz 를 그 배치의 같은 이름과 대조합니다. C9 훅과 같은 배열을 같은 기준으로 봅니다 (NaN 은 같은 자리면 동일, timestamp 제외). C9 훅이 함께 대조하는 에피소드 메타(사이드카에서 옮겨 싣는 랜덤화 · 종료 사유 · 홀드 해제 스텝 등 — `validator/README.md` 「저장소 훅이 대조하는 에피소드 메타」)는 보지 않습니다. 카메라 배열(상자 · 카메라 자세)은 **두 사이드카가 모두 `media` 를 가질 때만** 더하고, 줄마다 실제로 대조한 배열 수를 적습니다. 불일치가 있거나, 대조한 에피소드나 사이드카가 없으면 **exit 3** |

**`--out` 이 없으면 정본 배치가 아니라 `.tmp/episodes/S1a-watch-<시각>/` 에 씁니다.**
recorder 는 같은 이름의 파일을 묻지 않고 덮어쓰므로, 배치 디렉터리를 기본값으로 둔 채
`--gui` 를 붙이면 그 배치의 에피소드 0 이 조용히 덮입니다. **플래그 없는 헤드리스
실행의 기본값은 배치 디렉터리입니다.** 실행 전에
`--out` 을 보십시오. 그 디렉터리는 부팅 전에 만들어 잡으며 이미 있으면 `-2`, `-3` 을
붙입니다 — 같은 초에 시작한 두 실행이 한 디렉터리를 나눠 쓰지 않게. `--compare-with` 가
출력 디렉터리 자신이면(심볼릭 링크 포함) 부팅 전에 거부합니다 — 자기 자신과의 대조는
늘 통과라 아무것도 증명하지 못합니다.

**이것은 재생이 아니라 재시뮬레이션입니다.** 저장된 궤적을 그리는 것이 아니라 같은 시드로
다시 돌립니다. 재현은 실행 단위로만 성립하므로(아래 「재현은 실행 단위로만 됩니다」)
배치의 37번을 그대로 보려면 그 에피소드를 만든 실행을 첫 번호부터 돌려야 합니다 — 사이드카의
`run_start_seq` 가 0 이면 `--seq 0 --episodes 38`, 30 이면 `--seq 30 --episodes 8` 입니다.
`--speed 0` 이면 앞 에피소드를 빨리 넘깁니다.

**겉모습은 실행 앞에서 뽑고 렌더할 때만 입힙니다.** 조건표 시나리오(`S1e` · `S1f` · `S1g`)의 실행은 부팅 전에 정본 `appearance` 에서 그 행의 겉모습을
`SHA-256("<시나리오>:appearance:<행>")` 시드로 뽑아(`generation/appearance.py`) 사이드카의 `randomization.appearance` 에 적습니다 — 렌더하지 않는 실행도
같은 값을 뽑아 적습니다. 카메라 · `--video` · `--gui` 로 렌더하는 실행만 그 값을 첫 프레임 전에 한 번 장면에 입히고
(`generation/envs/hri/appearance_apply.py`), 물리 배열 · 라벨 · 사람 상자 · 카메라 자세는 바뀌지 않습니다. `--no-appearance` 는 아무것도 입히지 않고
`null` 을 적으며 겉모습을 켠 실행과 끈 실행을 견주는 용도입니다.

**렌더링이 궤적을 바꾸는지 쟀습니다 (2026-09-03).** 서브스텝 센서 읽기가 궤적을 바꾼
전례가 있어(「서브스텝 표집은 시뮬레이션을 건드립니다」) 가정하지 않고 `--compare-with`
로 대조했습니다.

| 실행 | 배치의 같은 에피소드와 |
|---|---|
| 플래그 없음, `--seq 0 --episodes 2` | **C9 대조 배열 전부 동일** 2/2 |
| `--video` (헤드리스 캡처) `--seq 0 --episodes 1` | **전부 동일**, 699 프레임 = 기록 699 스텝 |
| `--gui --video --seq 0 --episodes 1` (창 열림, 실시간 배속) | **전부 동일**, 699 프레임 = 기록 699 스텝. 부팅부터 종료까지 82 초 |
| `--video --seq 0 --episodes 9` (헤드리스) | 아홉 편 **전부 동일**. 745 초 — 녹화는 에피소드당 약 80 초로 헤드리스 생성의 세 배쯤 |

모두 같은 궤적입니다 — 이 배치에서는 렌더링도 캡처도 창도 물리에 닿지 않았습니다.
「동일」은 C9 훅이 대조하는 배열이 같다는 뜻입니다 — NaN 은 같은 자리면 같다고 보고
`+0.0`/`-0.0` 을 가르지 않으며 timestamp 는 보지 않으므로, 파일의 바이트 동일성보다
약한 진술입니다. 그리고 GPU 시뮬레이션의 재현성은 실행 단위 주장입니다.
보고 있는 것이 배치의 그 에피소드인지가 중요하면 `--compare-with` 를 붙여 매번 재십시오.

**mp4 는 닫은 뒤 파일에서 프레임을 다시 셉니다.** imageio_ffmpeg 는 ffmpeg 가 비정상
종료해도 조용히 돌아오므로, 쓴 프레임 수와 파일의 프레임 수가
다르면 실행이 실패합니다. 프레임이 없는 실행은 mp4 를 남기지 않고 옆의 옛 mp4 도
지웁니다 — npz 가 다른 실행의 영상과 짝지어지지 않게.

창이 뜨면 Isaac Sim 이 「Simulation Output Settings」 패널을 뷰포트 위에 겹쳐 띄웁니다.
Fabric GPU 출력을 알리는 Kit 자체 패널이고 닫아도 됩니다.

기전: Kit 캡처는 `/OmniverseKit_Persp` 카메라를 replicator 렌더 프로덕트로 읽습니다.
Isaac Lab 의 `VideoRecorder` 는 그 카메라를 **첫 프레임에 한 번만** 놓으므로 20 m 를
걷는 로봇이 화면 밖으로 나갑니다 — `watch.following_video_recorder` 가 프레임마다
로봇 루트 위치에 오프셋을 더해 다시 놓습니다. 창의 뷰포트는 Isaac Lab 의
`ViewportCameraController` 가 같은 오프셋으로 따라갑니다(`watch.track_robot`).
이 Isaac Lab 판(v3.0 beta)에서는 `headless=False` 만으로는 창이 안 뜨고
`visualizer=["kit"]` 을 줘야 합니다 — `watch.launcher_args` 가 그 인자를 만들고
시험이 「플래그 없음 = `{"headless": True}`」를 못박습니다.

**시드는 에피소드마다 파생됩니다.** `meta/scenarios.json` 의 `seed_rule`
(`seed = hash(scenario_id, episode_seq)`)을 `generation/seeding.py` 가 구현하고,
`run_s1a.py` 가 **각 에피소드의 리셋 직전에** 그 값으로 env 를 다시 시드합니다.
`--seed` 는 없습니다 — 시드를 손으로 주면 파일명과 정본 규칙이 어긋납니다.

### S1d — 자산을 만들고 실행하기

시나리오 `S1d` 는 S1a 의 진입 구도를 자체 제작 물류 통로 씬에 놓고, 사람 인형에 시각 메시를
입히고, 카메라(`meta/cameras.json` — 일곱 대. 400건 배치는 머리 뒤쪽 카메라 없이 여섯 대로 기록했습니다)의 영상을 함께 기록합니다(카탈로그 §5). 환경은 `generation/envs/s1d/` 이고 실행기는 S1a 와 같습니다.

```bash
# 자산 — 텍스처(CC0, 해시 대조) · 상자 라벨의 KETI CI(해시 대조) · 씬 · 시각 메시(Blender 와 MPFB 필요) · 메시를 입힌 인형
.venv/bin/python generation/scene/fetch_textures.py
.venv/bin/python generation/scene/fetch_keti_ci.py              # 받아 둔 압축 파일이면 --archive <KETI_CI.zip>
~/isaacsim/python.sh generation/scene/build_logistics_scene.py
~/isaacsim/python.sh generation/scene/check_scene_build.py      # 빌드한 USD 를 배치 · 명세와 대조
.venv/bin/python generation/dummy/skin/make_skin.py
~/isaacsim/python.sh generation/dummy/build_dummy.py            # 입히지 않은 인형 — 아래 대조의 기준
~/isaacsim/python.sh generation/dummy/build_dummy.py --skin generation/dummy/skin/build/keti_skin.obj
~/isaacsim/python.sh generation/dummy/check_skin_physics.py     # 입힌 인형이 입히지 않은 것과 엔진 값으로 같은지

# 실행 — 기본 출력은 .tmp/episodes/S1d
~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1d --seq 0 --episodes 1
~/isaacsim/python.sh generation/envs/s1d/check_s1d_scene.py     # 카메라 일곱 대의 모양 · 투영, 인형의 방향과 붙잡기
```

씬의 치수는 `generation/scene/logistics_aisle.yaml`, 시각 메시는 `generation/dummy/skin/skin_spec.yaml`, 카메라는
`meta/cameras.json` 이 정본입니다. **두 YAML 은 바이트가 통째로 해시되어 빌드한 자산에 구워집니다** — 주석 한 줄만 고쳐도
자산의 해시가 달라지고 `meta/assets.json` 의 해시를 함께 고쳐야 합니다. 카메라의 장착 규칙을 고쳤으면
`generation/envs/s1d/extract_camera_mounts.py --write` 로 오프셋을 자산에서 다시 유도합니다(손으로 적지 않습니다).
3인칭과 CCTV 는 시나리오의 자리마다 `meta/cameras.json` 의 `sites` 에 놓입니다(새 시나리오의 자리는 아래 절).
정본을 고친 뒤 어디부터 다시 돌려야 하는지는 SOP §10 에 있습니다.

### S1e · S1f · S1g — 장면 · 자산 · 카메라를 만들고 확인하기

새 시나리오 셋(`S1e` · `S1f` · `S1g` — 카탈로그 §5)은 S1d 와 같은 창고를 씁니다. 창고는 랙 바닥 단에 팔레트와
큰 골판지 상자를 두고 선반 상자와 그 큰 상자마다 y 를 보는 두 옆면에 흰 라벨과 KETI CI 를 붙였고(치수는 자체 설정 —
`generation/scene/logistics_aisle.yaml` 의 `rack.floor_loads` · `labels`), S1d 400건 배치는 그 짐과 라벨 없이 기록했습니다. 사람은 조끼를 입힌 관절 인형, 로봇은 배포본에 폴더째 실리는 H1 이고, `S1e` · `S1f` · `S1g` 의 로봇은 KETI CI 를
붙인 상자를 듭니다. 자리(출발 · 목표 · 길 · 사람 구역)는 `meta/scenarios.json` 의 `scene.site`, 3인칭과 CCTV 는
`meta/cameras.json` 의 `sites` 가 정본이고(`S1g` 의 둘은 사람이 오는 쪽마다 하나 — `by_approach_side`, 환경 설정의
`camera_approach_side` 가 조건표 행의 `approach_side` 로 고름), 값은 자체 설정입니다. 환경은 `generation/envs/hri/` 이고 실행기는 S1d 와 같습니다 —
한 프로세스가 조건표(`meta/scenarios.json` 의 `conditions`)의 한 행이고 행 번호가 `--seq` 입니다(맨 아래 블록). 아래는
자산을 만들고 확인하는 순서이고, 깨끗한 클론에서 위에서부터 돌립니다. `~/isaacsim/python.sh` 로 돌리는 줄이 Isaac Sim 의
파이썬(pxr · Kit)이 필요한 단계이고, 나머지는 `.venv` 로 돕니다.

```bash
# 받기 — 정본이 고정한 해시와 같은 바이트만 놓습니다(놓는 곳은 추적하지 않는 디렉터리)
.venv/bin/python generation/scene/fetch_textures.py             # 씬 텍스처(CC0)
.venv/bin/python generation/scene/fetch_keti_ci.py              # 상자 라벨(창고 · 들고 다니는 상자)의 KETI CI — 받아 둔 압축 파일이면 --archive <KETI_CI.zip>
.venv/bin/python generation/robots/fetch_robot.py --robot unitree_h1 --nucleus <Isaac Lab 자산 루트>
                                                                # H1 폴더(h1.usd · instanceable_meshes.usd · LICENSE.txt) — 루트는 https 주소 또는 그 사본 디렉터리

# 빌드 — 씬 · 인형 · 상자
~/isaacsim/python.sh generation/scene/build_logistics_scene.py
~/isaacsim/python.sh generation/scene/check_scene_build.py      # 빌드한 USD 를 배치 · 명세와 대조
.venv/bin/python generation/dummy/skin/make_skin.py             # 피부 내보내기(Blender 와 MPFB 필요 — v1 입힌 인형과 함께 씀)
.venv/bin/python generation/dummy/gait/build_gait.py --archive <100STYLE.zip>     # 걸음 표 — 원본 묶음은 NOTICE §2 의 주소
~/isaacsim/python.sh generation/dummy/build_dummy.py --jointed --gait generation/dummy/build/keti_gait_100style.json
                                                                # 맨몸 관절 인형 — 물리 대조의 기준(배포본에는 싣지 않음)
~/isaacsim/python.sh generation/dummy/build_dummy.py --jointed --gait generation/dummy/build/keti_gait_100style.json \
    --skin generation/dummy/skin/build/keti_skin.obj
                                                                # 피부만 입힌 관절 인형(조끼 없음) — S1d 환경을 관절 인형으로 돌리는
                                                                # 확인 도구(check_jointed_walk.py)가 띄움(배포본에는 싣지 않음)
~/isaacsim/python.sh generation/dummy/build_dummy.py --jointed --gait generation/dummy/build/keti_gait_100style.json \
    --skin generation/dummy/skin/build/keti_skin.obj --vest generation/dummy/vest.yaml --seams generation/dummy/skin_seams.yaml
                                                                # 조끼 입은 관절 인형 — 같은 입력의 맨몸을 따로 지어 물리를 대조 ·
                                                                # 피부 이음새의 섞는 비율을 피부 메시에 실음(실행이 렌더마다 섞음)
~/isaacsim/python.sh generation/scene/carried_box.py            # 들고 다니는 상자 — 받아 둔 KETI CI 를 USD 옆에 복사

# 자산에서 재는 값 — 인자 없이 돌리면 정본과 대조합니다(H1 이나 장착 규칙을 바꿨으면 --write 로 쓰고 다시 대조).
# 둘 다 fetch_robot.py 가 받아 둔 폴더가 아니라 시뮬레이션이 띄우는 H1(환경 설정의 spawn 주소 — Isaac Lab 자산 루트)을 읽고,
# extract_camera_mounts.py 는 Isaac Sim 자산 루트의 공식 사본(MPL-2.0 경로)도 읽습니다 — 그 자산 팩이 설치되어 있거나 주소가 열려야 합니다
~/isaacsim/python.sh generation/robots/extract_link_boxes.py    # 「봄」이 쓰는 H1 링크마다의 가림 상자
~/isaacsim/python.sh generation/envs/s1d/extract_camera_mounts.py   # 로봇 몸의 카메라 다섯의 자리(머리 뒤쪽 포함)

# 자리마다 3인칭 · CCTV 가 사람과 로봇을 보는지 — 창고 레이아웃 위의 시선으로(Isaac 없음 · S1g 는 사람이 오는 쪽마다)
.venv/bin/python generation/scene/check_site_cameras.py

# 엮은 장면(Stage) — 위의 자산이 모두 있어야 합니다
~/isaacsim/python.sh generation/scene/build_stage.py            # 받아 둔 H1 에서 서는 높이를 잼 · 정본에 고정할 값 셋을 찍음
~/isaacsim/python.sh generation/scene/check_stage.py            # 배포본의 assets/ 처럼 놓고 열어 봄 · 다시 지어 바이트 대조
~/isaacsim/python.sh generation/scene/look_stage.py --out <디렉터리>   # 카메라마다 한 장 렌더(보기용 — 검사가 아님)

# 조건표 — 시나리오의 표 전체를 파일로 봄(Isaac 없음 · 행 i 가 생성의 --seq i · 가드에 걸린 시도는 attempt 열에)
.venv/bin/python -m generation.conditions --scenario S1e --out <csv>

# 실행 — 조건표의 한 행(한 프로세스 · --episodes 1). 기본 출력은 .tmp/episodes/<시나리오>
~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1e --seq 0
~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1f --seq 0 --policy <체크포인트>
                                                                # 보행 정책을 바꿈 — H1 형식 rsl_rl 체크포인트(기본은 Isaac Lab 이 게시한 것)
~/isaacsim/python.sh generation/envs/run_s1a.py --scenario S1g --seq 0 --behavior <모듈:이름>
                                                                # 과제 명령과 보행 정책 사이의 행동 층을 바꿈
                                                                # (기본 generation.behavior:PassThrough — 과제 명령 그대로)
```

`--policy` 와 `--behavior` 는 에피소드의 `sim_backend.policy` 에 남습니다(체크포인트의 출처와 SHA-256 · 행동 층의 `모듈:이름`). 행동 층은
`generation.behavior.BehaviorInput` 을 받아 (앞속도, 요 속도)를 돌려주는 호출 가능한 것이면 됩니다 — 시각 · 로봇의 자세와 속도 · 과제 명령과
단계 · 로봇 카메라 다섯 가운데 하나라도 사람을 「봄」이면 로봇에서 본 그 사람의 상대 위치와 상대 속도(월드 축 · 행의 `sensor_delay_ms`
만큼 늦게 · 안 보이면 `None`)를 받습니다. 기록 데이터는 기본값으로 만듭니다 — 사람에 반응하지 않는 기준 로봇입니다.

검증기 C9 의 재현 훅은 이 기록으로 에피소드를 다시 돌립니다(`generation/rerun_policy.py` 가 배포본의 행에서 읽어 인자로 바꿈) —
게시된 체크포인트면 `--behavior` 만, 저장소 안의 체크포인트 파일이면 SHA-256 이 기록과 같고, 불리는 이름 그대로 git 이 추적하며 그 바이트가
HEAD 의 것일 때 `--policy` 와 함께. 저장소 밖의 체크포인트는 기록에 파일 이름만 남아 다시 돌릴 수 없고, 훅은 시뮬레이터를 띄우기 전에 exit 2 로
끝납니다. 행동 층의 모듈도 — import 가 먼저 도는 부모 패키지의 `__init__.py` 까지 — 불리는 이름 그대로 git 이 추적하고 바이트가 HEAD 의 것인
저장소 파일이어야 하고, 그 곁에 import 가 대신 집을 바이트코드 · 확장 모듈이 없어야 합니다(재현 실행이 그것을 import 함 — 설치된 모듈 · 저장소 밖의
파일 · 추적하지 않는 파일이나 링크 · 커밋하지 않은 변경이 있는 파일(인덱스의 표시와 상관없이)이면 같은 exit 2이고, git 체크아웃의 뿌리가 아닌 저장소도
같습니다. 배포본이 적은 해시는 그 배포본이 주는 것이라 닻이 되지 못하고, 닻은 저장소의 git 이력입니다) — 재현을 확인받을 데이터는 게시된
체크포인트나 저장소에 커밋한 파일로, 저장소에 커밋한 모듈인 행동 층으로 만듭니다.

다시 돌린 에피소드는 배포본이 싣는 스텝 열과 대조합니다(`generation/compare_published.py`). 배포본 자신의
`meta/safety_schema.json` 이 선택 열(`optional_step_fields`)로 적은 열을 그 배포본이 싣지 않았으면 대조하지 않고
성공 줄에 그 수를 적습니다 — 선택 열은 배포본이 뺄 수 있는 열입니다. 규격의 필수 · 조건부 목록에 든 열이 없으면 대조가
실패하고, 실린 선택 열은 다른 열처럼 비트로 대조합니다. 선택 열을 그 배포본의 시나리오(`required_fields`)나
규칙(`inputs`)이 요구하는지는 이 대조가 묻지 않습니다 — 그런 열이 빠진 배포본은 검증기 C8 이 오류로 말합니다.
배포본에 `meta/safety_schema.json` 이 없거나, 읽히지 않거나, 그 파일이 이 목록을 적지 않으면 대조하는 열을 모두 요구합니다.

관절 인형의 피부는 한 장의 메시를 부위 몸마다 갈라 붙인 것이라 관절이 굽으면 갈린 가장자리가 떨어집니다. `--seams` 로 지은 인형은 이음새 근처의
피부 정점마다 이웃 몸과 섞는 비율을 싣고(`generation/dummy/skin_seams.py`), 시나리오 실행은 렌더할 때마다 그 정점을 몸들의 지금 자세 사이로 옮깁니다
(`generation/envs/s1d/skin_blend.py` — 영상과 사람 상자에만 들고 물리 · 기록 배열은 그대로입니다).

빌드한 자산의 해시는 `meta/assets.json` 이 고정합니다 — 명세(`generation/dummy/vest.yaml` · `generation/dummy/skin_seams.yaml` 처럼 해시가 USD 에 구워지는 파일)나 빌더를 고치면
다시 빌드하고 그 해시를 함께 고칩니다. 자리(`scene.site`)나 카메라 정본을 고치면 `check_site_cameras.py` 를 다시 돌리고
Stage 를 다시 빌드합니다(Stage 의 해시와 빌더가 찍는 `plan_sha256` · `robot_rest` 를 함께 고칩니다 — 다시 짓지 않으면
`tests/generation/test_stage.py` 가 pxr 없이 빨강을 냅니다). 함께 고치는 곳은 `CONTRIBUTING.md` 「함께 고치는 곳」의 「봄」 · 「자리」 ·
「카메라 정본」 · 「재배포 자산」 행에 있습니다.

**짓지 않고 배포본의 자산을 놓아도 됩니다.** 배포본의 `assets/` 에 실린 USD · 텍스처는 위 순서가 짓고 받는 것과
같은 바이트입니다 — `meta/assets.json` 이 그 해시를 고정하고(`usd_sha256` · `bundle.files` · `ship.files`), 검증기 C6 와 게시
점검이 배포본에서 대조합니다. USD 는 텍스처를 옆의 `textures/` 에서 상대 경로로 읽으므로 빌더가 놓는 자리에 같은 이름으로
놓으면 받기와 빌드(Blender · MPFB · 100STYLE)를 건너뛰고 조건표의 행을 돌릴 수 있습니다. 놓는 것은 행이 띄우는 셋(장면 ·
든 상자 · 조끼 입은 관절 인형)과 그 텍스처뿐입니다 — 로봇(H1)은 행을 돌릴 때 시뮬레이션이 Isaac Lab 자산 루트에서 띄우고,
엮은 장면은 시나리오를 돌리지 않으며, 배포본에 없는 빌드 산출물(맨몸 · 피부만 입힌 관절 인형 · 걸음 표 파일 · 피부
내보내기)은 행을 돌리는 데 쓰이지 않습니다. 엮은 장면을 짓거나 확인하려면(`build_stage.py` · `check_stage.py`) 위 순서대로 받고 짓습니다.

```bash
D=<배포본 루트>
mkdir -p generation/scene/build/textures generation/dummy/build/textures
cp $D/assets/keti_logistics_aisle_v1.usd $D/assets/keti_carried_box_v1.usd generation/scene/build/
cp $D/assets/keti_dummy_ts15066_v2_jointed_skin_vest.usd generation/dummy/build/
cp $D/assets/textures/* generation/scene/build/textures/
cp $D/assets/textures/* generation/dummy/build/textures/
PYTHONPATH= .venv/bin/python -m pytest -q tests/generation/test_assets_canon.py    # 놓은 USD 의 해시를 정본과 대조
```

26.11 배포본에서는 검증기의 시드 재현 검사(C9 · 2026-11-17)가 이 저장소에서 빌드한 자산으로 에피소드 여섯 —
시나리오마다 무작위 행 26, `S1f` 의 무작위 행 15, `S1e` 의 격자 행 75 · 79 — 을 다시 생성했고, 여섯 모두 배포본과 대조 열 34개 ·
에피소드 메타 9 필드 · `notes` 멤버 17 · 정책이 비트 단위로 같았습니다(RGB 영상은 대조하지 않습니다). 배포본의 자산을 바로 위처럼
놓고 다시 생성한 것은 이 검사에 들지 않습니다.

### ZMP 자이로 항 실측 (2026-08-22)

```bash
~/isaacsim/python.sh generation/envs/s1a/check_zmp_gyroscopic.py --steps 400
```

`zero_moment_point` 은 `I·α` 만이 아니라 `I·α + ω×(I·ω)` 전체를 씁니다.
**같은 스텝에서 두 식을 나란히 계산한 결과**(384 스텝):

| | 중앙값 | p95 | 최대 |
|---|---|---|---|
| ZMP 이동량 \|full − α만\| | **0.7 mm** | 3.6 mm | **8.0 mm** |
| 자이로 토크 합 \|Σ ω×(Iω)\| | 0.38 N·m | 1.01 N·m | 3.83 N·m |

차이는 밀리미터 단위이고 중앙값은 1 mm 미만입니다. 그래도 전체 식을 쓰는 이유는 크기가 아니라
정의입니다 — 각운동량 변화율은 두 항의 합이고, 한쪽만 쓴 값은 ZMP 가 아닙니다.

### 변환·배포에서 부딪힌 것 — LeRobot v3.0 의 제약 다섯

스키마를 LeRobot v3.0 에 실을 때 맞춰야 하는 제약입니다.

1. **프레임마다 `task` 문자열이 필수** — 스키마 §6 필드가 아니므로 `field_map` 에 넣지
   않고 `meta/scenarios.json` 의 `robot_task` 에서 붙입니다
2. **`np.ndarray` 만 받습니다** — dtype 을 보존해 변환합니다. float32 로 일괄
   캐스팅하면 C9 가 깨집니다(훅이 float64 npz 를 배포 컬럼과 대조하므로).
   **예외 하나: `timestamp`** 는 LeRobot 이 `frame_index`/`fps` 로 직접 만들며
   float32 입니다. 그래서 C9 대조에서 제외합니다(`compare_repro.EXCLUDED_FROM_COMPARISON`)
3. **`string` dtype 이 스칼라 전용** — `region_names` · `contact.type` ·
   `safety.event` 는 프레임당 JSON 문자열 하나로 싣습니다. 검증기 `as_list()` 가 이미
   그 형태를 읽습니다
4. **`save_episode(episode_data=…)` 는 프레임 버퍼를 받습니다** — 스키마 §5 레코드는
   `finalize()` 뒤에 `meta/episodes/*.parquet` 에 조인합니다
5. **Hub 리포에 `codebase_version` 태그가 없으면 로드를 거부합니다** (2026-08-24
   실측) — 파일을 전부 올리고 검증기가 통과해도 `LeRobotDataset(repo_id)` 이
   `RuntimeError: Your dataset must be tagged with a codebase version` 로 죽습니다.
   태그 이름은 `meta/info.json` 의 `codebase_version`(`v3.0`)입니다. 손으로 기억하는 절차에
   맡기면 빠지므로 `publish_dataset.py` 가 업로드 · 태그 · 왕복 대조를 한 명령으로 묶습니다

검증기도 이 형태를 읽습니다 — `elem_len` 은 JSON 문자열의 원소 수를 세고, C7 은 단조 증가를 파일
전체가 아니라 에피소드별로 잽니다(LeRobot v3.0 은 여러 에피소드를 한 parquet 에 담습니다).

### 서브스텝 표집은 시뮬레이션을 건드립니다 (2026-08-25 실측)

접촉 센서를 서브스텝마다 계산시키면 **궤적 자체가 달라집니다.** 기록 방식만 바꾼
것이 아니라 시뮬레이션 결과가 바뀝니다. 추측이 아니라 대조군으로 확인했습니다.

| 실행 | 훅 | 직전 배치와 |
|---|---|---|
| 대조군 | 걸되 **아무것도 읽지 않음** | **비트 단위 동일** (3/3 에피소드) |
| 인체 센서만 | 서브스텝마다 `dummy_contact` 읽기 | 에피소드 0 동일, 1·2 갈라짐 |
| 운영 | 두 센서 모두 | 인체 센서만인 경우와 같은 지점에서 같은 크기로 갈라짐 |

훅을 거는 것 자체는 아무 영향이 없고 **읽는 것이 영향을 줍니다.** 에피소드 0001 은
스텝 353 에서 2.3e-3 rad 로 갈라져 몇십 스텝 만에 0.3 rad 가 됩니다. 그 에피소드는
직전 배치에서 로봇이 더미에 닿아 `fail`/S3 였는데, 새 배치에서는 닿지 않아
`near_miss`/S1 입니다. 둘 다 유효한 전개이고 어느 쪽이 옳다고 할 수 없습니다.

**정확한 기전은 확인하지 못했습니다.** 인체 센서를 읽는 코드가 물리에 쓰는 것은
없습니다. 서브스텝 도중 센서 버퍼를 계산시키면 PhysX 상태의 동기 시점이 달라지고,
더미를 세우는 외력 렌치가 큐에 들어가는 타이밍이 그에 걸리는 것으로 보입니다.
**가설이며 확인하지 않았습니다.**

**재현성은 그대로입니다 — 이것이 받아들일 수 있는 이유입니다.** 두 성질을 다시
쟀습니다(NaN 을 같은 값으로 보는 비교, `compare_repro.compare_arrays` 와 같은 방식):

- 실행 재현: 같은 명령 두 번이 3 에피소드 전부 비트 단위 동일
- 접두 재현: `--episodes 3` 이 `--episodes 10` 의 앞 3개와 비트 단위 동일

즉 바뀐 것은 「이 코드가 만드는 궤적」이지 「같은 코드가 매번 다른 궤적을 만든다」가
아닙니다. C9 도 통과합니다. **직전 배치의 수치를 새 배치의 것으로 인용하지 마십시오.**

**함정 둘.** (1) `np.array_equal` 을 `equal_nan=True` 없이 쓰면
`safety.zmp` 가 비행 구간에서 NaN 이라 **자기 자신과도 다르다고 나옵니다.** 저장소의 `compare_arrays` 는 NaN 을 같은 자리면 같은 값으로 봅니다.
(2) 실험용으로 훅을 고쳐 둔 채 파이프라인을 돌리면 C9 가 「재현 실패」로 빨간불을
냅니다. 실험 뒤에는 원복부터 하십시오.

### 변환은 재현되지만 `meta/stats.json` 만 키 순서가 다릅니다 (2026-08-25 실측)

같은 에피소드 디렉터리를 두 번 변환해 산출물 17개를 SHA-256 으로 대조했습니다.
**데이터 parquet 을 포함해 15개가 바이트 단위로 같고**, 다른 둘은 이렇습니다.

| 파일 | 차이 | 뜻 |
|---|---|---|
| `meta/stats.json` | **키 순서만** 다름 (키 집합·값 동일) | LeRobot 이 이 파일을 쓰는 순서가 실행마다 달라집니다. 우리 코드가 아니고, 의미는 같습니다 |
| `meta/rules.json` | 정본이 그 사이 바뀜 | 변환기는 정본을 그대로 동봉하므로 정본을 고치면 재변환이 필요합니다 |

두 번째가 실용적인 함의입니다 — **정본을 고쳤으면 배포 전에 다시 변환하십시오.**
데이터를 다시 만들 필요는 없습니다(같은 npz 에서 같은 parquet 이 나옵니다).

첫 번째는 **알고만 있으면 됩니다.** 두 변환본을 `diff` 로 대조하면 `stats.json`
하나가 뜨는데 결함이 아닙니다. 값을 비교하려면 파싱해서 보십시오 — 그리고
`safety.zmp` 의 통계는 NaN 이라 `!=` 로 비교하면 **자기 자신과도 다르다**고 나옵니다.

### 재현은 실행 단위로만 됩니다 (2026-08-20 실측)

| 무엇을 재현하나 | 방법 | 결과 |
|---|---|---|
| 실행 전체 | 같은 `--seq`·`--episodes` 로 다시 실행 | **비트 단위 동일** (3 에피소드 × 2회, npz·json 전부) |
| 실행의 앞부분 | `--seq {run_start_seq} --episodes {episode_seq − run_start_seq + 1}` | **비트 단위 동일** — C9 훅이 쓰는 방법. `run_start_seq` 는 사이드카와 에피소드 메타에 있습니다 |
| 에피소드 하나만 | `--seq {episode_seq} --episodes 1` | **아직 재현되지 않습니다** |

```bash
# 실행 전체 재현 — 성립합니다
~/isaacsim/python.sh generation/envs/run_s1a.py --seq 0 --episodes 3
```

**에피소드 단독 재현이 안 되는 이유는 시드가 아닙니다.** `--seq 2 --episodes 1` 과
`--seq 0 --episodes 3` 의 세 번째 에피소드는 **기록된 랜덤화 값이 전부 일치**합니다 —
`robot_base_mass_kg_realized` 가 양쪽 모두 정확히 15.166628 kg 입니다. 그런데 궤적은
0 스텝부터 갈라집니다.

**원인은 아직 모릅니다.** 확인된 사실은 딱 하나 — **같은 입력에서 다른 궤적이
나온다**는 것입니다. "에피소드를 넘어 남는 상태가 있다" 는 그 자체가 가설이며,
GPU 비결정성이 원인이라면 상태 이월은
성립하지 않습니다 — 두 설명은 양립하지 않으므로 하나로 좁혀야 합니다.

가장 싼 구별 실험: **같은 에피소드를 같은 방식으로 두 번** 돌려 봅니다. 배치 재실행이
비트 단위로 같았으므로(위 표) 프로세스 안에서는 결정론적이고, 따라서 GPU 비결정성은
이미 상당히 약한 설명입니다. 다음은 에피소드마다 프로세스를 새로 띄워 보는 것입니다.

**그래서 "파일명만으로 재현"은 절반만 성립합니다.** 정본(`meta/scenarios.json` `seed_rule`)과
카탈로그 §6 은 재현을 「실행 단위로만 성립」으로 적습니다.

실행기가 이 성질을 지키려고 하는 것 둘입니다.

- `env.reset()` 을 `torch.inference_mode()` 안에서 부릅니다 — 밖이면 두 번째 에피소드에서
  「Inplace update to inference tensor」로 죽습니다
- 몸통 질량 랜덤화(`add_base_mass`)는 `mode="reset"` 이라 **에피소드마다** 새로 뽑힙니다. 상속받은
  `mode="startup"` 은 구성 시점에 한 번만 뽑으므로, 구성 시드를 `--seq` 에서 파생하면 배치의 시작 위치에
  따라 다른 로봇이 만들어지고 한 실행 안에서는 질량이 변하지 않습니다. 구성 시점에 남은 startup
  항(`physics_material`, 범위가 축퇴돼 상수)은 `scenario_startup_seed()` 의 시나리오 상수를 씁니다

### 시나리오가 무엇을 랜덤화하는지는 환경 설정에서 읽습니다

환경은 Isaac Lab 의 보행 설정을 상속합니다 — 바닥 마찰은 고정이고(폭이 0 인 범위) 몸통 링크 하나의 질량에
0.8 – 1.25 배를 곱합니다. 정본의 선언이 환경이 실제로 하는 것과 어긋나지 않도록, 환경이 있는 시나리오의 네
블록(`randomization` · `randomization_distribution` · `randomization_fixed` · `policy_observation_noise`)을
**손으로 적지 않고** 풀린 환경 설정에서 읽습니다.

```bash
~/isaacsim/python.sh generation/envs/s1a/check_randomization.py           # 정본과 대조 (다르면 exit 1)
~/isaacsim/python.sh generation/envs/s1a/check_randomization.py --write   # 블록을 정본에 씀
```

환경을 만들지 않고 설정만 읽으므로 약 20초입니다(대부분 부팅). 환경 설정 · 커맨드 설정을 고쳤거나 Isaac Lab
체크아웃을 바꿨으면 돌립니다 — 읽는 값 대부분이 이 저장소에 없는 Isaac Lab 의 설정에서 옵니다.

조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 행 0 으로 지은 설정을 읽습니다. 환경 설정에서 에피소드마다 바뀌는 것은 행이고 행은 환경을 만들기 전에
`generation/conditions.py` 가 정하므로(무작위 행은 뽑고 격자 행은 칸에서 읽음), 행이 정하는 것 — 로봇의 출발 자리(`reset_base` 가 돌리는 `reset_robot_to_site`) · 사람의
대본(`reset_dummy_entry` 가 돌리는 `reset_dummy_scripted`) · 든 상자(`author_carried_box`) · 과제의 걷는 속도와 센서 지연(행을 든 명령 항) —
는 뽑기로 읽지 않습니다(`generation/randomization.py` `FROM_THE_ROW` — 항의 이름이 아니라 돌리는 함수로 가름). 행의 열은 블록에 적지 않고,
이 셋의 `randomization` 은 비어 있습니다 — 블록을 조건표 바로 뒤에 넷 모두 둡니다(빈 것은 「환경 설정이 에피소드마다 뽑는 축이 없음」 — 에피소드마다 뽑는 겉모습은 이 대조 밖의 `appearance` 블록).

| | |
|---|---|
| 읽는 것 | 이벤트 항(리셋 · 재질 · 질량 · 진입) · 커맨드의 보행 속도와 센서 지연 · 지면 재질 · 정책 관측의 잡음. 폭이 0 인 범위는 축이 아니라 고정값으로 읽습니다. 조건표의 행이 정하는 항과 행을 든 명령 항은 읽지 않습니다(위 문단) |
| 양방향 | 뽑는 축 · 분포 · 정책 관측 잡음은 정본과 환경이 **서로** 같아야 합니다 — 한쪽에만 있는 축이 차이입니다 |
| 한 방향 | `randomization_fixed` 는 정본이 적은 것만 환경과 대조합니다. 환경이 고정한 값은 훨씬 많고(시작 방향 0 · 관절 시작 배율 1.0 …) 전부 적지 않습니다 |
| 모르는 항 | 읽을 줄 모르는 이벤트 · 커맨드 항이 켜져 있으면 차이로 보고합니다. 환경에 뽑기를 더하면 `generation/randomization.py` 에 읽는 줄도 더합니다 |
| `mode="startup"` | 뽑는 축이 startup 이면 거부합니다 — 실행마다 한 번만 뽑혀 한 실행의 모든 에피소드가 같은 값을 씁니다 |
| 못 보는 것 | 범위를 **어떻게** 뽑는지(센서 지연은 0 – 100 ms 에서 뽑아 20 ms 단위로 내림하므로 실현값은 다섯), 이벤트 · 커맨드 · 정책 관측 밖의 난수, 항이 실제로 그 바디에 닿는지(사이드카의 `*_realized` 가 봅니다), 로봇과 지면 밖의 재질, 조건표의 행이 정하는 항 안에 누가 더한 뽑기 |
| CI | Isaac 이 없어 실제 설정을 못 봅니다. `tests/generation/test_randomization.py` 가 2026-10-03 에 풀린 설정의 모양과 그 설정을 조건표가 있는 시나리오가 바꾸는 모양(행 0)을 대역으로 들고 정본을 거기에 묶습니다 — 정본을 손으로 고치면 빨강이고, 정본은 그대로인데 설정이 바뀐 것은 위 스크립트만 봅니다. 행이 정하는 항의 함수 이름은 그 항을 놓는 줄과 대조합니다 |

**궤적에 구워지는 것은 정본이 아니라 환경 설정입니다.** 설정의 범위를 고치면 재생성이고(SOP §10.1), 정본을
설정이 이미 하는 대로 고쳐 적는 것은 재생성이 아닙니다. 블록 옆의 메모(`randomization_note` ·
`randomization_wiring_note`)는 산문이라 도구가 쓰지 않습니다 — `--write` 가 무언가를 바꿨으면 다시 읽습니다.

### 배치는 청크로 만듭니다

400건을 한 실행으로 돌리면 끊겼을 때 0번부터 다시 돌리고, 무손실 중간 영상(에피소드당 약 1.2 GB)이
디스크를 채웁니다. 그래서 **에피소드 10건을 청크**로 삼아, 청크마다 변환 · 검증한 뒤 무손실을
지우고 끝에 하나로 조립합니다. 조건표가 없는 시나리오(`S1d`)의 청크는 10건짜리 실행 하나이고, 재현이 실행
단위라 다시 만들 수 있는 최소 단위가 청크입니다.

조건표 시나리오(`S1e` · `S1f` · `S1g`)는 행 하나가 생성기의 실행 하나입니다(행은 시뮬레이터를 띄우기 전에
정해집니다). 그래서 청크는 **행 10개**이고, 실행기가 청크의 행마다 프로세스를 하나씩 띄웁니다
(`run_s1a.py --seq <행> --episodes 1`) — 청크 하나가 프로세스 10개이고, 다시 만들 수 있는 최소 단위는 행입니다. 시나리오 여럿을 한 배치로 돌리면 청크가 시나리오를 번갈아 돕니다 — 시나리오마다
첫 청크, 그다음 둘째 청크 … 의 차례라 도중에 멈춘 배치에도 시나리오마다 비슷한 수의 행이 들어 있습니다(이 차례는
자체 설정입니다).

```bash
# 깨끗한 작업 트리에서. 끊기면 같은 명령으로 이어 돌립니다 — 봉인된 청크는 건너뜁니다
.venv/bin/python generation/run_batch.py --batch .tmp/batches/S1d-400

# 조건표 시나리오 셋을 한 배치로 — --rows 는 JSON {시나리오: [행, …]}(키가 --scenario 와 같아야 함 ·
# 없으면 조건표의 행 전부(무작위 행과 격자 칸)). --episodes 는 S1d 의 것이라 여기에 쓰지 않습니다
.venv/bin/python generation/run_batch.py --batch .tmp/batches/<이름> --scenario S1e S1f S1g --rows <rows.json> --chunk 10

# 멈추려면 (지금 청크를 끝내고 종료)
touch .tmp/batches/S1d-400/STOP

# 조립 — 카드 · NOTICE · 정본을 고친 뒤에는 이것만 다시 돌립니다 (영상을 다시 인코딩하지 않음)
HF_HUB_OFFLINE=1 .venv-lerobot/bin/python generation/assemble_batch.py --batch .tmp/batches/S1d-400

# 같은 생성 커밋의 배치 여럿을 한 배포본으로 — 놓을 자리(--out)를 주고, 싣지 않기로 한 행은 --drop 으로 이름을 댑니다
HF_HUB_OFFLINE=1 .venv-lerobot/bin/python generation/assemble_batch.py --batch .tmp/batches/<배치 A> \
    --batch .tmp/batches/<배치 B> --out .tmp/batches/<배포본> --drop <시나리오>:<행>

# 검증기 전체 (조용한 기계에서)
.venv/bin/python validator/validate_dataset.py .tmp/batches/S1d-400/dataset \
    --repro-hook generation/rerun_seed.sh --repro-sample 8 --verbose
```

| | |
|---|---|
| 청크 하나 | 생성 → 라벨링 → 변환 → 검증(ERROR 0) → 측정 → 봉인(`SEALED.json`) → 무손실 삭제. 조건표 시나리오의 청크는 생성이 행마다 실행 하나(로그 `logs/generate-<행 4자리>.log`)이고, 측정이 점수표(`.venv/bin/python generation/scorecard.py --dataset <청크>/dataset --out logs/measure.json` — 배포본을 pyarrow 로만 읽어 lerobot 이 필요 없습니다)입니다 |
| 청크 이름 | `chunks/<첫 행 4자리>`(S1d — 예: `0040`) · `chunks/<시나리오>-<3자리>`(조건표 시나리오 — 그 시나리오의 몇 번째 청크인지, 예: `S1f-002`). 조건표 시나리오와 S1d 는 한 배치에 섞지 않습니다 |
| 재개 | 봉인된 청크는 건너뜁니다. 봉인 안 된 S1d 청크는 **지우고 처음부터** — 지우지 못하면 그 경로를 말하고 멈춥니다. 생성이 멈춰 다시 하는 시도의 `generate.log` 는 지우기 전에 `<배치>/stalled/` 에 옮겨 둡니다(아래 조건표 행의 로그도 같습니다 — 배치를 다시 돌리면 시도 번호가 1 부터 다시 세어지므로, 앞 실행이 남긴 같은 이름은 덮어쓰지 않고 `….attempt<N>.<k>.generate.log` 의 다음 번호 k = 2, 3, … 로 남깁니다). 봉인 안 된 조건표 청크는 **행 단위로 이어 갑니다**: 끝난 행은 그대로 두고, 끝나지 않은 행은 그 행의 파일(npz · 사이드카 · 라벨 · 무손실 영상 — 파일 이름이 그 에피소드 ID 로 시작하는 것)을 지우고 다시 만듭니다. 끝난 행은 npz 가 있고(0 바이트가 아님), 사이드카가 JSON 객체로 읽히며 `episode_id` · `run_start_seq`(= 그 행) · 생성 커밋 · `generator_dirty` false 가 맞고, 사이드카의 `media` 가 적은 무손실 영상(`meta/cameras.json` 의 카메라마다 하나 — 지금 일곱)이 모두 있고 0 바이트가 아닌 행입니다 — 생성기는 npz → 사이드카 → 영상의 차례로 쓰므로 그 사이에서 끊긴 행은 끝난 행이 아닙니다. 어느 행의 것도 아닌 파일도 지웁니다. 생성 뒤 단계의 산출물(`dataset/` · 라벨 · `generate-*.log` 밖의 로그)은 청크를 다시 돌릴 때마다 지우고 모든 행에 대해 처음부터 만듭니다. 행마다 시도는 셋이고 멈추면 그 행만 다시 합니다(앞 행의 파일은 그대로) — 멈춘 시도의 로그는 `<배치>/stalled/<청크>.<행 4자리>.attempt<N>.generate.log` 에 남기고, 세 번 모두 멈추면 「행 <행>」을 말하고 실패합니다. 0 으로 끝났는데 끝난 행이 아니어도(예: 영상 하나가 없음) 실패합니다. 끝난 행으로 판정한 직후에 디스크에 내려 씁니다(`sync`) — 그 뒤의 갑작스러운 정지(전원 차단 등)가 0 바이트는 아니지만 잘린 파일을 끝난 행으로 남기는 틈을 좁힙니다 |
| 첫 쓰기 기한 | 조건표 행의 실행이 띄운 지 **120 초**(`FIRST_WRITE_S` — 자체 설정) 안에 청크의 `episodes/` 아래에 파일을 하나도 쓰지 않으면 멈춘 것으로 보고 끝냅니다. 정상이면 카메라 파일이 16–18 초에 생기고, 출력 한 줄 없이 부팅에서 멈춘 일이 400건 배치의 생성 시도 43번 가운데 2번 있었습니다 — 26.11 배포본의 행 297개를 행마다 띄우면 같은 비율로 약 14번이고, 유예(480 초)까지 기다리면 한 번에 약 8.5 분이 듭니다. 앞 행이 남긴 파일은 세지 않습니다 — 띄우기 직전에 `episodes/` 아래의 파일을 경로 · inode · 크기 · 수정 시각 · 상태 변경 시각으로 적어 두고, 새로 생겼거나 그 가운데 하나라도 바뀐 파일(같은 이름 · 크기 · 수정 시각으로 갈아 끼운 파일 포함)만 이 실행이 쓴 것으로 셉니다. 파일의 시각을 시계와 견주지 않으므로, 시계를 뒤로 고친 뒤 앞 행의 파일이 미래 시각으로 남아 있어도 · 거친 시계가 새 파일의 시각을 시작보다 앞으로 적어도 바르게 셉니다. 기한은 단조 시계로 잽니다. 파일이 생긴 뒤의 멈춤 기준(유예 480 초 · 300 초 동안 쓰기 없음)은 그대로이고, S1d 의 청크 생성에는 이 기한이 없습니다 |
| 시작 조건 | `git status --short` 가 비어 있고 HEAD 가 배치를 시작한 커밋 — 배치 도중 이 체크아웃에 커밋하지 않습니다 |
| 트리를 다시 읽습니다 | 생성기는 시작할 때 트리를 한 번만 읽습니다. 그래서 실행기가 **청크를 시작하기 전과 봉인하기 직전에** 다시 읽어, HEAD 가 배치의 커밋이 아니거나 트리가 더러우면 그 청크를 봉인하지 않고 멈춥니다(보인 커밋이나 「더럽다」를 말합니다). **배치가 열려 있는 동안 다른 작업은 별도 워크트리에서 합니다** — 검증기나 측정 도구를 고치면 배치의 커밋이 달라져 이미 봉인된 청크 전부가 그 코드로 만든 것이 아니게 됩니다. 코드를 고쳐야 하면 이 배치는 그 커밋으로 끝낼 수 없고 새 배치로 시작합니다 |
| 봉인 기록 | `SEALED.json` 에 시나리오(`scenario`) · 청크의 행(`rows` — 만든 차례) · 시작 번호(`start_seq` = 첫 행) · 에피소드 ID · 생성 커밋 · 파일 해시 · 단계별 시간에 더해 **검증기 요약**(`validation` — 심각도별 건수)과 **측정 요약**(`measurement` — S1d 는 진입 속도 비 전체와 붙잡힌 것만의 최소 · 중앙 · 최대, 방향 편차 최대, 라벨 건수 · 조건표 청크는 점수표의 수 — 결과 셋의 건수(점수표처럼 degenerate 는 빼고) · `degenerate` 건수 · 로봇 넘어짐 · 랙 · 바닥 접촉 사건 수는 청크의 무작위 행의 것이고, 격자 행이 든 청크는 `grid` 도 싣습니다: 격자 행의 수와 그 가운데 결과 셋 각각의 건수 · `degenerate` 건수를 점수표의 실패 지도에서 읽은 것이며, 지도의 칸을 센 수가 격자 행의 수와 맞지 않거나 점수표가 격자 행의 수를 싣지 않았으면 요약은 `null` 입니다)이 들어갑니다. 조건표 청크에는 **`rows_s`** 도 들어갑니다 — 이번 실행에서 만든 행마다 그 행을 만든 시도의 생성 초(`{"<행>": 초}` — 앞 실행이 끝내 둔 행은 없음)이고, `stages_s.generate` 는 이번 실행이 행을 만드는 데 쓴 시간 전부입니다. 진행 줄 「청크 NNNN 봉인 — …」이 같은 숫자를 찍습니다. 단계가 0 으로 끝났는데 `logs/validate.json` 이나 `measure.json` 을 읽지 못하면 그 값은 `null` 이고 줄이 「요약 없음」이라 말합니다(청크는 실패시키지 않습니다). **속도 비에 문턱은 없습니다** — 수가 보일 뿐이고 어디까지 받을지는 사람이 정합니다. 이 두 요약은 `logs/` 에서 온 것이라 해시가 지키지 않습니다. 파일 해시는 `episodes/` 바로 아래의 파일과 `dataset/` 의 모든 파일의 것입니다 — 파일을 가리키는 심링크는 가리키는 바이트로 세고, `dataset` · `episodes` 자체가 링크이면 따라가 셉니다. **`dataset/` 안에서 디렉터리를 가리키는 심링크를 만나면 봉인하지 않습니다**: 훑기가 링크 아래로 내려가지 않아 그 아래의 파일은 해시 없이 남는데, 조립은 데이터셋을 경로로 열어 그 파일을 읽기 때문입니다. **목록을 읽지 못하는 디렉터리**도 같은 이유로 봉인하지 않습니다(지나치면 그 아래의 파일이 말없이 봉인에서 빠지는데, 경로를 아는 파일은 여전히 열립니다). 실행기는 「청크 NNNN 실패 — seal: …」로 어느 링크 · 어느 디렉터리인지 말하고 멈추며, 조립도 청크의 봉인을 다시 볼 때 같은 문장으로 멈춥니다(링크가 봉인 전에 있었든 뒤에 생겼든). 변환기는 그런 링크를 만들지 않습니다. `episodes/` 아래의 디렉터리는 훑지 않습니다 — 봉인할 때 무손실 영상이 거기 있고, 조립이 읽는 것은 바로 아래의 파일뿐입니다 |
| 실패한 청크 | 실행기가 멈추고 무손실을 남깁니다. 다음 실행이 S1d 청크는 지우고 조건표 청크는 끝나지 않은 행의 파일을 지우므로, 보려면 옮겨 둡니다 |
| 같은 자리에서 늘 실패하는 청크 | 시드가 정해져 있어 다시 돌려도 같은 곳에서 같은 이유로 실패합니다(예: 측정이 고정된 시드에서 「진입은 있는데 실현 속도가 없음」을 거부). 실행기는 **그 청크에서 멈춘 채로 있습니다** — 다음 청크로 넘어가지 않고, 다시 돌려도 같은 청크에서 다시 만들어 같은 곳에서 멈춥니다. 로그(`logs/`)와 남긴 무손실을 보고 **결정은 사람이 합니다**: 원인이 코드면 새 배치, 그 시드가 정말 받을 수 없는 에피소드면 수량을 줄인 새 배치 — 멈춘 배치의 봉인된 청크는 그대로 두고, 아직 봉인되지 않은 행에서 그 행을 뺀 새 배치를 **같은 커밋에서** 만든 뒤, 조립이 두 배치를 그 행의 `--drop` 과 함께 한 배포본으로 묶습니다(아래 「조립」). 실행기가 스스로 건너뛰는 경로는 없습니다 |
| 한 번에 하나 | 실행기는 시작하면서 `<배치>/LOCK` 에 `flock`(배타 · 기다리지 않음)을 걸고 끝날 때까지 쥡니다 — 두 실행기가 동시에 확인하고 동시에 쥐는 틈이 없고, 프로세스가 어떻게 죽든 커널이 풀므로 낡은 잠금을 따질 일이 없습니다. 파일 자체는 지우지 않습니다(남아 있어도 잠금이 아닙니다). PID 와 단계의 프로세스 그룹은 따로 `<배치>/RUNNER` 에 적습니다 — 임시 이름에 쓰고 바꿔 치우므로 읽는 쪽이 빈 파일을 보지 않습니다. 두 번째 실행기는 `RUNNER` 에서 쥔 쪽의 PID 를 읽어 말하고 거절합니다 |
| 단계가 남긴 것 | 단계는 자기 세션에서 돌아 실행기가 죽어도 살아남습니다. 다음 실행은 잠금을 쥔 직후, `RUNNER` 가 가리키는 그룹에 **아직 도는 프로세스가 하나라도 있고** 그 명령줄이 이 배치의 경로를 담고 있으면 SIGTERM → 유예 → SIGKILL 로 끝냅니다 — 리더가 먼저 죽은 그룹도 같습니다(생성기의 리더는 `exec` 하지 않는 bash 래퍼라 시뮬레이터보다 먼저 죽을 수 있습니다). 한 실행 안에서는 그룹을 쓸어 내는 때가 정해져 있습니다: 멈춘 단계를 끝낼 때, 실행기가 중단되거나 기록을 쓰지 못했을 때(리더가 아직 도는 경우), 그리고 **리더가 시그널로 끝났을 때**(손으로 죽이거나 메모리 부족으로 죽은 래퍼 — 자식을 기다리지 않았으므로 남은 시뮬레이터는 낙오자이고, 몇 개를 끝냈는지 진행 줄이 말합니다) 입니다. **리더가 종료 코드로 끝났으면(0이든 아니든) 그룹은 건드리지 않습니다** — 정상으로 끝난 래퍼는 자식을 기다렸으니 그룹에 남은 것은 시뮬레이터가 일부러 남긴 것이고, Kit 이 띄우는 Omniverse 허브(`hub --mode=shared`)가 그렇습니다: 같은 머신의 다른 세션이 함께 쓰는 데몬이라 정상 단계 뒤에도 계속 돕니다(검증기 C9 의 훅도 그렇게 남깁니다). 이름으로 가려내지 않으며, 알고 놓치는 것은 종료 코드로 끝난 리더의 그룹에 그 리더 자신의 자식이 아직 도는 경우입니다. 정상 단계 뒤에는 `RUNNER` 가 `None` 이라 다음 실행도 그 그룹을 끝내지 않습니다(실행기가 단계 도중에 죽으면 다음 실행이 허브가 든 그룹까지 끝냅니다 — 비정상 경로로 받아들입니다). SIGKILL 뒤에도 남으면 그 번호를 말하고 멈추며(종료 코드 2) `RUNNER` 는 그 그룹을 계속 가리킵니다 |
| `STOP` 파일 | 청크 사이에서만 봅니다 — 조건표 청크도 행 사이에서 멈추지 않고 그 청크를 끝냅니다. 실행기가 **지우지 않습니다.** 남아 있으면 다음 실행도 봉인 안 된 첫 청크 앞에서 멈추므로, 이어 돌리기 전에 파일을 지웁니다 |
| 종료 코드 | **0** — 모든 청크가 봉인되어 끝난 것과 `STOP` 으로 멈춘 것이 둘 다입니다. 0 이 배치의 완성을 뜻하지 않고, 둘을 가르는 것은 조립이 봉인 안 된 청크가 있는 배치를 거부하는 것입니다(배치 여럿을 함께 조립하거나 `--drop` 을 주면, 봉인 안 된 청크는 그 행이 모두 봉인된 다른 청크에 있거나 `--drop` 의 행일 때만 건너뛰고 아니면 거부합니다). **1** — 청크 하나가 실패해 멈춘 것(단계 · 트리가 바뀐 것 · 지우지 못한 것 · 봉인할 수 없는 것 포함). **2** — 시작하지 못한 것: 더러운 트리, 한 배치가 되지 않는 인자(S1d 와 조건표 시나리오를 섞음 · 같은 시나리오 둘 · 조건표 시나리오에 `--episodes` · S1d 에 `--rows` · `--rows` 의 키가 `--scenario` 와 다름 · 조건표 밖의 행), 이 배치가 아닌 인자(커밋 · 수량 · 행이 다름), 다른 실행기가 `LOCK` 을 쥐고 있는 것, 단계의 프로세스 그룹이 SIGKILL 뒤에도 남은 것(그룹 번호를 말하고 청크를 지우지 않습니다) |
| 진행 줄 | 줄마다 flush 하므로 출력을 파일로 돌려도 바로 쌓입니다 — 따로 옵션을 줄 것이 없습니다 |
| 조립 | 봉인된 청크만, 봉인 기록이 배치의 계획과 같은 청크만 받습니다 — 시나리오 · 행(`rows` — 만든 차례까지) · 시작 번호 · 에피소드 ID · 생성 커밋(`rows` 가 없는 봉인 기록은 `start_seq` 부터 `episodes` 건의 범위로 읽습니다 — 실행 하나인 청크). **같은 생성 커밋 · 같은 리포의 배치 여럿을 한 배포본으로 조립합니다**(`--batch` 를 여러 번 · `assemble_batch.release_chunks`): 실릴 행은 배치들이 계획한 (시나리오, 행) 전부에서 `--drop <시나리오>:<행>` 으로 이름을 댄 행을 뺀 것이고, 그 행마다 **배치들을 통틀어 봉인된 청크 정확히 하나**에 있어야 합니다 — 봉인된 청크 둘에 있어도, 어느 봉인된 청크에도 없어도 멈춥니다(뒤의 것은 그 행을 계획한 청크와 「봉인되지 않음」 · 「디렉터리가 없음」을 청크마다 한 줄로 말합니다). 봉인되지 않았거나 디렉터리가 없는 청크는 그 행이 모두 봉인된 다른 청크에 있거나 `--drop` 의 행일 때만 건너뜁니다. `--drop` 의 행은 어느 배치가 계획한 행이어야 하고(아니면 오타로 보고 멈춤) 봉인된 청크에 있으면 안 됩니다 — 봉인된 청크는 통째로 실리고, 청크에서 에피소드 하나를 빼는 길은 없습니다. 에피소드는 배치를 준 차례대로, 배치 안에서는 그 배치의 청크 차례대로 실립니다. 배치가 여럿이면 청크를 `<배치 디렉터리 이름>/<청크>` 로 말합니다(두 배치의 `S1e-000` 을 가르려고 — 그래서 이름이 같은 배치 둘 · 같은 배치 두 번은 받지 않습니다). 배치가 여럿이거나 `--drop` 을 주면 건너뛴 청크와 뺀 행을 조립의 첫 줄들에 적습니다. 배포본은 `--out` 에 놓습니다 — 배치 여럿이면 꼭 주어야 합니다. `--out` 은 링크와 `..` 를 풀어 한 경로로 보고(아직 없는 디렉터리를 지나는 `..` 도) 확인도 바꿔 넣기도 그 경로에서 합니다. 그 경로는 어느 배치 디렉터리(`batch.json` 이 있는 디렉터리 — 주지 않은 배치도)의 안도 아니어야 하고(배치와 그 청크는 읽기만 하는 입력이고, 청크의 `dataset/` 에도 `meta/info.json` 이 있습니다 — 배치 하나를 줄 때 그 배치의 실제 디렉터리 바로 아래 `dataset` 만 예외이고, 링크인 `<배치>/dataset` 은 가리키는 자리로 판단합니다), 이미 있으면 앞선 조립의 결과(`meta/info.json` 이 있는 디렉터리)여야 합니다 — 조립은 그 자리의 것을 치우고 결과를 놓습니다. 작업 디렉터리 · 병합에 넘기는 뷰 · 바꿔 넣는 동안의 이전 결과는 그 옆에 `<이름>.assembling` · `.views` · `.previous` 로 둡니다. 배치 하나에 `--out` 을 주지 않으면 배포본은 `<배치>/dataset` 입니다. **청크마다 과제가 달라도 한 배포본으로 조립합니다**(시나리오 여럿의 배치는 시나리오마다 과제 문장 — `meta/scenarios.json` 의 `robot_task` — 이 하나): 과제 표는 청크 차례대로 처음 나온 문장의 순서이고, 다시 쓰는 스텝의 `task_index` 는 그 청크의 과제 표의 행 자리에서 이 표의 행으로 다시 매깁니다. LeRobot 0.6.1 의 병합도 이 순서로 표를 짓고 자기 스텝과 에피소드의 `task_index` 통계를 그 표로 다시 매기며, 병합이 지은 표가 이 순서가 아니면 조립이 멈춥니다. LeRobot 0.6.1 의 병합이 깨뜨리는 셋(통계 · 데이터 parquet · 번호 통계)을 고치고 스스로 검증합니다 — 스텝 행(스키마 · null · 값) · 에피소드 메타(우리 열과 에피소드별 통계 열이 청크의 것과 같고, 번호 필드의 통계만 앞 청크만큼 옮겨지며 `task_index` 의 통계는 다시 매긴 번호의 것 · `run_start_seq` 는 청크의 계획이 주는 값 — 실행 하나인 청크는 첫 행, 조건표 청크는 행 자신 · 에피소드의 차례는 청크 차례대로, 청크 안에서는 변환기가 쓰는 차례(녹화 파일 이름의 차례 — 봉인 기록의 `rows` · 에피소드 ID 는 행을 요청한 차례 그대로라, 행을 오름차순이 아니게 요청한 청크도 조립됩니다) · 에피소드마다 스텝이 과제 표에서 가리키는 문장이 그 에피소드의 `tasks` · `meta/info.json` 의 `total_tasks` 가 과제 표의 행 수) · `meta/stats.json`(스텝 표에서 직접 잰 min · max · mean · std 와, 에피소드별 통계를 조립이 다시 합친 값 — 분위수는 후자만) · **카메라 · 에피소드마다** 병합본이 가리키는 영상 구간의 패킷이 그 에피소드의 청크 구간의 것과 같음 · 로더가 스텝과 그림을 읽음. 통계 허용오차는 리허설 배치에서 잰 값이고(`assemble_batch.py` 상단), 스텝 표 대비 std 가 느슨한 것은 LeRobot 이 에피소드별 std 를 float32 로 계산해, 값이 늘 같은 열(고정 카메라의 자세)에서 참값 0 대신 0.026 같은 값이 청크 자체에 들어 있기 때문입니다 |
| 조립의 정본 대조 | 병합 전에 청크마다 **청크가 싣고 온** `meta/rules.json` · `meta/scenarios.json` 을 **지금 작업 트리의 정본을 그 청크로 좁힌 것**과 대조하고(파싱한 문서로), 다르면 어느 규칙 · 시나리오의 어느 항목인지 말하고 멈춥니다 — 차이를 하나도 빼지 않고 문장이 아닌 것부터 적고, 머리에 「문장이 아닌 것 N건 · 문장 M건」을 적습니다. 라벨은 청크를 만들 때의 정본으로 매겨졌으므로 **값이 바뀐 것이면 재생성**입니다. `--accept-canon-change` 는 **설명 문장만 바뀐 것이 확실할 때** 씁니다 — 목록을 찍고 지금 정본을 싣습니다. 임계값이나 규칙의 구조가 목록에 있으면 쓰지 않습니다. **청크의 에피소드가 말하는 로봇의 관절 매핑**(`meta/joint_map/<로봇>.yaml`)도 같은 자리에서 대조합니다 — 통째로 싣는 파일이라 **바이트로** 대조하고, 다르면 목록에 `(파일 전체) (바이트)` 한 줄로 적어 「문장이 아닌 것」에 세며, **이 차이는 플래그로도 받아들이지 않습니다**(에피소드의 관절 이름은 청크를 만들 때의 그 파일에서 읽었고, 조립 끝의 라벨 검증은 관절 매핑을 읽지 않습니다 — 파일을 청크가 실은 것으로 되돌리거나 재생성입니다). **청크의 에피소드가 참조하는 재배포 자산 파일**(`assets/` 의 USD · 텍스처 · 폴더째 싣는 로봇의 파일)도 같은 자리에서 대조합니다 — 청크가 실은 사본의 해시(봉인 기록)를 **지금 정본이 그 청크에 싣겠다는 파일의 해시**(`meta/assets.json` 의 `usd_sha256` · `bundle.files` · `ship.files`)와. 조립은 그 파일을 작업 트리의 빌드에서 싣고 변환기 · 검증기 C6 · 게시기는 실린 파일을 지금의 정본이 고정한 해시와(게시기는 작업 트리의 빌드와도) 대조할 뿐 청크의 사본과는 대조하지 않으므로, **빌드와 그 해시를 함께 고친 것**은 이 대조만 봅니다. 해시가 다른 파일은 `(파일 전체) (바이트)`, 한쪽에만 있는 파일은 `(파일 전체) (청크에만 있음)` · `(파일 전체) (지금 정본에만 있음)` 으로 한 줄씩 적어 「문장이 아닌 것」에 세고, **이 차이도 플래그로 받아들이지 않습니다**(빌드와 정본의 해시를 청크가 실은 것으로 되돌리거나 재생성입니다). 차이가 없으면 몇 개를 대조했는지 말합니다(「자산 파일 N건」 — 0건이면 대조한 것이 없다는 뜻입니다). **에피소드가 참조하지만 싣지 않는 자산**(`redistributed` 가 참이 아닌 것 — 싣지 않는 로봇의 USD 등)은 실을 파일이 없으므로 같은 자리에서 그 **고정 해시**(`meta/assets.json` 의 `usd_sha256`)를 **청크가 실은 `meta/assets.json` 의 것**과 대조합니다 — 배포본이 그 자산에 대해 말하는 것은 그 해시뿐이고, 에피소드는 청크를 만들 때의 정본이 고정한 자산으로 생성되었습니다. 자산은 에피소드의 `scene_assets` 가 적은 경로로 찾아 청크의 정본에 적힌 항목 이름으로 적고(청크의 정본에 그 항목이 없으면 지금 정본의 이름으로), 해시가 다르면 `meta/assets.json: robots.<로봇>.usd_sha256 (문자열 값)`, 한쪽에만 있으면 `(청크에만 있음)` · `(지금 정본에만 있음)` 으로 적어 「문장이 아닌 것」에 세며 **플래그로 받아들이지 않습니다**(정본의 해시를 청크가 실은 것으로 되돌리거나 재생성입니다). 청크와 지금 **양쪽에서 싣는** 자산만 위의 파일 대조에 맡기고, 싣는지가 그사이 바뀐 자산은 해시도 대조합니다 — 파일은 실리는 이름(`assets/<파일 이름>`)으로 대조하므로 이름이 같은 다른 자산이 그 자리를 채우면 파일 대조로는 같아 보입니다. 같으면 몇 개를 대조했는지 말합니다(「고정 해시 N건」 — 양쪽 모두 싣지 않는 자산만 세고, 해시가 없는 자산(절차적 지면)은 세지 않습니다). 이 대조는 정본과 정본을 대조할 뿐 생성기가 띄운 USD 의 바이트는 보지 않습니다. 작업 트리는 이 대조 때와 병합 뒤 싣기 때 두 번 읽으므로 **조립이 도는 동안 작업 트리의 정본을 고치지 마십시오** — 그 사이에 고친 것은 대조되지 않고 실립니다. 통째로 싣는 나머지 정본(`meta/safety_schema.json` · 그 밖의 `meta/assets.json` — 라이선스 · 설명 · `usd_checked` 등 · `meta/cameras.json` 의 녹화 지문 밖)과 이 배치에 에피소드가 없는 로봇의 관절 매핑 · 에피소드가 참조하지 않는 자산은 이 대조가 보지 않습니다. 봉인 기록에 없는 파일은 이 대조가 보지 못합니다 — 그래서 봉인은 훑기가 내려가지 못하는 자리(**데이터셋 안의 디렉터리 심링크** · 목록을 읽지 못하는 디렉터리)를 거부하고(「봉인 기록」 행), 조립은 그런 청크에서 이 대조에 이르기 전에 멈춥니다. 청크마다 자기 에피소드가 참조하는 자산을 묻고, 싣기는 모든 청크의 자산을 한 번에 묻습니다 — **청크들의 자산을 한 배포본에 함께 실을 수 있는지**(같은 파일 이름의 USD 둘 · 한 텍스처 이름에 해시 둘)도 여기서, 병합 전에 봅니다. 작업 트리의 빌드가 정본이 고정한 바이트인지는 이 대조가 아니라 변환기가 자산을 실을 때(병합 뒤) 봅니다 |
| 조립의 라벨 검증 | 자기 검증의 마지막에 **조립한 파일만으로 에피소드마다 라벨을 다시 유도해** 실린 라벨과 대조합니다(`verify_labels` — 아래 「실린 라벨은 배포본만으로 다시 나옵니다」의 도구를 그대로 부릅니다). 하나라도 다르거나 라벨러가 받지 않으면 결과를 놓지 않습니다. **위 플래그 뒤에 서는 검사입니다** — 플래그는 값의 차이도 받아들이고, 그러면 새 임계값과 옛 라벨이 함께 실리는데 정본 대조는 넘어갔고 게시기는 실린 정본이 저장소와 같아 통과하며 검증기는 라벨끼리만 봅니다. 판정을 하나도 못 바꾸는 값의 변경은 여기서도 보이지 않습니다. 조립본을 바꿔 넣기 전에 돌므로 실패해도 앞의 결과는 그대로입니다 |

**배치 뒤에 `meta/rules.json` 의 값을 바꾸면 재생성입니다** — 라벨이 청크 배포본에 이미 들어 있고
무손실이 없습니다. 조립이 그 차이를 거부합니다(위). 설명 문장 · 카드 · NOTICE 는 조립만 다시 돌리면 됩니다.

**조립은 카드 옆에 문서 넷을 더 싣습니다** — 카드의 한국어 화면(`README.ko.md`) · 상세 문서 두 벌(`docs/dataset-reference.md` ·
`docs/dataset-reference.ko.md`) · 첫 화면의 그림(`docs/media/teaser.png`). 목록은 `generation/release_docs.py` 에 있고 저장소의
지금 파일을 그대로 싣습니다. 변환기는 카드(`README.md`)만 실으므로 변환기로 바로 만든 배포본에는 이 넷이 없고, 게시기는 넷 가운데
하나라도 없거나 저장소와 다른 배포본을 올리지 않습니다. 이 문서들을 고친 뒤에도 조립만 다시 돌리면 됩니다.

### 게시기가 올리기 전후에 더 보는 것

**공개로 바뀌는 것은 head 가 아니라 리포입니다.** 어떤 ref 에서든 닿는 커밋이 전부 함께 나가므로, 공개될 리포에는
처음부터 공개할 것만 올립니다. 배포본에서 뺀 것이 한 번이라도 커밋됐으면 이력을 누르지 않고 리포를 새로 만듭니다.

**배포본 안에 디렉터리를 가리키는 심링크나 목록을 읽지 못하는 디렉터리가 있으면 올리지 않습니다.** 올릴 파일을 고르는
탐색과 왕복 대조의 목록은 그 아래로 내려가지 않는데, 동봉 파일의 대조는 경로를 알고 열어 읽습니다 — 그대로 두면 「있다」고 본
파일이 빠진 채 올라갑니다.
파일을 가리키는 링크(받은 배포본의 캐시가 그렇게 놓입니다)는 두 목록이 모두 파일로 세므로 막지 않습니다.

| 플래그 | 하는 일 |
|---|---|
| `--preflight-only` | 올리기 전 점검만 하고 Hub 를 건드리지 않고 끝냅니다. 기록으로 남길 전체 검증(`--repro-sample 8`)을 이것으로 돌립니다 |
| `--log-dir` | 검증기의 시도별 출력(`validate-attempt-N.log`)과 보고서(`validate-attempt-N.json`, 같은 번호)를 둘 자리. 기본은 배포본 디렉터리 옆의 `publish/<시각>/`. 같은 자리를 다시 써도 앞 실행의 것을 덮지 않고 번호를 이어 갑니다. 상대 경로는 호출한 자리 기준입니다 |
| `--release-tag v26.11.0` | 왕복 확인 뒤에 릴리스 태그를 한 번 겁니다. 이미 다른 커밋에 있으면 **옮기지 않고**, `v3.0` 을 옮기기 전에 멈춥니다. 공개 직전의 마지막 게시에서만 씁니다 |
| `--audit-history` | `--repo-id` 의 모든 ref 를 걸어 「이력에만 있고 지금 head 에 없는 경로」와 「판이 둘 이상인 경로」를 찍습니다. 읽기 전용이고, 앞의 것이 하나라도 있으면 exit 1. 걷는 것은 **어느 ref 에서든 닿는 커밋**뿐입니다(어느 ref 에서도 닿지 않는 커밋은 보지 못함). Hub 이 스스로 넣고 첫 업로드가 고치는 `.gitattributes` 는 판이 둘이어도 따로 한 줄로 보이고 읽을 것을 청하지 않습니다 |

게시기가 스스로 하는 것 셋입니다.

- **검증기가 죽은 것과 판정을 가릅니다.** 크래시 시그널(SIGSEGV · SIGABRT · SIGBUS · SIGFPE · SIGILL)로 죽었거나 3(검증기
  자신의 예외)으로 끝났으면 다시 돌립니다 — 모두 합쳐 세 번까지. 판정(0 · 1 · 2)은 다시 돌리지 않습니다. 죽은 뒤 통과했으면
  점검 통과 줄 아래에 몇 번 죽었는지와 로그 경로를 찍습니다. **스택을 읽는 것은 사람의 일로 남습니다.** 누가 세운 것
  (SIGTERM · SIGINT · SIGHUP · SIGKILL — 음수 코드로, C9 훅을 기다리던 중이면 128+N 으로)은 다시 돌리지 않고 「멈췄습니다 —
  판정이 아닙니다」로 거부합니다(SIGKILL 은 메모리 부족일 수도 있다고 덧붙입니다). 그 밖의 코드(127 등)도 판정이 아니라고
  말하고 다시 돌리지 않습니다
- **Hub 가 답한 리포 이름이 요청한 이름과 다르면 올리지 않습니다.** 이름을 바꾼 리포의 옛 이름은 새 위치로
  리다이렉트됩니다. 게시기는 리포를 만들기 **전에** 그 이름을 묻고, 다른 리포로 답하면 아무것도 만들지 않고 멈춥니다 —
  그 이름에 만들기를 부르면 Hub 이 새 빈 리포를 만들어 리다이렉트가 사라지고(2026-10-06 · 2026-11-08 에 Hub 에서 잼),
  배포본은 그 새 리포로 가며 옛 이름을 가리키던 주소도 모두 그 리포에 닿습니다. 옛 이름에 새 리포가 정말 필요하면 먼저
  직접 만든 뒤 게시합니다. 만든 뒤에도 한 번 더 묻습니다 — 두 물음 사이에 이름이 바뀔 수 있고, 만들기가 거부되면
  클라이언트(`huggingface_hub` 1.27.0)가 401 · 402 · 403 을 「이미 있을 수 있음」으로 보고 `repo_info` 로 확인하는데,
  그것이 리다이렉트를 따라가 답하므로 거부를 삼킵니다. 그대로 두면 뒤따르는 호출이 리다이렉트를 따라가 그 리포에
  올라갑니다. 이 둘째 물음에서 멈추면 만들기를 부른 뒤이므로 그 이름에 빈 리포가 생겼을 수 있습니다(거부 문장이 그렇게
  말합니다). **보지 못하는 것** 둘 — 토큰이 읽을 수 없는 리포로 리다이렉트되는 이름에 Hub 이 「없음」으로 답하면 없는 이름과
  가르지 못하고(클라이언트의 코드로 본 것 · Hub 에서 재지 않음), 마지막 물음 뒤 올리는 도중에 이름이 바뀌는 것은 보지 않습니다
- **리포에 배포본에 없는 파일이 있으면 올리기 전에 멈춥니다.** 왕복 대조도 같은 것을 잡지만 전부 올리고 전부 내려받은
  뒤입니다. 지우는 것은 여전히 사람이 정합니다

업로드는 한 호출이 여러 커밋으로 나뉠 수 있고(`huggingface_hub` 1.27.0 · `hf_xet` 1.6.0) 끊기면 같은 명령을 다시
돌리면 이어집니다. 게시기는 돌려받은 **마지막** 커밋을 통째로 내려받아 대조한 뒤에만 태그합니다.

**실측(2026-10-06).** S1d 400건(180파일 · 22.2 GB)을 이 경로로 `KETI-IRRC/keti-humanoid-safety` 에 올렸습니다 —
점검(C9 표본 둘) 22분 · 업로드 5분(커밋 하나) · 내려받아 대조 약 4분.

**카드가 말하는 생성 커밋.** 카드에 `<!-- generator-commit: <40자 16진> -->` 표식이 있으면 점검이 그 해시를
배포본 에피소드의 `sim_backend.generator_commit` 과 대조합니다 — 에피소드가 모두 한 커밋이어야 하고, 같은 해시가
본문에 국문 · 영문으로 한 번씩 있어야 합니다. 표식이 없으면 넘어가되, **릴리스 태그를 거는 게시
(`--release-tag`)에서는 없는 것이 거부 사유**입니다. 「있을 때만 본다」로 두면 문장을 빠뜨린 날 이 검사가 꺼집니다.

### 실린 라벨은 배포본만으로 다시 나옵니다

검증기는 라벨끼리 대조하고 시계열과는 대조하지 않습니다(`validator/README.md` 「C4 가 하지 않는 일」). 카드는 받은
쪽에게 「`rules.json` 과 프레임으로 규칙을 직접 돌리라」고 하는데, 그 일을 전 에피소드에 하는 도구입니다.
**조립은 이것을 스스로 돌립니다**(`assemble_batch.verify_labels` — 자기 검증의 마지막, 통과해야 결과가 놓임).
변환기로 바로 만든 배포본이나 이미 놓인 조립본은 아래 명령으로 직접 돌립니다.

```bash
.venv/bin/python generation/labeling/check_release_labels.py --dataset <데이터셋_루트> --out <보고서.json>
```

라벨러(`label_episode`)를 그대로 쓰되 입력을 **배포본에 실린 파일에서만** 만듭니다 — 규칙이 읽는 스텝 열
(`observation.root_vel` · `observation.contact.force` · `observation.contact.type` · `observation.contact.region_names` ·
`safety.separation_violation` · `safety.event`), 에피소드 표의 `notes` · `termination`, `meta/rules.json` ·
`meta/scenarios.json`, `meta/info.json` 의 `fps`, `meta/safety_schema.json` 의 이벤트 이름. 녹화(npz · 사이드카)도
저장소의 정본도 읽지 않습니다. 대조하는 것은 라벨러가 내는 전부입니다 — `outcome` · `severity` · `triggered_rules` 와
`notes` 의 `not_evaluated` · `contact_observed` · `contact_detection_limits`. 조건표가 있는 시나리오의 에피소드는 근거 요약
(`evidence.*` 열 — 규격서 §5)도 대조합니다. 그 입력은 거리 두 열(`safety.min_distance_human` · `safety.separation_required`),
있으면 사람 자세 · 카메라 자세 · 보이는 몫 열, `notes` 의 `human_stature_m`, 배포본에 있으면 `meta/cameras.json` 입니다.

- **행은 에피소드 표가 가리키는 대로 읽습니다**(`dataset_from_index` · `dataset_to_index` · `data/file_index`). 그 행의
  `episode_index` 가 다른 에피소드이거나, 길이 · 파일이 어긋나거나, 어느 에피소드에도 속하지 않는 행이 있으면 대조하지
  않고 멈춥니다(종료 코드 2). 적재기를 다른 에피소드의 프레임으로 보내는 표가 「일치」로 읽히지 않게 하려는 것입니다
- **행과 에피소드 표는 저장된 순서 그대로 읽고, 정렬하지 않습니다.** LeRobot 은 스텝 행도 에피소드 표의 행도 **자리로**
  읽으므로(파일 이름 순 · 파일 안의 순서, `meta.episodes[k]`), `index` 나 `episode_index` 로 정렬해 읽으면 자리가 바뀐
  배포본이 멀쩡하게 보입니다. `index` · `episode_index` 가 자리와 다르면 멈춥니다. **읽는 파일도 적재기가 읽는 것만**입니다 —
  `data/` 와 `meta/episodes/` 의 한 단계 아래(`<chunk>/<file>.parquet`). 다른 깊이의 parquet 은 적재기가 읽지 않으므로
  여기서도 읽지 않고, 그 파일이 있어야만 온전한 표는 자리 검사에서 멈춥니다
- 항목은 **값과 형이 함께 같아야** 같은 것으로 봅니다 — `0` 은 `false` 가 아닙니다
- **적재기가 넘기는 수로도 한 번 더 유도합니다.** LeRobot 은 항목을 텐서로 넘기면서 실수를 전부 float32 로 만듭니다
  (0.6.1 실측 — float64 로 실린 `observation.contact.force` 가 `torch.float32` 로 옵니다). 규칙이 읽는 수가 그때
  달라지는 에피소드는 그 수로 다시 라벨링해 `outcome` · `severity` · `triggered_rules` 가 같아야 일치로 봅니다
  (`notes` 가 인용하는 수와 근거 요약은 실린 값으로만 대조 — 좁힌 힘으로 다시 낸 비율은 낮은 자리가 다릅니다). 달라지지 않으면 한 번으로 끝납니다. 좁힌 수를 **라벨러에 넣은**
  결과를 보는 것입니다(라벨러는 float64 로 비교) — float32 텐서를 임계값과 직접 비교하는 사람은 임계값도 float32 가
  되어, 임계값에서 float32 한 눈금 안의 값에서는 다른 답이 나올 수 있습니다
- **경로에 glob 문자(`*` `?` `[`)가 있는 자리의 배포본은 읽지 않습니다.** `datasets` 가 넘겨받은 파일 경로를 패턴으로
  풀기 때문입니다 — 실측: `x?y` 옆에 `x0y` 가 있으면 그쪽 파일을 먼저 읽고, `x[1]` 옆에 `x1` 이 있으면 그쪽만 읽습니다.
  보는 것은 **받은 그대로의 경로**(상대 경로면 작업 디렉터리만 앞에 붙임)입니다 — LeRobot 이 심링크를 풀지 않고 그 이름을
  그대로 넘기므로, 풀거나 `..` 을 정리해서 보면 glob 문자가 든 이름을 거쳐 가는 경로가 통과합니다
- 종료 코드 0 은 전 에피소드 일치, 1 은 다르거나 라벨러가 받지 않은 에피소드가 있음. 라벨러가 받지 않은 에피소드는
  건너뛰지 않고 사유와 함께 적습니다
- **일치가 말하지 않는 것** — 양쪽이 같은 라벨러이므로 라벨러가 규칙을 잘못 읽었으면 여기서도 잘못 읽습니다.
  규칙이 읽지 않는 열(관절 · 무게중심 · 영상 · 상자)은 보지 않습니다. 거리 두 열(`safety.min_distance_human` ·
  `safety.separation_required`)은 조건표 시나리오의 근거 요약이 읽고(있으면 로봇 카메라의 보이는 몫 · 카메라 자세 · 사람의
  뿌리 자세도), 그 밖의 시나리오에서는 보지 않습니다. `safety.separation_violation` 이 두 거리 열에서 나오는지는 검증기 C8 이 봅니다. 녹화가 어떤 이벤트를 **기록했는지**는 배포본에 없습니다(성립한 이름과 스키마의 허용
  이름뿐) — 「성립하지 않음」과 「기록되지 않음」은 내려받은 것만으로는 가를 수 없습니다. 실린 임계값이 라벨을 낸
  **그 값**이라는 것도 아닙니다 — 그 임계값으로 같은 라벨이 나온다는 것까지이고, 판정을 하나도 못 바꾸는 차이는 안 보입니다
- **녹화의 주기 자체는 배포본에 없습니다.** 변환기는 주기가 `1/fps` 에서 1e-9 초 안이면 받고 배포본에는 `fps` 만
  실리는데, 라벨러의 정지 예산은 그보다 작은 차이에도 한 프레임 달라질 수 있습니다(주기 0.0200000001 초면 17 이 아니라
  16 프레임). 그런 녹화의 라벨은 배포본에서 다시 나오지 않고 이 도구가 **차이로 보고합니다.** 기록기는 주기의 정확한
  배수를 쓰므로 지금까지의 배치에는 없는 경우이고, 변환기의 허용오차와 라벨러의 예산 계산을 맞추는 일은 하지 않았습니다

**실측(2026-10-07 · 26.11 배포본은 2026-11-17).** 네 배포본 모두 실린 라벨이 전부 다시 나왔습니다. 앞의 셋은 규칙이 읽는
수가 전부 float32 로 정확해(달라지는 에피소드 0건) 적재기로 읽어도 같은 수입니다. 26.11 배포본은 170건에서
규칙이 읽는 수가 float32 로 달라져, 그 수로 다시 유도한 결과도 함께 대조했고 같았습니다(위 「적재기가 넘기는 수로도 한 번 더 유도합니다」).

| 배포본 | 에피소드 · 스텝 | 일치 | 다시 유도한 결과 |
|---|---|---|---|
| S1d 400건 (공개하지 않은 배치) | 400 · 381,902 | **400 / 400** | fail 127 · near_miss 148 · nominal 125 |
| S1d 20건 (시험 배치) | 20 · 19,094 | 20 / 20 | fail 5 · near_miss 10 · nominal 5 |
| S1a 400건 (공개하지 않은 앞선 배치) | 400 · 383,462 | 400 / 400 | fail 99 · near_miss 161 · nominal 140 |
| 26.11 배포본 (`S1e` · `S1f` · `S1g` 각 99건) | 297 · 247,181 | **297 / 297** | fail 171 · near_miss 126 · nominal 0 |

**대조가 실제로 어긋남을 잡는지**는 S1d 400건의 입력을 하나씩 바꾼 사본(데이터는 심링크, 메타만 복사)으로 봤습니다.

| 바꾼 것 | 일치 | 달라진 항목 |
|---|---|---|
| 그대로 | 400 | – |
| BR-12 의 준정적 힘 한계를 100배로 | 296 | `triggered_rules` 104 · `outcome` 5 · `severity` 5 |
| 정지 규칙의 `max_stop_time` 을 0.42 로(실측한 정지 시간 p95 — 정본보다 24% 큼) | 382 | `triggered_rules` 18 · `outcome` 14 · `severity` 14 |
| `fps` 를 절반으로 | 280 | `triggered_rules` 120 · `outcome` 67 · `severity` 67 |
| 모든 에피소드의 `hold_release_step` 을 0 으로 | 397 | `triggered_rules` 3 |
| 라벨 열 셋을 한 에피소드씩 밀기 | 96 | `outcome` 271 · `severity` 281 · `triggered_rules` 304 |
| 에피소드 0 과 1 이 서로의 행을 가리키게 | – | 읽기를 거부(행의 `episode_index` 가 다름) |
| 에피소드 0 과 1 의 행 묶음을 파일 안에서 맞바꾸기(값은 그대로) | – | 읽기를 거부(`index` 가 자리와 다름) |
| 에피소드 표의 첫 두 행을 맞바꾸기 | – | 읽기를 거부(`episode_index` 가 자리와 다름) |

**조립기에 걸린 대조를 실제 청크로 끝까지 돌렸습니다**(2026-10-07 — 시험 배치의 청크 둘 · 20건, 실제 LeRobot
병합). 그 청크들은 정본의 랜덤화 선언을 고치기 전에 만든 것이라 지금 정본과 19군데가 다릅니다.

| 실행 | 결과 |
|---|---|
| 플래그 없이 | 정본 대조에서 거부(차이 19건을 나열) — 병합 전 |
| `--accept-canon-change` (받아들인 19건 가운데 라벨이 읽는 것은 없음) | 조립됨 — 「조립한 파일만으로 다시 유도한 라벨 20건」, 약 8초 |
| `--accept-canon-change` + BR-12 의 준정적 힘 한계를 100배로 고친 정본(받아들인 변경 20건) | 구조 검증은 다 통과하고 **라벨 검증에서 거부** — 일치 15 / 20, `triggered_rules` 가 달라진 다섯 에피소드를 짚고, 받아들인 변경을 `meta/rules.json` 의 그 숫자부터 나열. 앞 실행의 결과는 그대로 남음 |

**받은 쪽이 손으로 다시 유도할 때 걸리는 자리가 둘 있습니다**(S1d 400건에서 셈).

- **정지 규칙의 시간은 프레임 수로 셉니다.** 정본의 조건 문장은 그렇게 적지만 그 규칙의 `inputs` 에 `timestamp` 가
  있고, 배포본의 `timestamp` 열은 LeRobot 이 쓴 float32 입니다(스텝 간격이 일정하지 않음 — 라벨러의 주기 검사가 이 열을
  거부합니다). 그 열을 빼서 초로 비교하면 **예산과 정확히 같은 프레임 수에 멈춘 에피소드** 11건 가운데 8건이 초과로
  읽힙니다(뺄셈 결과가 그 시각의 float32 눈금에 달려 있어 전부는 아닙니다) — 정지 규칙 성립이 32건이 아니라 40건이 되고,
  늘어난 8건 중 6건은 `near_miss` 가 `fail` 로 바뀝니다. 이 도구는 주기를 `fps` 에서 얻습니다
- **접촉 규칙은 `notes.hold_release_step` 부터의 스텝만 봅니다**(−1 은 끝까지 붙잡힌 것이라 볼 스텝이 없음 — 283건).
  그 스텝 표시를 무시하면 세 에피소드에 접촉 규칙 하나가 더 성립하고 `outcome` 은
  바뀌지 않습니다. `R-HRI-CONTACT-01` 을 선언한 시나리오는 그에 더해 첫 `contact_onset` 스텝도 봅니다(규격서 §9.1)

26.11 배포본은 정지 규칙(`R-10218-STOP-01`)을 싣지 않아 앞의 자리는 해당하지 않습니다. 뒤의 자리 — 첫 접촉 스텝을 빼거나
모든 스텝을 비교하면 무엇이 바뀌는지 — 를 그 배포본에서 센 수는 데이터셋 상세 문서의 「이 배포본」 절에 있습니다.

**LeRobot 으로 읽으면 float64 열이 float32 로 옵니다**(S1d 400건에서 잰 것).
`observation.contact.force` · `safety.min_distance_human` · `safety.separation_required` 는 값이 전부 float32 로 정확해
달라지지 않고, 두 거리 열에서 float32 로 다시 계산한 이격거리 위반도 `safety.separation_violation` 과 전 스텝 같습니다.
`observation.contact.v_rel` · `safety.com` · `safety.zmp` · `safety.stability_margin` 은 float64 의 값이라 상대 6e-8 까지
달라집니다(v0.1 의 규칙이 읽지 않는 열). 실린 자릿수가 필요하면 parquet 을 직접 읽습니다 — 재실행이 「비트 단위로」
재현한다는 것도 parquet 에 실린 값에 대한 말이고, 이 네 열에서는 LeRobot 으로 읽은 값과 비트가 맞지 않습니다.

### 점수표 — 배포본 하나를 네 칸으로

`R-TS15066-SSM-02` 를 선언한 시나리오(`S1e` · `S1f` · `S1g`)의 배포본을 시나리오마다 네 칸으로 요약합니다 — 안전(결과 ·
심각도와, 에피소드마다 가장 작았던 정지거리 여유의 분포) · 과제(완료율 · 걸린 시간의 중앙값) · 접촉의 성격(다가가던 로봇이
낸 접촉 / 다가가지 않던 로봇의 접촉 — 첫 접촉 때 로봇의 다가가는 속도가 규칙의 문턱 이하, 곧 닿을 때 로봇이 사람 쪽으로 움직이고 있지 않던 접촉으로, 옆으로
걷거나 돌던 로봇도 듬 — 와, 그 각각에서 사람이 같은 문턱보다 빠르게 다가가던 것) · 이상 사건(로봇 넘어짐과
그 가운데 접촉 없이 · 넘어질 때 사람과의 거리 · 랙 · 바닥 접촉). **한 숫자로 합치지 않고 순위를 매기지 않습니다.** `degenerate`
에피소드는 빼고 그 수를 적습니다. 문턱은 배포본의 `meta/rules.json` 에서 읽고, 배포본만 읽습니다(Isaac · Hub 없음).
네 칸은 시나리오의 **무작위 행**만 셉니다. 격자 행(조건 기록의 `design` 이 `grid` 이고 `grid_cell` 이 `[로봇 쪽 칸, 사람 쪽 칸]` 인 행 —
`design` 이 없는 기록은 무작위 행)은 네 칸에서 빼고 `grid_episodes` 로 세며, 네 칸 옆의 **실패 지도**로 그립니다: 격자 행마다 한 칸에
그 행의 결과와 심각도를 놓습니다(가로는 로봇이 받는 과업의 양, 세로는 사람의 양 — 두 축의 이름과 범위, 칸의 수는 배포본 `meta/scenarios.json` 의
`conditions.grid` 에서 읽습니다). 행이 없는 칸은 비고, `degenerate` 인 행의 칸은 그렇다고 적습니다. 격자 행의 열 값은 조건표 열의 창 밖이어도
됩니다. 격자 행만 있고 격자 블록이 없는 항목(또는 그 반대)은 지도 대신 「실패 지도 없음」과 그 까닭을 냅니다.
표는 적재기가 읽는 대로 — 저장된 순서 그대로, 에피소드가 가리키는 행에서 — 읽습니다(「실린 라벨은 배포본만으로 다시 나옵니다」의 도구와 같은 읽기). 그렇게
읽히지 않는 배포본(행이 제자리에 없음 · 다른 에피소드의 행을 가리킴 · 경로에 glob 문자 · 읽을 수 없는 parquet · 에피소드 없음)과,
그 규칙을 선언하지 않은 시나리오의 배포본(열이 없음), 적힌 대로 셀 수 없는 배포본(거리 · 요구 이격거리 · 두 쪽의 다가가는 속도에 유한하지 않은 값 · 한 칸에 둘인 격자 행 · 격자 밖의 칸 · 정수 둘이 아닌 `grid_cell` · 읽을 수 없는 격자 블록 · 에피소드의 마지막 스텝 뒤의 완료 스텝 · 완료 기준이 있는 그 시나리오의 null 완료 스텝)은 셀 수 없고, 종료 코드 2 로 이유를 말합니다.

```bash
.venv/bin/python generation/scorecard.py --dataset <배포본> --out <점수표.json> --markdown <점수표.md>
```

## `envs/s1a/` 접촉력 — 붙잡기와 해제

환경은 로봇-더미 접촉을 부위별로 생성 · 기록합니다. 로봇 토르소와 더미의 특정 부위(BR 코드) 사이, 단일
Newton 쌍으로 힘이 잡히고, 어느 스텝에 닿았는지와 어느 부위가 닿았는지가 기록됩니다.

더미를 세워 두는 것은 외부 렌치 서보입니다(PD 홀드, `mdp/commands.py` 의 `_advance_dummy`). **홀드가 켜진 채 잰
접촉력은 로봇-인체 접촉만의 측정값이 아닙니다** — 서보 힘이 섞입니다. 바닥에 세운 채 끄는 S1a 의 이동 단계 150 스텝에서
잰 값입니다(`check_dummy_hold.py`):

| | 값 |
|---|---|
| 홀드력 평균 / 최대 | 1,213 N / 5,810 N (더미 자체 무게 859 N) |
| **접지 반력이 0 인 스텝** | **54/150 (36.0%)** — 더미가 공중에 떠 있음 |
| 최대 접지 반력 | 6,872 N (체중의 8배) |

강체 인형을 지면 마찰에 맞서 수평으로 끌면 기울고, 자세 보정 토크가 그걸 지면으로
되받아 튕깁니다. 그 상태에서 잰 접촉력은 무엇의 측정값도 아닙니다. S1d 는 붙잡는 동안 인형을 바닥에서 약 5 mm
들어 옮기지만(카탈로그 §5) 홀드가 접촉에 섞이는 것은 같고, 아래의 해제도 같습니다.

### 접촉 직전에 홀드를 끕니다

**해결은 조인트가 아니라 해제입니다.** 홀드가 어떤 형태든 접촉을 붙들면 그 반작용이
접촉력에 섞입니다. 그래서 **로봇이 더미 표면에서 `dummy_release_gap_m`(기본 0.10 m)
안으로 들어오면 홀드 서보를 그 에피소드 동안 완전히 끕니다**
(`S1aStraightWalkCommand._update_hold_release`). 그때부터 더미는 자유 물체이므로
접촉력은 **로봇-인체 상호작용 + 더미 자신의 관성**만으로 결정됩니다. 걸어가는 사람이
부딪히는 S1a 의 상황에 맞는 모델입니다.

**없앤 것은 서보 오염입니다.** 표 A.2 의 어느 열(transient / quasi-static)과 비교할지는 접촉 유형 분류가
정합니다 — 스키마 §8.1 「산출 방법」과 `classify_contact_type()`. `S1a` · `S1d` 에서는 크기 비교를 해제 뒤의
스텝에만 하고(`notes.hold_release_step` 부터), 분류되지 않은 접촉은 비교 대상이 되지 않으며 그런 접촉이 있는
에피소드는 라벨의 `not_evaluated` 에 스텝과 함께 남습니다. 조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 분류되지 않은
접촉도 엄한 쪽 한계 — 준정적 한계와 동적 한계 가운데 작은 쪽, 동적 접촉이 허용되지 않는 부위는 어떤 힘이든 — 로
비교합니다(판정하는 스텝은 이 절의 마지막 문단 · 규격서 §9.1).

**해제 조건은 둘이고, 보장은 그중 하나뿐입니다.**

| 조건 | 성격 |
|---|---|
| `min_distance_human` ≤ `dummy_release_gap_m`(0.10 m) | **최선 노력.** 이 값은 표면 간 거리가 아닙니다(아래) — 실측에서는 첫 접촉 3 스텝(0.06s) 전에 걸렸습니다 |
| 로봇 비-발 링크에 접촉력 > 10 N | **보장.** 홀드가 접촉을 *관통해서* 유지되는 일은 없습니다 |

간격 게이트를 보장으로 쓸 수 없는 이유: **`min_distance_human` 은 표면 대 표면 거리가
아닙니다.** 스키마가 명시하듯 로봇 쪽은 링크 **원점**을 점으로 쓰고(콜라이더 크기를
전혀 반영하지 않습니다), 인체 쪽은 부위마다 0.1 m 구 하나로 근사하는데 실제 축 방향
연장은 0.265 m 까지 갑니다(스키마 v0.2 「`safety.min_distance_human` 산출 방법」).
자산에서 실측한 바디 원점→표면 최대 거리는 로봇 0.755 m·더미 0.866 m 라, 이걸 되돌려
더하면 게이트가 1.6 m 밖에서 걸려 풀린 더미가 접촉 전에 넘어집니다. 그래서 간격
게이트는 작게 두고, 보장은 접촉 백스톱이 집니다.

**주장이 아니라 데이터로 확인합니다.** 에피소드 메타의 `hold_contaminated_steps` 가
"접촉이 있는데 홀드력도 0 이 아니었던" 스텝 수를 셉니다 — 백스톱이 발동한 경우 1,
거리 게이트가 먼저 걸린 경우 0 이 기대값이고, 그보다 크면 그 스텝들의
`contact_force` 는 쓸 수 없습니다.

**조건표가 있는 시나리오(`S1e` · `S1f` · `S1g`)는 거리 게이트를 쓰지 않습니다.** 환경이 `dummy_release_gap_m` 을
`None` 으로 두어(`generation/envs/hri/hri_env_cfg.py`), 붙잡기는 로봇 몸이나 든 상자가 인형에 처음 힘을 준 스텝 — 첫
`contact_onset` 이자 상자를 놓는 스텝, 같은 힘 행렬 · 같은 문턱(`generation/envs/hri/commands.py`) — 에만 풀립니다. 그 인형은
균형 제어가 없는 수동 몸이라, 거리로 미리 풀면 닿지 않고 스치는 근접사고에서 보조 없이 걷다 닿기 전에 기웁니다(2026-10-20 에 거리
게이트로 돌린 한 행에서 풀린 뒤 첫 접촉 전까지 25 스텝 동안 골반의 기울기가 1.9° 에서 18.4° 로). 그래서 접촉이 있는 에피소드는
`hold_contaminated_steps` 가 1(첫 접촉 스텝)이고 `hold_release_step` 은 그 다음 스텝입니다. 라벨러는 이 시나리오들(`R-HRI-CONTACT-01`
을 선언한 시나리오)에서 그 첫 접촉 스텝도 판정합니다 — 표 A.2 한계와의 비교와 머리 · 얼굴의 `S3` 가 첫 `contact_onset` 스텝과
`hold_release_step` 부터의 스텝을 읽어, 한 스텝만 닿는 접촉도 비교됩니다(`generation/labeling/label_episodes.py` · 규격서 §9.1).
닿지 않은 에피소드는 `hold_release_step` 이 `-1` 입니다.

> **`hold_force` 는 그 스텝의 물리에 실제로 작용한 힘입니다.**
> `CommandManager.compute()` 가 데시메이션 루프 **뒤**에 돌기 때문에 거기서 쓰는 wrench 는
> *다음* 스텝의 것입니다. 그것을 그대로 기록하면 모든 행의 홀드력이 한 스텝 밀리고, **백스톱이 발동한
> 바로 그 스텝에서 0 을 읽어** 오염 카운터가 자기 사례를 세지 못합니다. 그래서 `hold_force_in_effect` 를
> 기록합니다. `hold_release_step` 도 같은 이유로 **홀드 없이 물리가 돈 첫 스텝**을 가리킵니다.

실측(6 에피소드, 1000 스텝):

- **BR-12 외 부위에 힘이 잡힌 333 스텝 전부 `hold_force == 0`** (100.0%). 이 수는 넘어진 뒤 그 부위들이 받는
  지면 하중을 포함합니다 — 로봇 쪽 채널로 게이트한 로봇-더미 접촉은 그 가운데 26 스텝(655~680)입니다
- 접촉이 난 에피소드는 652 스텝에서 해제되고 655 스텝부터 접촉 — 3 스텝(0.06s) 여유라
  풀린 더미가 넘어질 시간이 없습니다
- 접촉이 없던 5 에피소드에서는 해제가 걸리지 않아 홀드가 끝까지 켜져 있습니다
  (최소거리 0.454~3.884 m)

에피소드 메타에 `hold_release_step` 이 남으므로, 어느 스텝부터가 자유 물체 구간인지
데이터만 보고 알 수 있습니다.

> **접촉 유형.** 스키마 §8.1 「산출 방법」과 `classify_contact_type()` 참고. 구분은 지속시간이 아니라
> **후퇴 가능 여부**이고, S1a 가 만들 수 있는 협착은 「넘어진 인체를 로봇이 지면 쪽으로 누름」 하나뿐이라
> 부위 높이로 판정합니다. BR-12 외 부위의 힘을 전부 로봇 접촉으로 세면 안 됩니다 — 넘어진 뒤에는 그
> 부위들도 지면 하중을 받습니다. 로봇 쪽 채널로 게이트해야 실제 로봇-더미 접촉 구간이 나옵니다
> (위 에피소드에서 26 스텝, 655~680).

> **`check_dummy_hold.py` 는 기본적으로 해제하지 않습니다** (`--release-gap` 음수).
> 그 스크립트가 재는 것은 홀드 서보 자체의 능력이고, 해제해 버리면 잴 것이 남지
> 않습니다. 운영 동작을 보려면 `--release-gap 0.10` 을 주십시오. 이 스크립트는 기본 인자에서
> 종료 코드 1 을 냅니다 — 홀드 서보의 최악 요 오차가 허용값을 넘는 것이 알려진 미결이고, 허용값을 늘려
> 통과시키지 않았습니다.

## `capture_environment.py`

실행 환경의 버전을 수집해 저장소 루트의 `environment.lock` 을 생성합니다.
**Isaac Sim 을 부팅하지 않으므로** 일반 `python3` 로 돕니다.

```bash
make lock          # 수집 → environment.lock
make lock-check    # 대량 생성 직전 드리프트 대조
```

설치 경로가 다르면 덮어쓸 수 있습니다: `make lock ISAACLAB_PATH=/opt/IsaacLab`

`environment.lock` 의 `sim_backend` 블록이 스키마 §5 의 동명 필드에 그대로 들어갑니다.
버전을 손으로 적지 않는 이유는 관절 이름과 같습니다 — 수기 문자열은 환경이 바뀌어도
그대로 남아 아무도 알아채지 못합니다.

`--check` 는 환경이 lock 과 다르면 종료 코드 1 을 냅니다. **대량 생성을 시작하기 전에
반드시 통과시키십시오.** 도중에 버전이 바뀌면 앞뒤 에피소드의 `sim_backend` 가 달라집니다.

CI 에는 넣지 않습니다 — CI 에는 Isaac Sim 이 없어 항상 실패합니다.

## Isaac 을 부팅하는 스크립트는 `isaac_boot.py` 를 거칩니다

새 스크립트에서 `AppLauncher`·`SimulationApp` 을 직접 만들지 마십시오.

```python
from generation.isaac_boot import launch_app          # Isaac Lab 쪽
simulation_app = launch_app()[1]

from generation.isaac_boot import launch_simulation_app  # isaacsim 직접 사용
app = launch_simulation_app()
```

**이유는 종료 코드입니다.** 스크립트들은 실행 구간을 `try/finally` 로 감싸고
`simulation_app.close(exit_code=code)` 로 코드를 보존하는데, 그 `try` 는
`simulation_app` 이 생긴 **뒤에야** 걸립니다. 부팅 자체가 실패하면 Kit 이 이미
등록해 둔 atexit 핸들러가 기본값 `exit_code=0` 으로 닫고 프로세스를 강제
종료하므로, **트레이스백이 다 찍히고도 종료 코드가 0** 이 됩니다.

2026-08-20 실측(GPU 가 있는 기계에서 `CUDA_VISIBLE_DEVICES=""`):

| | 종료 코드 |
|---|---|
| 부팅 실패 (`RuntimeError: No CUDA GPUs are available`) — 헬퍼 없이 | **0** |
| 같은 실패 — 헬퍼로 | 1 |
| 정상 실행 (`check_ssm.py` 완주) | 0 |

헬퍼는 실패 시 `os._exit(1)` 로 끝냅니다 — atexit 를 돌리지 않는 유일한 종료라
Kit 이 코드를 덮을 수 없습니다. 출력은 그 전에 직접 flush 합니다.

`tests/generation/test_isaac_boot.py` 가 CI 에서 강제합니다 — 헬퍼 밖에서
부팅하는 스크립트가 생기면 붉은불이 됩니다.

## 환경이 둘로 나뉘어 있습니다

| 환경 | 용도 | 무엇이 들어 있나 |
|---|---|---|
| `~/isaacsim` (번들 파이썬) | **생성** — 씬·물리·계측 | Isaac Sim 6.0.1, Isaac Lab v3.0.0-beta2.patch1, torch 2.10.0+cu128 |
| `.venv-lerobot` | **변환** — LeRobot v3.0 포맷 | lerobot 0.6.1 (`CODEBASE_VERSION = v3.0`), torch 2.11.0+cu130 |
| `.venv` | **검증·문서** | pandas, pyarrow, numpy, pyyaml, markdown |

lerobot 은 의존성이 222개고 torch 를 자체 버전으로 끌고 옵니다. 생성 환경에 함께 넣으면
Isaac Lab 설치가 깨질 수 있어 분리했습니다. 변환 단계는 Isaac Sim 이 필요 없으므로
분리해도 손해가 없습니다. `environment.lock` 의 `paths.lerobot_python` 이 위치를 기록합니다.

`.venv` 는 `make venv` 가 만들고, `.venv-lerobot` 은 아래처럼 만듭니다. LeRobot 의 `dataset`
선택 의존성이 표를 읽고 쓰는 `datasets` 와 영상을 다루는 `av` 를 깝니다 — 그것 없이
`lerobot==0.6.1` 만 깔면 변환이 LeRobot 의 데이터셋 모듈을 import 하는 자리에서 멈춥니다.
LeRobot 0.6.1 은 Python 3.12 이상을 요구하므로(`Requires-Python: >=3.12`), `python3` 이 그보다
낮으면 3.12 이상의 인터프리터로 만듭니다.

```bash
python3 -m venv .venv-lerobot
.venv-lerobot/bin/pip install "lerobot[dataset]==0.6.1"
```

실행 방법이 환경마다 다릅니다.

```bash
~/isaacsim/python.sh <스크립트>          # 생성 — Isaac Sim 필요
.venv-lerobot/bin/python <스크립트>      # 변환 — lerobot 필요
.venv/bin/python <스크립트>              # 검증 — 둘 다 불필요
```
