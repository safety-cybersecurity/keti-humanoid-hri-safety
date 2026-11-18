# 더미 시각 메시의 출처와 라이선스

`generation/dummy/skin/` 이 만드는 시각 메시(`build/keti_skin.obj` · `keti_skin.mtl` · `build/textures/`)와
그것을 붙인 `keti_dummy_ts15066_v1_skin.usd` 의 사람 형상 부분이 어디서 왔고 무슨 조건으로 배포되는지
적습니다. 물리(콜라이더·조인트·질량)는 이 문서와 무관하게 `build_dummy.py` 와 `anthropometry.yaml` 이
정합니다.

**결론 — 시각 메시와 텍스처는 CC0 1.0 입니다.** 배포본(CC BY 4.0)에 넣는 데 제약이 없고, 표기 의무도
없습니다. 그래도 자산 인벤토리 §1 의 원칙대로 `NOTICE` 에 출처를 남깁니다.

## 무엇으로 만들었나

| | |
|---|---|
| 도구 | MakeHuman 의 Blender 플러그인 **MPFB 2.0.17** (Blender 확장 플랫폼 배포판) · Blender 5.2.2 LTS |
| 절차 | `make_skin.py` 가 `skin_spec.yaml` 대로 헤드리스로 사람을 만들고 OBJ 로 내보냄. 매개변수·자세·키는 전부 명세에 있음 |
| 자산 | MakeHuman **시스템 자산 팩** `makehuman_system_assets_cc0.zip` 의 피부 1 · 의복 2 · 머리 1 · 눈 1 · 눈썹 1 |
| 얼굴 | **합성**입니다. 실존 인물의 얼굴·스캔·사진을 쓰지 않았습니다. 피부 텍스처는 팩의 범용 텍스처입니다 |

MPFB 의 **소스 코드는 GPL-3.0-or-later** 이지만 산출물에는 미치지 않습니다. MPFB 의 LICENSE.md 는
「MPFB 의 어떤 산출물에도 프로그램 로직의 흔적이 없다」고 보며 파일 내보내기(FBX · **OBJ** · DAE · …)·렌더·
저장 모델에 「어떤 권리도 주장하지 않는다」고 적습니다. 산출물은 따라서 **팩 자산(CC0) 과 우리 입력의
결합**이고, 그 팩이 CC0 이므로 결합물에도 제약이 없습니다.

- MPFB LICENSE.md — <https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md>
- MakeHuman 라이선스 안내 — <https://static.makehumancommunity.org/about/license.html>
  (「All core assets are shared under Creative Commons, CC0」)
- MakeHuman FAQ 「모델을 팔아도 되는가」 — MakeHuman 위키가 닫혀 인터넷 아카이브의 보관본을 가리킵니다:
  <https://web.archive.org/web/20260509175146/http://www.makehumancommunity.org/wiki/FAQ:Can_I_sell_models_created_with_MakeHuman%3F>
  (내보낸 모델은 CC0 — 이 문서는 MakeHuman 1.x 를 기준으로 쓰였고, MPFB 는 위 LICENSE.md 가 같은 취지를 적습니다)

## 시스템 자산 팩

| | |
|---|---|
| 파일 | `makehuman_system_assets_cc0.zip` · 280,737,770 B |
| 출처 | <https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip> |
| SHA256 | `b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107` |
| 라이선스 | 팩 메타데이터 `packs/makehuman_system_assets.json` 이 항목마다 `"license": "CC0"` 를 적고, 각 `.mhclo`/`.mhmat` 머리글이 「This asset was explicitly released as CC0 in september 2020」 을 적음 |
| CC0 이전의 저작권자 (머리글 기재) | Data Collection AB (MakeHuman 의 법적 연락처) · Joel Palmius · Jonas Hauquier |

설치는 MPFB 안에서 `AssetService.fix_and_extract_asset_pack_zip(<zip>, LocationService.get_user_data())`.
팩이 없으면 `make_skin.py` 가 멈추고 위 주소와 해시를 안내합니다.

## 쓴 자산 (팩 안 경로)

| 종류 | 이름 | 파일 |
|---|---|---|
| 피부 | Middleage asian male | `skins/middleage_asian_male/middleage_asian_male.mhmat` |
| 의복 | Male worksuit01 | `clothes/male_worksuit01/male_worksuit01.mhclo` |
| 의복 | Shoes01 | `clothes/shoes01/shoes01.mhclo` |
| 머리 | Short02 | `hair/short02/short02.mhclo` |
| 눈 | Low-poly | `eyes/low-poly/low-poly.mhclo` |
| 눈썹 | Eyebrow001 | `eyebrows/eyebrow001/eyebrow001.mhclo` |

내보내기에 함께 나가는 **텍스처 8장**과 사람을 만든 **팩 파일 16개**(위 자산 여섯의 `.mhclo`/`.mhmat` 과
그것이 가리키는 `.obj`·`.mhmat`)의 이름·SHA256·팩 안 경로는 `textures.lock` 이 한 줄씩 적고, `make_skin.py` 가
매번 대조합니다. **lock 은 모든 행을 팩 zip 과 바이트로 대조한 뒤에만 씁니다**(`--pack <zip>`, zip 의 SHA256 은
위 표) — MPFB 의 자산 디렉터리에 들어 있다는 것만으로는 그 파일이 팩에서 왔다는 증명이 되지 않기 때문입니다.
기본 메시(basemesh)·모프 타깃·내장 리그(`default`)는 MPFB 에 동봉된 자산이며 같은 LICENSE.md 가 CC0 로
적습니다.

## 배포본에 실릴 때

- 데이터셋은 CC BY 4.0 으로 나가고, 이 시각 메시는 CC0 라 그 안에 들어가는 데 제약이 없습니다
- `meta/assets.json` 의 `keti_dummy_ts15066_v1_skin` 항목이 이 문서를 `skin_source` 로 가리키고,
  `NOTICE` §2 가 출처를 적습니다
- 메시가 프리미티브 밖으로 나오는 곳(옷 주름 · 손끝 · 발 — 발은 12부위에 없음)은 접촉 계측 대상이
  아닙니다. 접촉은 콜라이더에서만 납니다. 이것은 자산의 한계이지 라이선스 문제가 아닙니다
