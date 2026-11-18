#!/usr/bin/env python3
"""더미의 접촉이 부위 코드 BR-01~12 로 분해되는지 확인한다.

    ~/isaacsim/python.sh generation/dummy/check_contact_resolution.py

시나리오 카탈로그 §9-2 의 미결 사항 — "BR-01~12 구분을 물리엔진 접촉점으로 안정적으로
분해할 수 있는지" — 를 표준 수치 없이 확인하는 시험이다. 분해가 안 되면 그것은 임계값과
무관한 구조 문제이고, 데이터를 만들기 전에 알아야 한다.

**왜 떨어뜨리지 않고 겹쳐 두는가.** 낙하로 시험하면 어느 부위에 먼저 닿을지가 형상과
중력에 따라 달라져, 실패했을 때 그것이 분해 실패인지 조준 실패인지 구분되지 않는다.
부위마다 프로브 구를 정면(+x)에 살짝 겹치게 두면 접촉 상대가 하나로 정해지므로,
보고된 코드가 기대한 코드와 다르면 그것은 분해 문제다.

**프로브를 부위 수만큼 한 번에 놓는다.** 프로브 하나를 옮겨 가며 시험하면 물리엔진이
붙은 뒤의 USD 변경을 읽지 않아 프로브가 제자리에 남는다. 접촉 헤더에는 양쪽 행위자가
모두 들어 있으므로, 프로브와 부위를 짝지어 한 번의 시뮬레이션으로 전부 확인할 수 있다.

중력은 끈다. 켜 두면 시험이 시작되기 전에 더미가 주저앉는다.

종료 코드: 0 전 부위 분해 성공 / 1 실패한 부위 있음 / 2 자산 없음 / 3 인자 오류.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# `~/isaacsim/python.sh <script>` puts only the script's own directory on
# sys.path, so `generation` is not importable without this.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.isaac_boot import launch_simulation_app  # noqa: E402
DEFAULT_USD = REPO_ROOT / "generation/dummy/build/keti_dummy_ts15066_v1.usd"

PROBE_RADIUS = 0.03
OVERLAP = 0.003          # 초기 침투량. 접촉이 첫 스텝에 잡히도록 하는 최소값
AABB_TOL = 0.010         # 기하 기대값의 여유. PhysX 의 contact offset 을 감안한 값
EXPECTED_CODES = {f"BR-{i:02d}" for i in range(1, 13)}   # 스키마 §8 의 부위 코드


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="더미 접촉의 부위 분해 가능 여부를 확인한다")
    p.add_argument("--usd", default=str(DEFAULT_USD), help="더미 USD 경로")
    p.add_argument("--steps", type=int, default=4,
                   help="물리 스텝 수. 브로드페이즈가 자리 잡을 만큼만 돌린다")
    p.add_argument("--dt", type=float, default=1.0 / 200.0, help="물리 스텝 간격 (s)")
    return p.parse_args()


def main() -> int:
    # SimulationApp.close() 는 프로세스를 강제 종료하므로, 표준출력이 파일이나 파이프로
    # 묶여 블록 버퍼링되면 시험 결과가 버퍼째 사라진다. 줄 단위로 바꿔 둔다.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    usd = Path(args.usd)
    if not usd.exists():
        print(f"더미 자산이 없습니다: {usd}", file=sys.stderr)
        print("  ~/isaacsim/python.sh generation/dummy/build_dummy.py --geometry-only",
              file=sys.stderr)
        return 2

    # 부팅을 헬퍼에 맡긴다 — 아래 try/finally 는 app 이 생긴 뒤에야 걸리므로,
    # 그전에 실패하면 Kit 의 atexit 가 종료 코드를 0 으로 덮는다
    # (generation/isaac_boot.py 참고).
    app = launch_simulation_app()

    # Set before every return below (and left at this default if some
    # unforeseen path skips them all) so the finally block always has a
    # real value to hand to app.close(exit_code=...) — see that call for
    # why this indirection exists at all.
    exit_code = 1
    try:
        from pxr import (Usd, UsdGeom, UsdPhysics, UsdUtils,
                         PhysicsSchemaTools, PhysxSchema, Gf, Sdf)
        import omni.physx

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityMagnitudeAttr(0.0)       # 시험 전 더미가 주저앉지 않도록

        stage.DefinePrim("/KetiDummy", "Xform").GetReferences().AddReference(str(usd))

        region_of: dict[str, str] = {}              # 부위 프림 경로 → BR 코드
        aabb: dict[str, tuple] = {}                 # 부위 프림 경로 → 월드 AABB
        placements: list[tuple[str, str, Gf.Vec3d]] = []
        for prim in stage.Traverse():
            attr = prim.GetAttribute("keti:bodyRegion")
            code = attr.Get() if attr else None
            if not code:
                continue
            path = str(prim.GetPath())
            region_of[path] = code
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            # The body's collider, not its subtree. In the v0.1 asset the body prim is the
            # shape; the skinned variant makes it an Xform with the shape under it as
            # `collider` (purpose guide) beside a visual `skin` mesh. A purpose-filtered
            # bound of the subtree would then measure the jacket and put the probe where
            # nothing collides, so the bound is the collider's own extent.
            shapes = [prim] if prim.HasAPI(UsdPhysics.CollisionAPI) else \
                [c for c in prim.GetChildren() if c.HasAPI(UsdPhysics.CollisionAPI)]
            if len(shapes) != 1:
                print(f"{path}: 콜라이더가 하나여야 합니다 ({len(shapes)}개)", file=sys.stderr)
                exit_code = 1
                return exit_code
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(shapes[0]),
                                                                 Usd.TimeCode.Default())
            if not extent:
                print(f"{path}: 형상의 extent 를 계산할 수 없습니다", file=sys.stderr)
                exit_code = 1
                return exit_code
            xf = UsdGeom.Xformable(shapes[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            box = Gf.Range3d()
            for x in (extent[0][0], extent[1][0]):
                for y in (extent[0][1], extent[1][1]):
                    for z in (extent[0][2], extent[1][2]):
                        box.UnionWith(xf.Transform(Gf.Vec3d(x, y, z)))
            mid = box.GetMidpoint()
            # 정면(+x) 바깥면에 살짝 겹치게. 다른 부위와 겹치지 않는 방향이다.
            aabb[path] = (tuple(box.GetMin()), tuple(box.GetMax()))
            placements.append((path, code,
                               Gf.Vec3d(box.GetMax()[0] + PROBE_RADIUS - OVERLAP,
                                        mid[1], mid[2])))

        if not placements:
            print("BR 코드가 붙은 프림이 없습니다 — 자산이 잘못되었습니다", file=sys.stderr)
            exit_code = 1
            return exit_code

        # 아티큘레이션 루트를 월드에 고정한다. 자산은 자유롭게 두되, 시험에서는 프로브가
        # 파고든 만큼의 반발로 더미 전체가 밀려나면 부위가 남의 프로브를 스치게 되어
        # 분해 실패와 구분되지 않는다. 실제 시나리오에서도 인체는 바닥에 서 있다.
        roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            print(f"아티큘레이션 루트가 {len(roots)}개입니다 (1개여야 함)", file=sys.stderr)
            exit_code = 1
            return exit_code
        # 자가 충돌을 끈다. 자산은 물리엔진 중립이므로 이것은 씬을 구성하는 쪽의 몫이고,
        # 여기서는 시뮬레이터가 떠 있어 PhysxSchema 를 쓸 수 있다. 끄지 않으면 인접 부위가
        # 매 스텝 서로 닿아 부위별 접촉 배열이 상시 포화된다.
        PhysxSchema.PhysxArticulationAPI.Apply(roots[0]).CreateEnabledSelfCollisionsAttr(False)

        rmn, rmx = aabb[str(roots[0].GetPath())]
        anchor = UsdPhysics.FixedJoint.Define(stage, "/anchor")
        anchor.CreateBody1Rel().SetTargets([roots[0].GetPath()])
        # body0 이 없으면 월드다. 로컬 프레임을 주지 않으면 루트를 원점으로 끌어당긴다.
        anchor.CreateLocalPos0Attr(Gf.Vec3f(*[float((a + b) / 2) for a, b in zip(rmn, rmx)]))
        anchor.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

        probe_of: dict[str, str] = {}               # 프로브 경로 → 겨냥한 부위 프림 경로
        probe_pos: dict[str, tuple] = {}
        stage.DefinePrim("/probes", "Xform")
        for path, _code, pos in placements:
            name = path.rsplit("/", 1)[-1]
            ppath = f"/probes/probe_{name}"
            probe = UsdGeom.Sphere.Define(stage, ppath)
            probe.CreateRadiusAttr(PROBE_RADIUS)
            probe.AddTranslateOp().Set(pos)
            # 더미는 저작한 그대로 동적 아티큘레이션으로 두고 루트만 월드에 고정한다.
            # 예전에는 부위를 kinematic 으로 바꿔 얼렸는데, 그러면 PhysX 가 고정조인트
            # 16개를 전부 "static body 사이에는 조인트를 만들 수 없다" 며 거부해, 시험이
            # 실제 아티큘레이션이 아니라 독립 콜라이더만 검증하게 된다.
            # 프로브는 동적 강체다. 정적 콜라이더로 두면 아티큘레이션 링크와의 접촉이
            # 보고되지 않는 경우가 있었다.
            UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
            UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr(0.02)
            PhysxSchema.PhysxContactReportAPI.Apply(probe.GetPrim()).CreateThresholdAttr(0.0)
            probe_of[ppath] = path
            probe_pos[ppath] = tuple(pos)

        # 프로브끼리는 부딪히지 않게 한다. 서로 닿아도 시험과 무관한 접촉만 늘어난다.
        group = UsdPhysics.CollisionGroup.Define(stage, "/probes/noProbeProbe")
        includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
        for ppath in probe_of:
            includes.AddTarget(Sdf.Path(ppath))
        group.CreateFilteredGroupsRel().AddTarget(group.GetPath())

        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        physx = omni.physx.get_physx_simulation_interface()

        pairs: set[tuple[str, str]] = set()         # (프로브 경로, 부위 경로)
        self_pairs: set[tuple[str, str]] = set()    # 부위끼리 닿은 쌍

        def on_contact(contact_headers, contact_data):
            for h in contact_headers:
                # 접촉 헤더의 actor 는 경로 문자열이 아니라 PhysX 가 인코딩한 정수다.
                a = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                for x, y in ((a, b), (b, a)):
                    if x in probe_of and y in region_of:
                        pairs.add((x, y))
                if a in region_of and b in region_of:
                    self_pairs.add(tuple(sorted((a, b))))

        # 첫 스텝의 접촉만 본다. 저작한 겹침이 그대로 살아 있는 유일한 시점이고, 그 뒤로는
        # 침투가 풀리며 부위가 움직여 "어디에 닿았나" 가 초기 배치와 달라진다.
        sub = physx.subscribe_contact_report_events(on_contact)
        physx.attach_stage(stage_id)
        for _ in range(args.steps):
            physx.simulate(args.dt, 0.0)
            physx.fetch_results()
        physx.detach_stage()
        del sub

        hit: dict[str, set[str]] = defaultdict(set)
        for ppath, rpath in pairs:
            hit[ppath].add(region_of[rpath])

        # 기하 기대값. 프로브 구가 어느 부위의 AABB 와 실제로 겹치는지 계산해 둔다.
        # "겨냥한 부위가 들어 있으면 통과" 만으로는 전 부위를 싸잡아 보고해도 통과하므로,
        # 겹치지도 않는 부위가 보고되면 그것은 잘못된 귀속이다.
        def overlaps(center, path) -> bool:
            mn, mx = aabb[path]
            d2 = sum((c - min(max(c, lo), hi)) ** 2
                     for c, lo, hi in zip(center, mn, mx))
            return d2 < (PROBE_RADIUS + AABB_TOL) ** 2

        results = []
        for ppath, target in probe_of.items():
            want = region_of[target]
            got = sorted(hit.get(ppath, set()))
            center = probe_pos[ppath]
            near = {region_of[rp] for rp in region_of if overlaps(center, rp)}
            missed = want not in got
            # 인접 부위가 함께 보고되는 것은 형상 그대로다 — 얼굴 앞을 누르면 두개골에도
            # 닿는다. 스텝 필드는 부위마다 값을 따로 담으므로 그대로 표현된다.
            spurious = sorted(set(got) - near)
            results.append((target.rsplit("/", 1)[-1], want,
                            not missed and not spurious, got, spurious))

        good = [r for r in results if r[2]]
        print(f"\n부위 분해 시험 — {len(good)}/{len(results)} 통과")
        for name, code, ok, got, spurious in results:
            if ok:
                extra = [g for g in got if g != code]
                note = f"   함께 닿은 인접 부위: {extra}" if extra else ""
            elif spurious:
                note = f"   겹치지도 않는 부위가 보고됨: {spurious}"
            else:
                note = f"   보고된 코드: {got or '없음'}"
            print(f"  {'통과' if ok else '실패'}  {name:16s} 기대 {code}{note}")

        # 자산이 부위를 통째로 잃어도 프로브가 안 생길 뿐이므로, 코드 커버리지를 따로 본다.
        covered = set(region_of.values())
        if covered != EXPECTED_CODES:
            print(f"\n부위 코드 커버리지 불일치 — 없음: {sorted(EXPECTED_CODES - covered)}"
                  f" / 규격 밖: {sorted(covered - EXPECTED_CODES)}")

        if self_pairs:
            print(f"\n부위끼리 접촉한 쌍 {len(self_pairs)}건 — 자가 충돌이 꺼지지 않았습니다:")
            for a, b in sorted(self_pairs)[:6]:
                print(f"  {region_of[a]} ↔ {region_of[b]}")

        if len(good) != len(results) or self_pairs or covered != EXPECTED_CODES:
            print("\n접촉이 부위 단위로 분해되지 않습니다. 이 상태로 데이터를 만들면"
                  " 부위별 접촉 배열을 신뢰할 수 없습니다.")
            exit_code = 1
            return exit_code
        print("\n전 부위가 자기 코드로 분해되었고 자가 충돌도 없습니다. "
              "임계값이 확보되면 같은 구조에 질량·강성만 채우면 됩니다.")
        print("인접 부위가 함께 보고되는 경우가 있으므로, 라벨링은 부위마다 값을 따로 담고"
              " 같은 코드의 프림은 최대값으로 모읍니다.")
        exit_code = 0
        return exit_code
    except Exception:
        # app.close() 가 프로세스를 강제 종료하므로, close() 이전에 잡아 찍지 않으면
        # 예외 traceback 이 조용히 사라진다.
        import traceback
        print("접촉 분해 시험 실패:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        exit_code = 1
        return exit_code
    finally:
        # A bare close() exits the process with 0 before any code after it
        # runs (same hazard documented in check_scene.py / run_s1a.py) — this
        # script's own docstring promises "0 전 부위 분해 성공 / 1 실패한
        # 부위 있음", which does not hold unless the real exit_code is passed
        # through explicitly instead of Kit's own default.
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
