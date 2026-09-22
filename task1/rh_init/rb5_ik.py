"""RB5-850E 6관절 IK -- allegro 파지자세(손목)를 팔로 도달시키는 관절값.

solve_arm_ik.py(assets/rb5_allegro) 를 함수화. 파지자세(캡 기준 손목 pos/quat)를
받아 캡 월드위치·allegro 마운트 오프셋을 반영, RB5 6관절을 Adam 다중시드로 푼다.

캡 월드위치 CAP_WORLD, 마운트(R_m,t_m)는 기존 셋업 캘리브레이션 값 그대로.
"""

from __future__ import annotations

import math

import torch

RB5_URDF = "/home/leegyuwon/Documents/task1/assets/rb5_850e/rb5_850e.urdf"
CAP_WORLD = (0.6, 0.0, 0.225)          # 텀블러 캡 원점의 RB5 베이스 기준 위치(m)
# allegro 마운트: tcp -> 손목(base_link). 실물 마운트(realmount/vmount) 기준:
# hand_mount origin xyz=[0,-0.06,0.095] rpy=Rz(-90도) => "J6축 ⊥ 팜면" (rb5_allegro.py 설계).
#   _R_M = R_tcp<-base = Rz(-90),  _T_M = p_tcp<-base = [0,-0.06,0.095].
# (참고: task1 기본 rb5_allegro.urdf 는 Rx(+90),[0,-0.125,0] 로 J6⊥팜 아님 -- 미사용.)
_R_M = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])   # Rz(-90도)
_T_M = torch.tensor([0.0, -0.060, 0.095])

_CHAIN = None
_NAMES = None
_LO = None
_HI = None


def _init():
    global _CHAIN, _NAMES, _LO, _HI
    if _CHAIN is not None:
        return
    import xml.dom.minidom as minidom

    import pytorch_kinematics as pk
    _CHAIN = pk.build_serial_chain_from_urdf(open(RB5_URDF, "rb").read(), "tcp")
    _NAMES = _CHAIN.get_joint_parameter_names()
    d = minidom.parse(RB5_URDF)
    lim = {}
    for j in d.getElementsByTagName("joint"):
        if j.getAttribute("type") == "revolute":
            l = j.getElementsByTagName("limit")[0]
            lim[j.getAttribute("name")] = (float(l.getAttribute("lower")), float(l.getAttribute("upper")))
    _LO = torch.tensor([lim[n][0] for n in _NAMES])
    _HI = torch.tensor([lim[n][1] for n in _NAMES])


def _quat_to_R(q):
    x, y, z, w = q
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def solve_rb5_ik(wrist_rel_pos, wrist_quat, cap_world=CAP_WORLD, trials=24, iters=600, seed=0,
                 elbow_up=True):
    """파지자세(손목) -> RB5 6관절.

    wrist_rel_pos (3,) 캡기준 손목 위치, wrist_quat (4,) xyzw 월드기준 손목 자세.
    elbow_up=True 면 팔꿈치(J3, elbow)를 위로 접는 IK 가지 선호 (J3>0 유도).
    반환 dict: {names, q(6, rad), q_deg, pos_err_mm, rot_err_deg, feasible}.
    """
    _init()
    wrp = torch.as_tensor(wrist_rel_pos, dtype=torch.float32)
    p_bl = torch.tensor(cap_world, dtype=torch.float32) + wrp
    R_bl = _quat_to_R([float(v) for v in wrist_quat])
    R_tcp = R_bl @ _R_M.T
    p_tcp = p_bl - R_tcp @ _T_M

    torch.manual_seed(seed)
    best = None
    for _ in range(trials):
        q0 = _LO + (_HI - _LO) * torch.rand(6)
        if elbow_up:
            q0[2] = torch.rand(1)[0].abs() * min(float(_HI[2]), 2.4)  # 팔꿈치 양수 시드
        q = q0.clone().requires_grad_(True)
        opt = torch.optim.Adam([q], lr=0.08)
        for _ in range(iters):
            opt.zero_grad()
            m = _CHAIN.forward_kinematics(q.unsqueeze(0)).get_matrix()[0]
            loss = (((m[:3, 3] - p_tcp) ** 2).sum() * 4.0
                    + ((m[:3, :3] - R_tcp) ** 2).sum() * 0.5
                    + ((q - q.clamp(_LO, _HI)) ** 2).sum() * 10.0 + (q ** 2).sum() * 2e-3)
            ms = _CHAIN.forward_kinematics(q.unsqueeze(0), end_only=False)
            for lname in ("link3", "link4", "link5"):
                lz = ms[lname].get_matrix()[0, 2, 3]
                loss = loss + torch.relu(0.15 - lz) ** 2 * 50.0
            if elbow_up:
                # 엘보우(link3) 높이를 손목(tcp) 위로 -- J3 부호가 아니라 링크 높이가
                # 시각적 엘보우업을 결정한다(어깨각에 따라 J3>0 이어도 처질 수 있음).
                l3z = ms["link3"].get_matrix()[0, 2, 3]
                loss = loss + torch.relu(m[2, 3] + 0.05 - l3z) ** 2 * 60.0
            loss.backward(); opt.step()
        with torch.no_grad():
            qc = q.clamp(_LO, _HI)
            m = _CHAIN.forward_kinematics(qc.unsqueeze(0)).get_matrix()[0]
            e = float((m[:3, 3] - p_tcp).norm())
            eo = float(torch.arccos(((torch.trace(m[:3, :3].T @ R_tcp) - 1) / 2).clamp(-1, 1)))
            ms = _CHAIN.forward_kinematics(qc.unsqueeze(0), end_only=False)
            minz = min(float(ms[l].get_matrix()[0, 2, 3]) for l in ("link3", "link4", "link5"))
            feasible = minz > 0.12
            l3z = float(ms["link3"].get_matrix()[0, 2, 3])
            tcpz = float(m[2, 3])
            # 엘보우업 = link3 가 tcp 보다 위. 아니면 선택에서 강한 페널티.
            elbow_pen = (5.0 if (elbow_up and l3z <= tcpz) else 0.0)
            score = e + 0.1 * eo + (0 if feasible else 10) + elbow_pen
            if best is None or score < best[0]:
                best = (score, e, eo, minz, qc.clone())
    _, e, eo, minz, q = best
    return {
        "names": _NAMES,
        "q": [float(v) for v in q],
        "q_deg": [round(math.degrees(float(v)), 2) for v in q],
        "pos_err_mm": round(e * 1000, 2),
        "rot_err_deg": round(math.degrees(eo), 2),
        "feasible": bool(minz > 0.12),
    }


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", help="파지자세 json (wrist_rel_pos, wrist_quat)")
    ap.add_argument("--cap", type=float, nargs=3, default=list(CAP_WORLD))
    a = ap.parse_args()
    d = json.load(open(a.pose))
    res = solve_rb5_ik(d["wrist_rel_pos"], d["wrist_quat"], cap_world=tuple(a.cap))
    print("RB5 관절(도):", dict(zip(res["names"], res["q_deg"])))
    print(f"IK 오차: 위치 {res['pos_err_mm']}mm 자세 {res['rot_err_deg']}도 도달가능 {res['feasible']}")
