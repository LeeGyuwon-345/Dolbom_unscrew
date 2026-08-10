"""allegro 초기 파지자세 -> RB5 6관절 IK (autograd + Adam 다중시드)."""
import json, math
import torch
import pytorch_kinematics as pk
import xml.dom.minidom as minidom

CAP = torch.tensor([0.6, 0.0, 0.225])
pose = json.load(open("/home/leegyuwon/Documents/task1/tools2_allegro/poses/grasp_al_r50.json"))
p_bl = CAP + torch.tensor(pose["wrist_rel_pos"])
x, y, z, w = pose["wrist_quat"]
R_bl = torch.tensor([
    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
cr, sr = 0.0, 1.0
R_m = torch.tensor([[1.,0,0],[0,cr,-sr],[0,sr,cr]])
t_m = torch.tensor([0.0, -0.125, 0.0])
R_tcp = R_bl @ R_m.T
p_tcp = p_bl - R_tcp @ t_m

urdf_path = "/home/leegyuwon/Documents/task1/assets/rb5_850e/rb5_850e.urdf"
chain = pk.build_serial_chain_from_urdf(open(urdf_path,"rb").read(), "tcp")
names = chain.get_joint_parameter_names()
d = minidom.parse(urdf_path)
lim = {}
for j in d.getElementsByTagName("joint"):
    if j.getAttribute("type") == "revolute":
        l = j.getElementsByTagName("limit")[0]
        lim[j.getAttribute("name")] = (float(l.getAttribute("lower")), float(l.getAttribute("upper")))
lo = torch.tensor([lim[n][0] for n in names]); hi = torch.tensor([lim[n][1] for n in names])
print("한계(도):", [(n, round(math.degrees(lim[n][0])), round(math.degrees(lim[n][1]))) for n in names])

torch.manual_seed(0)
best = None
for trial in range(24):
    q0 = lo + (hi-lo)*torch.rand(6)
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=0.08)
    for it in range(600):
        opt.zero_grad()
        m = chain.forward_kinematics(q.unsqueeze(0)).get_matrix()[0]
        loss = ((m[:3,3]-p_tcp)**2).sum()*4.0 + ((m[:3,:3]-R_tcp)**2).sum()*0.5 \
               + ((q-q.clamp(lo,hi))**2).sum()*10.0 + (q**2).sum()*2e-3
        # 중간 링크(팔꿈치·손목)가 바닥 위 15cm 이상 유지
        ms = chain.forward_kinematics(q.unsqueeze(0), end_only=False)
        for lname in ("link3","link4","link5"):
            lz = ms[lname].get_matrix()[0,2,3]
            loss = loss + torch.relu(0.15 - lz)**2 * 50.0
        loss.backward(); opt.step()
    with torch.no_grad():
        qc = q.clamp(lo,hi)
        m = chain.forward_kinematics(qc.unsqueeze(0)).get_matrix()[0]
        e = float((m[:3,3]-p_tcp).norm())
        cosang = (torch.trace(m[:3,:3].T @ R_tcp)-1)/2
        eo = float(torch.arccos(cosang.clamp(-1,1)))
        ms = chain.forward_kinematics(qc.unsqueeze(0), end_only=False)
        minz = min(float(ms[l].get_matrix()[0,2,3]) for l in ("link3","link4","link5"))
        feasible = minz > 0.12
        score = e + 0.1*eo + (0 if feasible else 10)
        if best is None or score < best[0]:
            best = (score, e, eo, minz, qc.clone())
score, e, eo, minz, q = best
print(f"IK 오차: 위치 {e*1000:.2f}mm  자세 {math.degrees(eo):.2f}도  중간링크 최저 z {minz*1000:.0f}mm")
print("관절각(도):", [round(math.degrees(float(v)),2) for v in q])
json.dump({"names": names, "q": [float(v) for v in q],
           "tumbler": [0.6,0,0], "pos_err_mm": e*1000, "rot_err_deg": math.degrees(eo),
           "note": "grasp_al_r50 초기자세 IK"},
          open("arm_ik_grasp.json","w"), indent=1)
print("저장: arm_ik_grasp.json")
