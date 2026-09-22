"""팁별 접촉 위치 분류: 캡 벽(z 215~231) / 몸통(z<215) / 상부이탈 / 비접촉. CSV+FK."""
from isaacgym import gymapi
import csv, sys, math
import numpy as np
URDF_DIR="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; URDF="rb5_allegro_vmount_siltip2.urdf"
CAPC=np.array([0.4,0.0]); CAPR=0.044; PAD=0.0115; PITCH=3.18309886e-03
CAP_LO=0.215; CAP_HI=0.231
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,URDF_DIR,URDF,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
TIPS=[(i,n) for i,n in enumerate(names) if n.endswith("_tip")]
LBL={"link_3.0_tip":"검지","link_15.0_tip":"엄지","link_7.0_tip":"중지","link_11.0_tip":"약지"}
rows=list(csv.DictReader(open(sys.argv[1])))
eps=[];cur=[];prev=-1
for r in rows:
    s=int(float(r["step"]))
    if s<prev and cur: eps.append(cur); cur=[]
    prev=s; cur.append(r)
if cur: eps.append(cur)
stat={n:{"cap":0,"body":0,"above":0,"off":0,"zs":[]} for _,n in TIPS if n in LBL}
tot=0
for ep in eps:
    if len(ep)<300: continue
    for r in ep:
        q=[float(r[f"arm_q_j{j+1}"]) for j in range(6)]+[float(r[f"hand_q_{j}"]) for j in range(16)]
        st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=np.array(q,dtype=np.float32); st["vel"]=0
        gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
        bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
        lift=PITCH*max(0.0,math.radians(float(r["unscrew_deg"])))
        tot+=1
        for bi,nm in TIPS:
            if nm not in LBL: continue
            p=bs["pose"]["p"][bi]; rr=math.hypot(p[0]-CAPC[0],p[1]-CAPC[1])
            near=(rr-PAD-CAPR)<0.001
            d=stat[nm]
            if not near: d["off"]+=1; continue
            z=p[2]
            if CAP_LO+lift-0.002<=z<=CAP_HI+lift+0.002: d["cap"]+=1; d["zs"].append(z-lift)
            elif z<CAP_LO+lift-0.002: d["body"]+=1
            else: d["above"]+=1
print(f"스텝 {tot} (캡 상승 보정 포함)")
for nm,d in stat.items():
    zs=np.array(d["zs"]) if d["zs"] else np.array([0.0])
    print(f"  {LBL[nm]}: 캡벽 {100*d['cap']/tot:5.1f}%  몸통 {100*d['body']/tot:4.1f}%  상부이탈 {100*d['above']/tot:4.1f}%  비접촉 {100*d['off']/tot:4.1f}%   캡접촉 z 평균 {zs.mean()*1000:.1f}mm")
