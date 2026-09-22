"""에피소드별 캡 회전각 + 손가락별 접촉%."""
from isaacgym import gymapi
import csv, sys
import numpy as np
URDF_DIR="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; URDF="rb5_allegro_vmount_siltip2.urdf"
CAPC=np.array([0.4,0.0]); CAPR=0.044; PAD=0.0105
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,URDF_DIR,URDF,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
TIPS=[(i,n) for i,n in enumerate(names) if n.endswith("_tip")]
LBL={"link_3.0_tip":"검지","link_15.0_tip":"엄지","link_7.0_tip":"중지","link_11.0_tip":"약지"}
def episodes(path):
    rows=list(csv.DictReader(open(path))); cur=[]; prev=-1
    for r in rows:
        s=int(float(r["step"]))
        if s<prev and cur: yield cur; cur=[]
        prev=s; cur.append(r)
    if cur: yield cur
for path in sys.argv[1:]:
    print(f"\n=== {path.split('/')[-1]}")
    for k,ep in enumerate(episodes(path)):
        if len(ep)<60: continue
        con=[[] for _ in TIPS]; caps=[]
        for r in ep:
            q=[float(r[f"arm_q_j{j+1}"]) for j in range(6)]+[float(r[f"hand_q_{j}"]) for j in range(16)]
            st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=np.array(q,dtype=np.float32); st["vel"]=0
            gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
            bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
            for t,(i,_) in enumerate(TIPS):
                p=bs["pose"]["p"][i]; rr=np.hypot(p[0]-CAPC[0],p[1]-CAPC[1])
                con[t].append(rr-PAD-CAPR<0.001)
            caps.append(float(r["unscrew_deg"]))
        cc=" ".join(f"{LBL[TIPS[t][1]]}{100*np.mean(con[t]):3.0f}%" for t in range(4))
        print(f"  ep{k}: 캡 {caps[-1]-caps[0]:+6.1f}°  접촉 {cc}")
