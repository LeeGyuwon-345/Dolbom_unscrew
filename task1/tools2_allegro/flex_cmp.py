"""폄 정도·손바닥 높이 비교: 손가락 굴곡합(도) + palm/너클의 캡윗면(231mm) 대비 z."""
import json, math, sys
import numpy as np
from isaacgym import gymapi
RD="/home/leegyuwon/Documents/task1/assets/rb5_allegro"; RU="rb5_allegro_vmount_siltip2.urdf"
gym=gymapi.acquire_gym(); sp=gymapi.SimParams(); sp.up_axis=gymapi.UP_AXIS_Z; sp.gravity=gymapi.Vec3(0,0,0)
sim=gym.create_sim(0,0,gymapi.SIM_PHYSX,sp)
o=gymapi.AssetOptions(); o.fix_base_link=True; o.disable_gravity=True; o.collapse_fixed_joints=False
asset=gym.load_asset(sim,RD,RU,o)
env=gym.create_env(sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
act=gym.create_actor(env,asset,gymapi.Transform(),"r",0,0)
names=gym.get_actor_rigid_body_names(env,act)
KN=[n for n in names if n in ("link_0.0","link_4.0","link_8.0","link_12.0")]  # 너클(MCP 링크)
CAP_TOP=0.231
for path in sys.argv[1:]:
    q=np.array(json.load(open(path))['dof_pos'],dtype=np.float32)
    st=gym.get_actor_dof_states(env,act,gymapi.STATE_ALL); st["pos"]=q; st["vel"]=0
    gym.set_actor_dof_states(env,act,st,gymapi.STATE_ALL); gym.simulate(sim); gym.fetch_results(sim,True)
    bs=gym.get_actor_rigid_body_states(env,act,gymapi.STATE_POS)
    palm=bs["pose"]["p"][names.index("base_link")]
    knz=[float(bs["pose"]["p"][names.index(n)][2]) for n in KN]
    # 굴곡합: 각 손가락의 굴곡 관절(J1..J3; J0 는 abduction) 절대합
    grp={'검지':(7,8,9),'중지':(11,12,13),'약지':(15,16,17),'엄지':(19,20,21)}
    fx={g:sum(abs(math.degrees(q[i])) for i in idx) for g,idx in grp.items()}
    print(f"{path.split('/')[-1]}")
    print(f"  굴곡합(도): " + "  ".join(f"{g} {v:.0f}" for g,v in fx.items()) + f"   합계 {sum(fx.values()):.0f}")
    print(f"  palm z: {float(palm[2])*1000:.1f}mm (캡윗면 대비 {1000*(float(palm[2])-CAP_TOP):+.1f})   너클 z 평균: {1000*np.mean(knz):.1f}mm (캡윗면 대비 {1000*(np.mean(knz)-CAP_TOP):+.1f})")
