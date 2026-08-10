import math
import os
import pickle
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory


def pack_data(data, dexhand):
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze()
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze()
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class Mano2Dexhand:
    def __init__(self, args, dexhand, obj_urdf_path):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        self.wrist_w = getattr(args, "wrist_w", 1.0)  # wrist 키포인트 loss 가중치 (기본 1)
        self.fix_wrist = bool(getattr(args, "fix_wrist", 0))  # 손목을 타겟에 하드 고정
        self.two_stage = bool(getattr(args, "two_stage", 0))  # 2단계: (1)wrist+비엄지, (2)thumb만
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file)).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 0.0001
            element.rolling_friction = 0.0001
            element.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_rh_props_asset)

        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        default_dof_state["pos"][8] = 0.8
        default_dof_state["pos"][9] = 0.05
        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = False
        asset_options.disable_gravity = True
        asset_options.density = 200

        current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
        for element in rigid_shape_props_asset:
            element.friction = 0.00001
        self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)

        self.envs = []
        self.hand_idxs = []

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            self.obj_actor = self.gym.create_actor(env, current_asset, gymapi.Transform(), "manip_obj", i, 0)

            # 타겟 키포인트 시각화용 sphere(26개/env)는 headless에선 불필요.
            # env당 액터 28→2개로 줄여 startup을 대폭 단축 (loss는 target_joints를 직접 사용).
            if self.headless:
                continue
            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        # headless면 sphere를 안 만들었으므로 빈 리스트 (fitting 루프의 업데이트가 no-op이 됨)
        self.mano_joint_points = (
            []
            if self.headless
            else [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
                for i in range(len(self.dexhand.body_names))
            ]
        )

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )

    def _opt_loop(
        self,
        max_iter,
        weight,
        opti,
        opt_wrist_pos,
        opt_wrist_rot,
        opt_dof_pos,
        target_wrist_pos,
        target_mano_joints,
        obj_trajectory,
    ):
        # 가중 position loss로 Adam 최적화 루프 1회 실행. 최종 (dof_clamped, isaac_joints) 반환.
        # weight가 0인 키포인트에만 영향을 주는 DOF는 grad가 0이 되어 자동으로 고정된다.
        iter = 0
        past_loss = 1e10
        opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
        isaac_joints = None
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])
            self._root_state[:, self.obj_actor, :3] = obj_trajectory[:, :3, 3]
            self._root_state[:, self.obj_actor, 3:7] = rotmat_to_quat(obj_trajectory[:, :3, :3])[:, [1, 2, 3, 0]]

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            # Update jacobian and mass matrix
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            # Step rendering
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            isaac_joints = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )

            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]

            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
            opti.zero_grad()
            loss.backward()
            opti.step()

            if iter % 100 == 0:
                cprint(f"{iter} {loss.item()}", "green")
                if iter > 1 and past_loss - loss.item() < 1e-5:
                    break
                past_loss = loss.item()

        return opt_dof_pos_clamped, isaac_joints

    def fitting(self, max_iter, obj_trajectory, target_wrist_pos, target_wrist_rot, target_mano_joints):

        assert target_mano_joints.shape[0] == self.num_envs
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        target_mano_joints = target_mano_joints.view(-1, 3)
        target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
        target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)

        obj_trajectory = self.mujoco2gym_transf @ obj_trajectory

        middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        obj_pos = obj_trajectory[:, :3, 3]
        offset = middle_pos - obj_pos
        offset = offset / torch.norm(offset, dim=-1, keepdim=True) * 0.2

        # fix_wrist: 손목을 타겟에 하드 고정(offset 없이, grad off)하고 손가락만 최적화
        _wrist_grad = not self.fix_wrist
        opt_wrist_pos = torch.tensor(
            target_wrist_pos if self.fix_wrist else target_wrist_pos + offset,
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=_wrist_grad,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot), device=self.sim_device, dtype=torch.float32, requires_grad=_wrist_grad
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        # 관절 초기값 override. 환경변수로 지정:
        #   DG5FS_INIT_J12=-1.0            (joint_1_2만, 하위호환)
        #   DG5FS_INIT="joint_1_2=-1.0,joint_2_3=0.3,joint_2_4=0.3"  (임의 관절, dof이름=값)
        _init_spec = os.environ.get("DG5FS_INIT")
        _init_j12 = os.environ.get("DG5FS_INIT_J12")
        overrides = {}
        if _init_j12 is not None:
            overrides["joint_1_2"] = float(_init_j12)
        if _init_spec:
            for tok in _init_spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                jn, val = tok.split("=")
                overrides[jn.strip()] = float(val)
        if overrides:
            with torch.no_grad():
                for jn, val in overrides.items():
                    di = self.dexhand.dof_names.index(jn)
                    opt_dof_pos[:, di] = val
            cprint(f"[init] 관절 초기값 override: {overrides}", "yellow")
        if self.fix_wrist:
            cprint("[fix_wrist] 손목을 타겟에 고정, 손가락 DOF만 최적화", "yellow")
            opti = torch.optim.Adam([{"params": [opt_dof_pos], "lr": 0.0004}])
        else:
            opti = torch.optim.Adam(
                [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, {"params": [opt_dof_pos], "lr": 0.0004}]
            )

        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            if "wrist" in k:
                weight.append(self.wrist_w)
            elif "thumb_tip" in k:
                weight.append(4)
            elif "index_tip" in k or "middle_tip" in k:
                weight.append(3)
            else:
                # thumb 몸통 포함 나머지 모든 관절 균등 1
                weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)

        if self.two_stage:
            # thumb 키포인트 마스크 (body_names 순서)
            is_thumb = torch.tensor(
                [("thumb" in self.dexhand.to_hand(k)[0]) for k in self.dexhand.body_names],
                device=self.sim_device,
                dtype=torch.bool,
            )
            w_s1 = weight.clone()
            w_s1[is_thumb] = 0.0  # Stage1: thumb 제외 → thumb DOF는 grad 0이라 자동 고정
            w_s2 = weight.clone()
            w_s2[~is_thumb] = 0.0  # Stage2: thumb만 → 비엄지 DOF는 grad 0이라 자동 고정
            cprint("[two_stage] Stage 1/2: wrist + 비엄지 손가락 최적화", "yellow")
            self._opt_loop(
                max_iter, w_s1, opti, opt_wrist_pos, opt_wrist_rot, opt_dof_pos,
                target_wrist_pos, target_mano_joints, obj_trajectory,
            )
            cprint("[two_stage] Stage 2/2: thumb DOF만 최적화 (wrist 고정)", "yellow")
            opt_wrist_pos.requires_grad_(False)
            opt_wrist_rot.requires_grad_(False)
            opti2 = torch.optim.Adam([{"params": [opt_dof_pos], "lr": 0.0004}])
            opt_dof_pos_clamped, isaac_joints = self._opt_loop(
                max_iter, w_s2, opti2, opt_wrist_pos, opt_wrist_rot, opt_dof_pos,
                target_wrist_pos, target_mano_joints, obj_trajectory,
            )
        else:
            opt_dof_pos_clamped, isaac_joints = self._opt_loop(
                max_iter, weight, opti, opt_wrist_pos, opt_wrist_rot, opt_dof_pos,
                target_wrist_pos, target_mano_joints, obj_trajectory,
            )

        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 4000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "1906",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--wrist_w",
                "type": float,
                "default": 1.0,
            },
            {
                "name": "--fix_wrist",
                "type": int,
                "default": 0,
            },
            {
                "name": "--two_stage",
                "type": int,
                "default": 0,
            },
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, idx):

        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device="cuda:0",
            mujoco2gym_transf=torch.eye(4, device="cuda:0"),
            dexhand=dexhand,
            verbose=False,
        )

        demo_data = pack_data([demo_d[idx]], dexhand)

        parser.num_envs = demo_data["mano_joints"].shape[0]

        mano2inspire = Mano2Dexhand(parser, dexhand, demo_data["obj_urdf_path"][0])

        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["obj_trajectory"],
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            demo_data["mano_joints"].view(parser.num_envs, -1, 3),
        )

        _stage = idx.split("@")[-1]  # 두 자리 stage(@10 등)도 올바로 (idx[-1]은 마지막 글자라 버그)
        if dataset_type == "oakink2":
            dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{_stage}.pkl')}"
        elif dataset_type == "favor":
            dump_path = (
                f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
            )
        elif dataset_type == "grabdemo":
            dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.npy', '.pkl')}"
        elif dataset_type == "oakink2_mirrored":
            dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1].replace('.pkl', f'@{_stage}.pkl')}"
        elif dataset_type == "favor_mirrored":
            dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{os.path.split(demo_data['data_path'][0])[-1]}"
        else:
            raise ValueError("Unsupported dataset type")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)

    run(_parser, _parser.data_idx)
