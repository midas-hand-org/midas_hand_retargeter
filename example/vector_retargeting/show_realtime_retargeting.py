import multiprocessing
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

# Prefer local source tree over site-packages when running from this repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_SRC = REPO_ROOT / "src"
if LOCAL_SRC.exists():
    local_src_str = str(LOCAL_SRC)
    if local_src_str not in sys.path:
        sys.path.insert(0, local_src_str)

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

# ── Debug constants ──────────────────────────────────────────────────────────
_FINGER_LABELS = ["엄지(Thumb)", "검지(Index)", "중지(Middle)", "약지(Ring)"]
_TASK_LINK_NAMES = ["thumb_dip", "dip_link_2", "dip_link_1", "dip_link"]
_ORIGIN_LINK_NAME = "arm_interface"
_HUMAN_KP_INDICES = [4, 8, 12, 16]
_ASSEMBLY1_PIP_DIP_COUPLING = [
    ("revolute_2_0", "revolute_1_0"),
    ("revolute_2_1", "revolute_1_1"),
    ("revolute_2_2", "revolute_1_2"),
]
_ASSEMBLY1_DIP_GAIN = 1.0
_LOOKUP_CANDIDATE_PATHS = [
    Path("/home/beom/midas/midas_hand_mujoco-norman_dev/midas_hand_mujoco-norman_dev/assets/pip_dip_linkage_lookup.csv"),
    Path(__file__).resolve().parents[2] / "assets" / "pip_dip_linkage_lookup.csv",
]

# Per-finger joint groups (name → joints to display)
_FINGER_JOINT_GROUPS = {
    "thumb  (→thumb_dip )": ["revolute_1_3", "revolute_2_3", "revolute_3_3", "revolute_4_3"],
    "finger2(→dip_link_2)": ["revolute_3_2", "revolute_4_2", "revolute_2_2"],
    "finger1(→dip_link_1)": ["revolute_3_1", "revolute_4_1", "revolute_2_1"],
    "finger0(→dip_link  )": ["revolute_3_0", "revolute_4_0", "revolute_2_0"],
}


def _load_pip_dip_lookup() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Path]]:
    for path in _LOOKUP_CANDIDATE_PATHS:
        if path.exists():
            data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] < 2:
                logger.warning("PIP-DIP lookup format is invalid: {}", path)
                return None, None, None
            q_pip = data[:, 0]
            q_dip = data[:, 1]
            order = np.argsort(q_pip)
            return q_pip[order], q_dip[order], path
    logger.warning("PIP-DIP lookup csv not found. Candidates: {}", _LOOKUP_CANDIDATE_PATHS)
    return None, None, None


def _apply_pip_dip_coupling(
    qpos: np.ndarray,
    q_pip_table: np.ndarray,
    q_dip_table: np.ndarray,
    joint_idx: Dict[str, int],
    dip_limits: Dict[str, Tuple[float, float]],
) -> List[Tuple[str, float, str, float]]:
    debug_pairs: List[Tuple[str, float, str, float]] = []
    for pip_name, dip_name in _ASSEMBLY1_PIP_DIP_COUPLING:
        if pip_name not in joint_idx or dip_name not in joint_idx:
            continue
        pip_idx = joint_idx[pip_name]
        dip_idx = joint_idx[dip_name]
        # Linkage csv is defined by flexion magnitude.
        pip_q = float(np.clip(abs(qpos[pip_idx]), q_pip_table[0], q_pip_table[-1]))
        dip_q = float(np.interp(pip_q, q_pip_table, q_dip_table))
        dip_q *= _ASSEMBLY1_DIP_GAIN
        if dip_name in dip_limits:
            lo, hi = dip_limits[dip_name]
            dip_q = float(np.clip(dip_q, lo, hi))
        qpos[dip_idx] = dip_q
        debug_pairs.append((pip_name, float(qpos[pip_idx]), dip_name, dip_q))
    return debug_pairs


def _draw_debug_overlay(
    bgr: np.ndarray,
    hand_detected: bool,
    human_vecs: Optional[np.ndarray],
    robot_vecs: Optional[np.ndarray],
    scaling: float,
    opt_loss: float,
    debug_mode: bool,
) -> np.ndarray:
    overlay = bgr.copy()

    if hand_detected:
        cv2.putText(overlay, "HAND: DETECTED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    else:
        cv2.putText(overlay, "HAND: NOT DETECTED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

    hint = "[d] debug terminal ON" if not debug_mode else "[d] debug terminal OFF"
    cv2.putText(overlay, hint, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    if opt_loss >= 0:
        loss_text = f"opt_loss={opt_loss:.5f}"
        loss_color = (0, 210, 0) if opt_loss < 1e-3 else (0, 165, 255) if opt_loss < 1e-2 else (0, 0, 220)
        cv2.putText(overlay, loss_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, loss_color, 1)

    if human_vecs is not None and robot_vecs is not None:
        # Compare robot_vec vs scaled human_vec (matches what optimizer does internally)
        scaled_human = human_vecs * scaling
        short_labels = ["Thumb", "Index", "Middle", "Ring"]
        for i, label in enumerate(short_labels):
            shv = scaled_human[i]
            rv = robot_vecs[i]
            sh_norm = float(np.linalg.norm(shv))
            r_norm = float(np.linalg.norm(rv))
            err = float(np.linalg.norm(shv - rv))
            pct = err / (sh_norm + 1e-6) * 100

            # angle error (direction)
            sh_dir = shv / (sh_norm + 1e-6)
            r_dir = rv / (r_norm + 1e-6)
            cos = float(np.clip(np.dot(sh_dir, r_dir), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cos)))

            text = f"{label}: target={sh_norm:.3f}m robot={r_norm:.3f}m err={err:.3f}m({pct:.0f}%) ang={ang:.1f}deg"
            y = 88 + i * 20
            color = (0, 210, 0) if pct < 5 else (0, 165, 255) if pct < 15 else (0, 0, 220)
            cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    return overlay


def _print_debug_snapshot(
    joint_pos: np.ndarray,
    human_vecs: np.ndarray,
    robot_vecs: np.ndarray,
    full_qpos: np.ndarray,
    joint_names: list,
    joint_limit_map: Dict[str, Tuple[float, float]],
    scaling: float,
    opt_loss: float,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DEBUG SNAPSHOT")
    print(sep)

    # ── Stage 1: MediaPipe 키포인트 ──────────────────────────────────────────
    print("\n[Stage 1] MediaPipe 키포인트 (MANO 좌표계, 손목 기준 상대좌표)")
    kp_info = {
        0:  "손목(wrist)     ",
        4:  "엄지끝(thumb_tip)",
        8:  "검지끝(index_tip)",
        12: "중지끝(mid_tip)  ",
        16: "약지끝(ring_tip) ",
        20: "소지끝(pinky_tip)",
    }
    for kp_idx, name in kp_info.items():
        p = joint_pos[kp_idx]
        print(f"  kp{kp_idx:02d} {name}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    # ── Stage 2: Human 벡터 (raw & scaled) ──────────────────────────────────
    print(f"\n[Stage 2] Human Vectors  (scaling_factor={scaling})")
    print(f"  {'손가락':<10} {'robot_link':<14} {'raw_norm':<10} {'scaled_norm':<12} {'scaled_방향 [x, y, z]'}")
    print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*12} {'-'*24}")
    scaled_human = human_vecs * scaling
    for i in range(4):
        hv = human_vecs[i]
        shv = scaled_human[i]
        raw_norm = float(np.linalg.norm(hv))
        scaled_norm = float(np.linalg.norm(shv))
        direction = shv / (scaled_norm + 1e-6)
        print(f"  {_FINGER_LABELS[i]:<10} {_TASK_LINK_NAMES[i]:<14} {raw_norm:<10.4f} {scaled_norm:<12.4f} "
              f"[{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}]")

    # ── Stage 3: 로봇 조인트 ─────────────────────────────────────────────────
    print("\n[Stage 3] 로봇 조인트 각도 (degrees)")
    for finger_label, jnames in _FINGER_JOINT_GROUPS.items():
        parts = []
        for jname in jnames:
            if jname in joint_names:
                idx = joint_names.index(jname)
                parts.append(f"{jname}={np.degrees(full_qpos[idx]):+6.1f}°")
            else:
                parts.append(f"{jname}=N/A")
        print(f"  {finger_label}: " + "  ".join(parts))

    # Thumb distal joint specific trace
    thumb_joint = "revolute_4_3"
    if thumb_joint in joint_names:
        idx = joint_names.index(thumb_joint)
        q = float(full_qpos[idx])
        if thumb_joint in joint_limit_map:
            lo, hi = joint_limit_map[thumb_joint]
            span = max(hi - lo, 1e-6)
            ratio = (q - lo) / span
            print(
                f"  [thumb distal] {thumb_joint}={np.degrees(q):+.2f}° "
                f"(rad={q:+.4f}, limit=[{lo:+.4f}, {hi:+.4f}], ratio={ratio:.3f})"
            )
        else:
            print(f"  [thumb distal] {thumb_joint}={np.degrees(q):+.2f}° (rad={q:+.4f})")

    # ── Stage 4: 벡터 잔차 (scaled human vs robot) ───────────────────────────
    print(f"\n[Stage 4] 벡터 잔차 (scaled human vs robot)  opt_loss={opt_loss:.6f}")
    print(f"  ※ optimizer는 내부적으로 human*{scaling} vs robot을 최소화함")
    print(f"  {'손가락':<10} {'robot_link':<14} {'target(m)':<11} {'robot(m)':<11} {'err(m)':<8} {'%':<7} {'판정'}")
    print(f"  {'-'*10} {'-'*14} {'-'*11} {'-'*11} {'-'*8} {'-'*7} {'-'*8}")
    total_err = 0.0
    for i in range(4):
        shv = scaled_human[i]
        rv = robot_vecs[i]
        sh_norm = float(np.linalg.norm(shv))
        r_norm = float(np.linalg.norm(rv))
        err = float(np.linalg.norm(shv - rv))
        pct = err / (sh_norm + 1e-6) * 100
        total_err += err
        flag = "OK" if pct < 5 else "주의" if pct < 15 else "!!크다!!"
        print(f"  {_FINGER_LABELS[i]:<10} {_TASK_LINK_NAMES[i]:<14} "
              f"{sh_norm:<11.4f} {r_norm:<11.4f} {err:<8.4f} {pct:<7.1f} {flag}")
    print(f"  총 잔차: {total_err:.4f}m")

    # ── Stage 5: 방향 오차 (크기 무관, 좌표계 문제 탐지) ─────────────────────
    print("\n[Stage 5] 방향 오차  (크기 무시, 좌표계 미스매치 탐지)")
    print(f"  ※ 각도오차 >30° 이면 MANO↔robot 좌표계 불일치 가능성")
    all_large_angle = True
    for i in range(4):
        shv = scaled_human[i]
        rv = robot_vecs[i]
        sh_dir = shv / (np.linalg.norm(shv) + 1e-6)
        r_dir = rv / (np.linalg.norm(rv) + 1e-6)
        cos_sim = float(np.clip(np.dot(sh_dir, r_dir), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))
        if angle_deg < 30:
            all_large_angle = False
        flag = "OK" if angle_deg < 15 else "주의" if angle_deg < 30 else "!! 좌표계?"
        print(f"  {_FINGER_LABELS[i]:<10} → {_TASK_LINK_NAMES[i]:<14}: "
              f"cos={cos_sim:.4f}  각도오차={angle_deg:.2f}°  {flag}")
    if all_large_angle:
        print("  >> 전체 손가락 방향오차 30°+ : MANO/robot 좌표계 불일치 강력 의심!")

    # ── Stage 6: 크기만 비교 (조인트리밋/DIP 문제 탐지) ─────────────────────
    print("\n[Stage 6] 크기(norm) 비교  (관절한계/DIP 고정 문제 탐지)")
    print(f"  ※ target > robot 이면 로봇이 충분히 못뻗음 (조인트리밋 or DIP=0)")
    for i in range(4):
        shv = scaled_human[i]
        rv = robot_vecs[i]
        t = float(np.linalg.norm(shv))
        r = float(np.linalg.norm(rv))
        ratio = r / (t + 1e-6)
        flag = "OK" if 0.9 < ratio < 1.1 else "짧음(DIP?)" if ratio < 0.9 else "김"
        print(f"  {_FINGER_LABELS[i]:<10} → {_TASK_LINK_NAMES[i]:<14}: "
              f"target={t:.4f}m  robot={r:.4f}m  ratio={ratio:.3f}  {flag}")

    print(sep + "\n")


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retargeting.set_qpos(np.zeros(retargeting.optimizer.robot.dof, dtype=np.float32))

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)
    logger.info("Retarget config urdf_path: {}", config.urdf_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5
    elif "assembly_1" in robot_name:
        loader.scale = 1.4

    if "glb" not in robot_name:
        glb_candidate = Path(str(filepath).replace(".urdf", "_glb.urdf"))
        if glb_candidate.exists():
            filepath = str(glb_candidate)
        else:
            logger.warning("GLB urdf not found ({}). Falling back to {}", glb_candidate, filepath)
            filepath = str(filepath)
    else:
        filepath = str(filepath)
    logger.info("Viewer urdf path: {}", filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "assembly_1" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    robot_obj = retargeting.optimizer.robot
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
    joint_name_to_idx = {name: i for i, name in enumerate(retargeting_joint_names)}
    joint_limit_map: Dict[str, Tuple[float, float]] = {
        name: tuple(robot_obj.joint_limits[robot_obj.get_joint_index(name)])
        for name in retargeting_joint_names
        if name in robot_obj.dof_joint_names
    }
    dip_limits = {
        name: tuple(robot_obj.joint_limits[robot_obj.get_joint_index(name)])
        for _, name in _ASSEMBLY1_PIP_DIP_COUPLING
        if name in joint_name_to_idx
    }

    coupling_enabled = "assembly_1" in robot_name
    q_pip_table, q_dip_table, lookup_path = _load_pip_dip_lookup() if coupling_enabled else (None, None, None)
    if coupling_enabled and q_pip_table is not None and q_dip_table is not None:
        logger.info("Enabled PIP-DIP lookup coupling for {} using {}", robot_name, lookup_path)
    elif coupling_enabled:
        logger.warning("PIP-DIP coupling requested but lookup is unavailable; DIP will remain uncoupled.")

    num_fixed = len(retargeting.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros(num_fixed, dtype=np.float32)
    if num_fixed > 0:
        logger.info(
            "Using {} fixed joints: {}",
            num_fixed,
            retargeting.optimizer.fixed_joint_names,
        )

    # Pre-cache pinocchio link indices for robot-side vector computation
    try:
        origin_link_idx = robot_obj.get_link_index(_ORIGIN_LINK_NAME)
        task_link_indices = [robot_obj.get_link_index(n) for n in _TASK_LINK_NAMES]
        can_compute_robot_vecs = True
    except ValueError as e:
        logger.warning("Debug FK cache failed ({}). Robot vector overlay disabled.", e)
        can_compute_robot_vecs = False

    # Pre-cache scaling factor for debug
    scaling_factor = float(getattr(retargeting.optimizer, "scaling", 1.0))
    logger.info("Optimizer scaling_factor = {}", scaling_factor)

    # ── Debug state ──────────────────────────────────────────────────────────
    debug_mode = False
    last_debug_print_t = 0.0
    last_opt_loss = -1.0
    logger.info("키 조작: [d] 디버그 터미널 토글  [q] 종료")

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        hand_detected = joint_pos is not None

        if keypoint_2d is not None:
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")

        human_vecs: Optional[np.ndarray] = None
        robot_vecs: Optional[np.ndarray] = None
        coupling_debug_pairs: List[Tuple[str, float, str, float]] = []

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            human_vecs = ref_value.copy()

            if not np.isfinite(ref_value).all():
                logger.warning("Skip frame due to non-finite hand keypoints")
                continue

            try:
                qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
                if not np.isfinite(qpos).all():
                    logger.warning("Skip frame due to non-finite qpos")
                    continue

                if coupling_enabled and q_pip_table is not None and q_dip_table is not None:
                    coupling_debug_pairs = _apply_pip_dip_coupling(
                        qpos,
                        q_pip_table,
                        q_dip_table,
                        joint_name_to_idx,
                        dip_limits,
                    )

                # Read optimizer convergence value
                try:
                    last_opt_loss = float(retargeting.optimizer.opt.last_optimum_value())
                except Exception:
                    last_opt_loss = -1.0

                # ── Robot-side vector: use UNFILTERED optimizer result ────────
                # retargeting.last_qpos = optimizer output BEFORE LP filter
                # This directly reflects what the optimizer converged on and
                # should numerically match scaled_human_vec if opt_loss is low.
                if can_compute_robot_vecs:
                    opt_qpos = np.zeros(robot_obj.dof, dtype=np.float32)
                    opt_qpos[retargeting.optimizer.idx_pin2fixed] = fixed_qpos
                    opt_qpos[retargeting.optimizer.idx_pin2target] = retargeting.last_qpos
                    if coupling_enabled and q_pip_table is not None and q_dip_table is not None:
                        _apply_pip_dip_coupling(
                            opt_qpos,
                            q_pip_table,
                            q_dip_table,
                            joint_name_to_idx,
                            dip_limits,
                        )
                    robot_obj.compute_forward_kinematics(opt_qpos)
                    origin_pos = robot_obj.get_link_pose(origin_link_idx)[:3, 3]
                    robot_vecs = np.array([
                        robot_obj.get_link_pose(idx)[:3, 3] - origin_pos
                        for idx in task_link_indices
                    ])

                # Debug terminal print (rate-limited to 1 Hz)
                now_t = time.perf_counter()
                if debug_mode and can_compute_robot_vecs and (now_t - last_debug_print_t) >= 1.0:
                    last_debug_print_t = now_t
                    _print_debug_snapshot(
                        joint_pos, human_vecs, robot_vecs, opt_qpos,
                        retargeting_joint_names, joint_limit_map, scaling_factor, last_opt_loss,
                    )
                    if coupling_debug_pairs:
                        print("[PIP->DIP coupling]")
                        for pip_name, pip_q, dip_name, dip_q in coupling_debug_pairs:
                            print(f"  {pip_name}={pip_q:+.4f} -> {dip_name}={dip_q:+.4f}")

                robot.set_qpos(qpos[retargeting_to_sapien])
            except Exception:
                logger.exception("Retargeting failed for current frame; skipping")
                continue

        # Draw overlay onto camera window
        frame_display = _draw_debug_overlay(
            bgr, hand_detected, human_vecs, robot_vecs, scaling_factor, last_opt_loss, debug_mode
        )
        cv2.imshow("realtime_retargeting_demo", frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
            logger.info("Debug mode: {}", "ON — 터미널에서 1초마다 스냅샷 출력" if debug_mode else "OFF")

        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        import re
        m = re.search(r'(\d+)$', camera_path)
        cam_index = int(m.group(1)) if m else 0
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
