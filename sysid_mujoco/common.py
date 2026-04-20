from __future__ import annotations

import importlib.util
import json
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


SYSID_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SYSID_DIR / "generated"


@dataclass
class ProcessedTrajectory:
    source_path: Path
    sequence_name: str
    times: np.ndarray
    measured_qpos: np.ndarray
    measured_qvel: np.ndarray
    desired_qpos: np.ndarray
    ctrl: np.ndarray
    joint_names: list[str]
    actuator_names: list[str]
    kp: np.ndarray
    kd: np.ndarray


def _stub_colorama() -> None:
    if importlib.util.find_spec("colorama") is not None:
        return

    class _Ansi:
        def __getattr__(self, _name: str) -> str:
            return ""

    module = types.ModuleType("colorama")
    module.Fore = _Ansi()
    module.Style = _Ansi()
    module.init = lambda *args, **kwargs: None
    sys.modules.setdefault("colorama", module)


def _stub_tabulate() -> None:
    if importlib.util.find_spec("tabulate") is not None:
        return

    module = types.ModuleType("tabulate")

    def tabulate(
        rows: list[list[Any]],
        headers: tuple[str, ...] | list[str] = (),
        tablefmt: str | None = None,
        disable_numparse: bool = False,
    ) -> str:
        del tablefmt
        del disable_numparse
        lines: list[str] = []
        if headers:
            lines.append(" | ".join(str(header) for header in headers))
        for row in rows:
            lines.append(" | ".join(str(value) for value in row))
        return "\n".join(lines)

    module.tabulate = tabulate
    sys.modules.setdefault("tabulate", module)


def _stub_default_report() -> None:
    if importlib.util.find_spec("plotly") is not None:
        return

    report_package = types.ModuleType("mujoco.sysid.report")
    defaults_module = types.ModuleType("mujoco.sysid.report.defaults")

    def default_report(*args, **kwargs):
        del args
        del kwargs
        raise RuntimeError(
            "Plotly is not installed in the active environment, so "
            "mujoco.sysid.default_report is unavailable."
        )

    defaults_module.default_report = default_report
    sys.modules.setdefault("mujoco.sysid.report", report_package)
    sys.modules.setdefault("mujoco.sysid.report.defaults", defaults_module)


def import_mujoco():
    import mujoco

    if not hasattr(mujoco, "MjModel"):
        raise RuntimeError(
            "The active Python interpreter does not provide the DeepMind "
            "MuJoCo API. Run these scripts with "
            "`micromamba run -n mpx_env python ...`."
        )
    return mujoco


def import_mujoco_sysid():
    import mujoco
    import mujoco.sysid as sysid

    return mujoco, sysid


def load_torch_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    import torch

    try:
        raw_data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw_data = torch.load(dataset_path, map_location="cpu")

    dataset: dict[str, np.ndarray] = {}
    for key, value in raw_data.items():
        if torch.is_tensor(value):
            dataset[key] = value.detach().cpu().numpy()
        else:
            dataset[key] = np.asarray(value)
    return dataset


def _normalize_per_joint_values(
    values: Any,
    num_joints: int,
) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).reshape(-1)
    if normalized.size == 1:
        return np.full(num_joints, normalized[0], dtype=np.float64)
    if normalized.size != num_joints:
        raise ValueError(
            f"Expected actuator gains to have size 1 or {num_joints}, "
            f"got {normalized.size}."
        )
    return normalized

def load_dataset_actuator_gains(
    dataset_path: Path,
    num_joints: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_torch_dataset(dataset_path)
    if num_joints is None:
        dof_pos = np.asarray(dataset["dof_pos"], dtype=np.float64)
        num_joints = int(dof_pos.shape[1])

    if 'Kp' not in dataset:
        print(f"{dataset_path} does not contain `Kp` switching to config default")
    if 'Kd' not in dataset:
        print(f"{dataset_path} does not contain `Kd` switching to config default")

        kp = dataset['Kp'] if 'Kp' in dataset else np.full(num_joints, config.Kp, dtype=np.float64)  
        kd = dataset['Kd'] if 'Kd' in dataset else np.full(num_joints, config.Kd, dtype=np.float64)
    
    return kp, kd


def build_actuator_gain_map(
    joint_names: list[str],
    kp: np.ndarray,
    kd: np.ndarray,
) -> dict[str, tuple[float, float]]:
    if len(joint_names) != len(kp) or len(joint_names) != len(kd):
        raise ValueError(
            "Joint names and actuator gains must have the same length."
        )
    return {
        joint_name: (float(joint_kp), float(joint_kd))
        for joint_name, joint_kp, joint_kd in zip(joint_names, kp, kd, strict=True)
    }

def get_robot_scene_path(robot: str) -> Path:
    scene_path = REPO_ROOT / "robot_model" / robot / "scene_flat.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"Could not find scene file: {scene_path}")
    return scene_path


def get_robot_model_xml_path(robot: str) -> Path:
    scene_path = get_robot_scene_path(robot)
    scene_tree = ET.parse(scene_path)
    scene_root = scene_tree.getroot()
    include_element = scene_root.find("include")
    if include_element is None:
        raise RuntimeError(f"No <include> tag found in scene file {scene_path}.")
    include_path = include_element.get("file")
    if include_path is None:
        raise RuntimeError(f"The <include> tag in {scene_path} has no `file` attribute.")
    return (scene_path.parent / include_path).resolve()


def _remove_all_by_tag(root: ET.Element, tag: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)

def _absolutize_file_attributes(root: ET.Element, base_dir: Path) -> None:
    for element in root.iter():
        file_attribute = element.get("file")
        if file_attribute is None:
            continue
        file_path = Path(file_attribute)
        if file_path.is_absolute():
            continue
        element.set("file", str((base_dir / file_path).resolve()))


def _rewrite_actuators_as_general(
    root: ET.Element,
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> None:
    actuator_element = root.find("actuator")
    source_actuators: list[dict[str, str]] = []
    if actuator_element is not None:
        for actuator in actuator_element:
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue
            actuator_spec = {
                "joint": joint_name,
                "gear": "1",
            }
            actuator_name = actuator.get("name")
            if actuator_name is not None:
                actuator_spec["name"] = actuator_name
            # force_range = actuator.get("forcerange") or actuator.get("ctrlrange")
            # if force_range is not None:
            #     actuator_spec["forcerange"] = force_range
            source_actuators.append(actuator_spec)

    _remove_all_by_tag(root, "motor")

    joint_ranges: dict[str, str] = {}
    for element in root.iter("joint"):
        joint_name = element.get("name")
        joint_range = element.get("range")
        if joint_name is not None and joint_range is not None:
            joint_ranges[joint_name] = joint_range

    if actuator_element is None:
        actuator_element = ET.SubElement(root, "actuator")
    else:
        for child in list(actuator_element):
            actuator_element.remove(child)

    for actuator_spec in source_actuators:
        joint_name = actuator_spec["joint"]
        joint_range = joint_ranges.get(joint_name)
        if joint_range is not None:
            actuator_spec["ctrlrange"] = joint_range
        if actuator_gains is not None and joint_name in actuator_gains:
            kp, kd = actuator_gains[joint_name]
            actuator_spec["biastype"] = "affine"
            actuator_spec["gainprm"] = f"{kp:.12g}"
            actuator_spec["biasprm"] = f"0 {-kp:.12g} {-kd:.12g}"
        ET.SubElement(actuator_element, "general", actuator_spec)


def _disable_all_collisions(root):
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def build_fixed_base_model_xml(
    robot: str,
    actuator_mode: str = "general",
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> Path:
    source_xml = get_robot_model_xml_path(robot)
    output_dir = GENERATED_DIR / robot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / f"{robot}_fixed_base_sysid.xml"

    tree = ET.parse(source_xml)
    root = tree.getroot()
    _remove_all_by_tag(root, "freejoint")
    _remove_all_by_tag(root, "keyframe")
    _remove_all_by_tag(root, "accelerometer")
    _remove_all_by_tag(root, "gyro")
    _remove_all_by_tag(root, "framepos")
    _remove_all_by_tag(root, "framequat")
    if actuator_mode == "general":
        _rewrite_actuators_as_general(root, actuator_gains=actuator_gains)
    elif actuator_mode != "motor":
        raise ValueError(
            f"Unsupported actuator mode `{actuator_mode}`. "
            "Expected `motor` or `general`."
        )
    _absolutize_file_attributes(root, source_xml.parent)

    _disable_all_collisions(root)

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def get_actuated_joint_and_actuator_names(mujoco, model) -> tuple[list[str], list[str]]:
    joint_names: list[str] = []
    actuator_names: list[str] = []
    for actuator_id in range(model.nu):
        actuator_names.append(model.actuator(actuator_id).name)
        joint_id = int(model.actuator_trnid[actuator_id][0])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        joint_names.append(joint_name)
    return joint_names, actuator_names


def build_measurement_names(joint_names: list[str]) -> list[str]:
    return [f"{joint_name}_qpos" for joint_name in joint_names] + [
        f"{joint_name}_qvel" for joint_name in joint_names
    ]


def compute_pd_torques(
    desired_qpos: np.ndarray,
    desired_qvel: np.ndarray,
    measured_qpos: np.ndarray,
    measured_qvel: np.ndarray,
    kp: float,
    kd: float,
    ctrlrange: np.ndarray | None = None,
) -> np.ndarray:
    ctrl = kp * (desired_qpos - measured_qpos) - kd * (desired_qvel - measured_qvel)
    if ctrlrange is None:
        return ctrl
    lower = ctrlrange[:, 0]
    upper = ctrlrange[:, 1]
    return np.clip(ctrl, lower, upper)


def load_processed_dataset(
    dataset_path: Path,
    model,
    mujoco,
    actuator_mode: str = "general",
) -> ProcessedTrajectory:
    dataset = load_torch_dataset(dataset_path)
    if "time" not in dataset:
        raise KeyError(f"{dataset_path} must contain `time`.")
    elif "dof_pos" not in dataset:
        raise KeyError(f"{dataset_path} must contain `dof_pos`.")
    elif "des_dof_pos" not in dataset:
        raise KeyError(f"{dataset_path} must contain `des_dof_pos`.")
    elif "dof_vel" not in dataset:
        raise KeyError(f"{dataset_path} must contain `dof_vel`.")
    elif "des_dof_vel" not in dataset:
        raise KeyError(f"{dataset_path} must contain `des_dof_vel`.")

    times = np.asarray(dataset["time"], dtype=np.float64)
    measured_qpos = np.asarray(dataset["dof_pos"], dtype=np.float64)
    measured_qvel = np.asarray(dataset["dof_vel"], dtype=np.float64)
    #measured_qvel = np.zeros_like(measured_qpos) # Use placeholder if you don't have velocity measurements in the dataset
    desired_qpos = np.asarray(dataset["des_dof_pos"], dtype=np.float64)
    desired_qvel = np.asarray(dataset["des_dof_vel"], dtype=np.float64)
    #desired_qvel = np.zeros_like(desired_qpos) # Use placeholder if you don't have velocity measurements in the dataset

    joint_names, actuator_names = get_actuated_joint_and_actuator_names(mujoco, model)
    kp, kd = load_dataset_actuator_gains(dataset_path, num_joints=len(joint_names))
    print("=======================================================")
    print("|| joint_names:", joint_names, "||")
    print("=======================================================")
    print("|| actuator_names:", actuator_names, "||")
    print("=======================================================")
    if measured_qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"{dataset_path} has {measured_qpos.shape[1]} joints, "
            f"but the MuJoCo model expects {len(joint_names)}."
        )
    if desired_qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"{dataset_path} desired state has {desired_qpos.shape[1]} joints, "
            f"but the MuJoCo model expects {len(joint_names)}."
        )

    if actuator_mode == "general":
        ctrl = desired_qpos
    elif actuator_mode == "motor":
        if "des_dof_torque" in dataset:
            ctrl = np.asarray(dataset["des_dof_torque"], dtype=np.float64)
        else:
            ctrl = compute_pd_torques(
                desired_qpos=desired_qpos,
                desired_qvel=desired_qvel,
                measured_qpos=measured_qpos,
                measured_qvel=measured_qvel,
                kp=kp,
                kd=kd,
                ctrlrange=model.actuator_ctrlrange if model.nu else None,
            )
    else:
        raise ValueError(
            f"Unsupported actuator mode `{actuator_mode}`. "
            "Expected `motor` or `general`."
        )
    return ProcessedTrajectory(
        source_path=dataset_path,
        sequence_name=dataset_path.stem,
        times=times,
        measured_qpos=measured_qpos,
        measured_qvel=measured_qvel,
        desired_qpos=desired_qpos,
        ctrl=ctrl,
        joint_names=joint_names,
        actuator_names=actuator_names,
        kp=kp,
        kd=kd,
    )


def chunk_processed_trajectory(
    trajectory: ProcessedTrajectory, chunk_size: int
) -> list[ProcessedTrajectory]:
    if chunk_size <= 0:
        return [trajectory]
    if chunk_size < 2:
        raise ValueError("chunk_size must be >= 2.")

    chunks: list[ProcessedTrajectory] = []
    num_steps = trajectory.times.shape[0]
    start = 0
    chunk_index = 0
    print(trajectory.times)
    while start + chunk_size <= num_steps:
        end = start + chunk_size
        chunks.append(
            ProcessedTrajectory(
                source_path=trajectory.source_path,
                sequence_name=f"{trajectory.sequence_name}_chunk_{chunk_index:03d}",
                times=trajectory.times[start:end] - trajectory.times[start],
                measured_qpos=trajectory.measured_qpos[start:end],
                measured_qvel=trajectory.measured_qvel[start:end],
                desired_qpos=trajectory.desired_qpos[start:end],
                ctrl=trajectory.ctrl[start:end],
                joint_names=list(trajectory.joint_names),
                actuator_names=list(trajectory.actuator_names),
                kp=trajectory.kp,
                kd=trajectory.kd,
            )
        )
        start = end
        chunk_index += 1
    return chunks or [trajectory]


def processed_to_sysid_trajectory(sysid, model, trajectory: ProcessedTrajectory):
    
    measurement_ts = sysid.TimeSeries.from_names(
        trajectory.times,
        trajectory.measured_qpos,
        model
    )
    control_ts = sysid.TimeSeries(
        trajectory.times,
        trajectory.ctrl
    )
    initial_state = sysid.create_initial_state(
        model,
        trajectory.measured_qpos[0],
        trajectory.measured_qvel[0],
    )
    return measurement_ts, control_ts, initial_state


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def make_armature_modifier(joint_name):
    """Create a modifier that sets armature on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).armature = param.value[0]
    return modifier

def make_frictionloss_modifier(joint_name):
    """Create a modifier that sets frictionloss on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).frictionloss = param.value[0]
    return modifier

def make_damping_modifier(joint_name):
    """Create a modifier that sets damping on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).damping[0] = param.value[0]
    return modifier


def build_parameter_dict(sysid, model, joint_names: list[str], bounds: dict[str, tuple[float, float]]):
    parameter_dict = sysid.ParameterDict()
    for joint_name in joint_names:
        joint = model.joint(joint_name)
        for attribute in ("armature", "frictionloss","damping"):
            lower, upper = bounds[attribute]
            current_value = _as_scalar(getattr(joint, attribute))
            parameter_name = f"{joint_name}_{attribute}"
            parameter = sysid.Parameter(
                parameter_name,
                nominal=current_value,
                min_value=lower,
                max_value=upper,
                modifier=make_damping_modifier(joint_name) if attribute == "damping" else
                         make_armature_modifier(joint_name) if attribute == "armature" else
                         make_frictionloss_modifier(joint_name),
            )
            if attribute == "frictionloss":
                parameter.value[:] = current_value*0.1
            elif attribute == "armature":
                parameter.value[:] = current_value + np.ones_like(parameter.value) * 1e-2
            elif attribute == "damping":
                parameter.value[:] = current_value*2

            parameter_dict.add(parameter)
    print("Initial parameter vector:", parameter_dict.as_vector())
    return parameter_dict
