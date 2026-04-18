from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import build_fixed_base_model_xml
from sysid_mujoco.common import build_actuator_gain_map
from sysid_mujoco.common import build_parameter_dict
from sysid_mujoco.common import chunk_processed_trajectory
from sysid_mujoco.common import get_actuated_joint_and_actuator_names
from sysid_mujoco.common import load_dataset_actuator_gains
from sysid_mujoco.common import load_processed_dataset
from sysid_mujoco.common import processed_to_sysid_trajectory

import mujoco
import mujoco.rollout as rollout
from mujoco import sysid
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from absl import logging
import config
import numpy as np


def default_converted_paths(robot: str) -> list[Path]:
    return sorted((REPO_ROOT / "sysid_mujoco" / "converted" / robot).glob("*.npz"))


def default_output_dir(robot: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / robot / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-joint damping, armature, and frictionloss from "
            "quadruped datasets using MuJoCo sysid."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=Path,
        help="Raw repository datasets (.pt). If provided, conversion is done in-memory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the fitted parameters and identified XML models.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("mujoco", "scipy", "scipy_parallel_fd"),
        default="mujoco",
        help="Optimizer backend exposed by mujoco.sysid.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If > 0, split each source trajectory into non-overlapping chunks of this size.",
    )
    parser.add_argument(
        "--damping-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.5, 3.0),
        help="Bounds for each joint damping parameter.",
    )
    parser.add_argument(
        "--armature-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.0, 0.6),
        help="Bounds for each joint armature parameter.",
    )
    parser.add_argument(
        "--frictionloss-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.01, 5.0),
        help="Bounds for each joint frictionloss parameter.",
    )
    return parser.parse_args()


def build_model_sequences_from_source(
    sysid,
    mujoco,
    model,
    dataset_paths: list[Path],
    chunk_size: int,
):
    measurement_ts = []
    control_ts = []
    initial_states = []

    for dataset_path in dataset_paths:

        processed = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
        )
        for chunk in chunk_processed_trajectory(processed, chunk_size):
            measurement_data, control_data, initial_state = processed_to_sysid_trajectory(sysid, model, chunk)
            measurement_ts.append(measurement_data)
            control_ts.append(control_data)
            initial_states.append(initial_state)

    return measurement_ts, control_ts, initial_states


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise ValueError("Use --dataset to pass a dataset")

    dataset_kp, dataset_kd = load_dataset_actuator_gains(args.dataset[0])


    fixed_base_xml = build_fixed_base_model_xml(args.robot)
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_model = fixed_base_spec.compile()
    actuated_joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, fixed_base_model)
    fixed_base_xml = build_fixed_base_model_xml(
        args.robot,
        actuator_gains=build_actuator_gain_map(
            actuated_joint_names,
            dataset_kp,
            dataset_kd,
        ),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_model = fixed_base_spec.compile()

    measurement_ts, control_ts, initial_states = build_model_sequences_from_source(
        sysid=sysid,
        mujoco=mujoco,
        model=fixed_base_model,
        dataset_paths=args.dataset,
        chunk_size=args.chunk_size,
    )
    input_descriptions = [str(path.resolve()) for path in args.dataset]

    joint_names = [fixed_base_model.joint(i).name for i in range(fixed_base_model.njnt)]
    sequence_names = [f"sequence_{index:03d}" for index, _ in enumerate(measurement_ts)]
    model_sequences = [sysid.ModelSequences(
        args.robot,
        fixed_base_spec,
        sequence_names[i],
        initial_states[i],
        control_ts[i],
        measurement_ts[i]
    ) for i in range(len(measurement_ts))]

    bounds = {
        "damping": tuple(float(value) for value in args.damping_bounds),
        "armature": tuple(float(value) for value in args.armature_bounds),
        "frictionloss": tuple(float(value) for value in args.frictionloss_bounds),
    }

    params = build_parameter_dict(
        sysid=sysid,
        model=fixed_base_model,
        joint_names=joint_names,
        bounds=bounds,
    )
   
    residual_fn = residual_fn = sysid.build_residual_fn(models_sequences=model_sequences)

    opt_params, opt_result = sysid.optimize(
    initial_params=params,
    residual_fn=residual_fn,
    optimizer='mujoco'
    )



    output_dir = args.output_dir or default_output_dir(args.robot)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = sysid.default_report(
    models_sequences=model_sequences,
    initial_params=params,
    opt_params=opt_params,
    residual_fn=residual_fn,
    opt_result=opt_result,
    title=f"System Identification Report for {args.robot}",
    generate_videos=False,
    )
    def display_report(report, report_path: Path) -> Path:
        html = report.build()
        report_path.write_text(html, encoding="utf-8")
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            if getattr(get_ipython(), "kernel", None) is not None:
                display(HTML(html))
        except Exception:
            pass
        print(f"Report written to {report_path}")
        return report_path

    display_report(report, output_dir / "report.html")
    # write_joint_parameter_model(
    #     source_xml=get_robot_model_xml_path(args.robot),
    #     output_xml=floating_base_model_xml,
    #     identified_parameters=identified_parameters,
    #     fixed_base=False,
    # )
    # write_joint_parameter_model(
    #     source_xml=fixed_base_xml,
    #     output_xml=fixed_base_model_out_xml,
    #     identified_parameters=identified_parameters,
    #     fixed_base=True,
    # )
    # write_simple_scene(floating_base_scene_xml, floating_base_model_xml)
    # write_simple_scene(fixed_base_scene_xml, fixed_base_model_out_xml)

    # try:
    #     opt_params.save_to_disk(output_dir / "identified_parameters.yaml")
    # except Exception as exc:
    #     print(f"Warning: could not save ParameterDict YAML: {exc}")

    # summary_payload = {
    #     "robot": args.robot,
    #     "inputs": input_descriptions,
    #     "optimizer": args.optimizer,
    #     "max_iters": int(args.max_iters),
    #     "fixed_base_model_xml": str(fixed_base_xml.resolve()),
    #     "initial_cost": initial_cost,
    #     "final_cost": final_cost,
    #     "cost_reduction": initial_cost - final_cost,
    #     "bounds": bounds,
    #     "joint_names": joint_names,
    #     "num_sequences": len(initial_states),
    #     "parameters": parameter_summary,
    #     "artifacts": {
    #         "floating_base_model_xml": str(floating_base_model_xml.resolve()),
    #         "floating_base_scene_xml": str(floating_base_scene_xml.resolve()),
    #         "fixed_base_model_xml": str(fixed_base_model_out_xml.resolve()),
    #         "fixed_base_scene_xml": str(fixed_base_scene_xml.resolve()),
    #     },
    # }
    # if hasattr(opt_result, "extras"):
    #     summary_payload["optimizer_extras"] = {
    #         key: value if isinstance(value, (int, float, str, bool)) else len(value)
    #         for key, value in opt_result.extras.items()
    #     }
    # write_json(output_dir / "fit_summary.json", summary_payload)

    # print(f"Initial cost: {initial_cost:.6f}")
    # print(f"Final cost:   {final_cost:.6f}")
    # print_parameter_summary(parameter_summary)
    # print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
