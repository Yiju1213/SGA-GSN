import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="SGA-GSN public VTG training entry")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--subset", type=str, default="test", help="specific test subset")
    parser.add_argument("--_3dvtg", action="store_true", default=False, help="use 3D VTG dataset")
    parser.add_argument("--vtg3d_shape_completion", action="store_true", default=False, help="3D VTG shape completion task")
    parser.add_argument("--vtg3d_grasp_stability", action="store_true", default=False, help="3D VTG grasp stability task")
    parser.add_argument("--_2dvtg", action="store_true", default=False, help="use 2D VTG dataset")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic CUDNN options")
    parser.add_argument("--sync_bn", action="store_true", default=False, help="use sync batch norm")
    parser.add_argument("--exp_name", type=str, default="default", help="experiment name")
    parser.add_argument("--start_ckpts", type=str, default=None, help="checkpoint used to initialize model")
    parser.add_argument("--ckpts", type=str, default=None, help="checkpoint used in test mode")
    parser.add_argument("--val_freq", type=int, default=1, help="validation frequency")
    parser.add_argument("--resume", action="store_true", default=False, help="resume interrupted training")
    parser.add_argument("--test", action="store_true", default=False, help="test mode")
    args = parser.parse_args()

    if args._3dvtg == args._2dvtg:
        raise ValueError("Exactly one of --_3dvtg or --_2dvtg must be set.")
    if args._3dvtg and (args.vtg3d_shape_completion == args.vtg3d_grasp_stability):
        raise ValueError("For --_3dvtg, set exactly one of --vtg3d_shape_completion or --vtg3d_grasp_stability.")
    if args._2dvtg and (args.vtg3d_shape_completion or args.vtg3d_grasp_stability):
        raise ValueError("2D VTG does not use 3D VTG task flags.")
    if args.test and args.resume:
        raise ValueError("--test and --resume cannot both be active")
    if args.resume and args.start_ckpts is not None:
        raise ValueError("--resume and --start_ckpts cannot both be active")
    if args.test and args.ckpts is None:
        raise ValueError("--ckpts is required in test mode")

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.test:
        args.exp_name = "test_" + args.exp_name
    args.experiment_path = os.path.join("./experiments", Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join("./experiments", Path(args.config).stem, Path(args.config).parent.stem, "TFBoard", args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.tfboard_path, exist_ok=True)
