from functools import lru_cache
from typing import Optional
import cv2
import numpy as np
import open3d as o3d
import torch
import os
from utils.logger import *
from models.AdaPoinTr import AdaPoinTr
from models.GraspStability_PPCT import SGSNet
from pointnet2_ops import pointnet2_utils
from scipy.spatial.transform import Rotation as R

def _gather_by_index(data, idx):
    if data is None:
        return None
    idx = idx.unsqueeze(-1).expand(-1, -1, data.size(-1))
    return torch.gather(data, 1, idx)

def _get_sample_indices(vis_coor, target, method):
    batch_size, vis_len, _ = vis_coor.shape
    if method == 'fps' and vis_coor.is_cuda:
        return pointnet2_utils.furthest_point_sample(vis_coor.contiguous(), target)
    idx = torch.randperm(vis_len, device=vis_coor.device)[:target]
    return idx.unsqueeze(0).expand(batch_size, -1)

def sample_visual_inputs(vis_f, vis_coor, min_points=512, max_points=576, method='random',
                         training=True, eval_mode='dynamic', eval_points=None):
    vis_len = vis_coor.shape[1]
    if vis_len <= min_points:
        return vis_f, vis_coor
    max_points = min(max_points, vis_len)
    if (not training) and eval_mode == 'fixed':
        target = eval_points if eval_points is not None else max_points
        target = max(min_points, min(int(target), max_points))
    else:
        target = torch.randint(min_points, max_points + 1, (1,), device=vis_coor.device).item()
    if target == vis_len:
        return vis_f, vis_coor
    idx = _get_sample_indices(vis_coor, target, method)
    vis_coor = _gather_by_index(vis_coor, idx)
    vis_f = _gather_by_index(vis_f, idx)
    return vis_f, vis_coor

def replaceVisualEncoderParam(ckpt_path:str, model:SGSNet, weights_only=True):
    """
    user must self-check if two encoder has same config
    """
    assert type(model) == SGSNet, f"model must be SGSNet, but {type(model)} found"
    vis_encoder_ckpt:dict = torch.load(ckpt_path, weights_only=weights_only)
    print(vis_encoder_ckpt.items())
    vis_encoder_ckpt = vis_encoder_ckpt['base_model']
    adapointr_encoder_dict:dict = {}

    
    for k, v in vis_encoder_ckpt.items():
        if "base_model" in k and ("grouper" in k or "encoder" in k or "pos_embed" in k or "input_proj" in k):  
            new_key = k.replace("module.base_model", "visual_encoder")  
            adapointr_encoder_dict[new_key] = v

    model_dict = model.state_dict()
    model_dict = {k: v for k,v in model_dict.items() if "visual_encoder" in k}

    pretrained_dict = {k: v for k, v in adapointr_encoder_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


class RobustnessTestInjector:
    
    def __init__(self, random_seed: Optional[int] = None):
        self.levels = {
            1: {'t_min': 0.0, 't_max': 0.0005, 'r_min': 0.0, 'r_max': np.deg2rad(1)},  # 0-0.5mm, 0-1deg
            2: {'t_min': 0.0005, 't_max': 0.0010, 'r_min': np.deg2rad(1), 'r_max': np.deg2rad(2)},  # 0.5-1mm, 1-2deg
            3: {'t_min': 0.0010, 't_max': 0.0015, 'r_min': np.deg2rad(2), 'r_max': np.deg2rad(3)},  # 1-1.5mm, 2-3deg
            4: {'t_min': 0.0015, 't_max': 0.0020, 'r_min': np.deg2rad(3), 'r_max': np.deg2rad(4)},  # 1.5-2mm, 3-4deg
            5: {'t_min': 0.0020, 't_max': 0.0025, 'r_min': np.deg2rad(4), 'r_max': np.deg2rad(5)},  # 2-2.5mm, 4-5deg
        }
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_error_transform(self, level) -> np.ndarray:
        """
        Generate an error transformation matrix for the given level.
        
        Args:
            level: error level (1-5)
            
        Returns:
            4x4 error transformation matrix
        """
        params = self.levels[level]
        
        # Generate a random translation error with magnitude in [t_min, t_max] and random direction.
        t_magnitude = np.random.uniform(params['t_min'], params['t_max'])
        t_direction = np.random.randn(3)
        t_direction = t_direction / np.linalg.norm(t_direction)  # Normalize direction.
        t_error = t_magnitude * t_direction
        
        # Generate a random rotation error with angle in [r_min, r_max] and random axis.
        r_angle = np.random.uniform(params['r_min'], params['r_max'])
        r_axis = np.random.randn(3)
        r_axis = r_axis / np.linalg.norm(r_axis)  # Normalize axis.
        r_error = R.from_rotvec(r_angle * r_axis)
        
        # Compose the 4x4 transformation matrix.
        T_error = np.eye(4)
        T_error[:3, :3] = r_error.as_matrix()
        T_error[:3, 3] = t_error
        
        return T_error
    
    def apply_error_to_points(self, points, level) -> np.ndarray:
        """
        Apply the error transformation to point cloud data.
        
        Args:
            points: point cloud array (N, 3) or a list of point cloud arrays [list[np.ndarray]]
            level: error level
            
        Returns:
            transformed point cloud (N, 3) or a list of point cloud arrays [list[np.ndarray]]
        """
        # Generate one error transformation and apply it consistently to all point clouds.
        T_error = self.generate_error_transform(level)
        
        def _transform_single_pc(pc: np.ndarray) -> np.ndarray:
            if pc.shape[0] == 0:
                return pc
            # Convert to homogeneous coordinates.
            points_homo = np.hstack([pc, np.ones((pc.shape[0], 1))])
            # Apply transformation.
            points_transformed = (T_error @ points_homo.T).T
            return points_transformed[:, :3]
        
        # Support a single point cloud or a list of point clouds.
        if isinstance(points, list):
            return [_transform_single_pc(pc) for pc in points]
        else:
            return _transform_single_pc(points)
    
    def get_level_info(self, level) -> dict:
        """Return error-parameter information for the given level."""
        assert 1 <= level <= 5, f"Level must be 1-5, got {level}"
        params = self.levels[level]
        return {
            'level': level,
            'translation_min_mm': params['t_min'] * 1000,
            'translation_max_mm': params['t_max'] * 1000,
            'rotation_min_deg': np.rad2deg(params['r_min']),
            'rotation_max_deg': np.rad2deg(params['r_max']),
            'translation_range_m': f"[{params['t_min']:.5f}, {params['t_max']:.5f}]",
            'rotation_range_rad': f"[{params['r_min']:.5f}, {params['r_max']:.5f}]"
        }
    
    def reset_seed(self, seed: Optional[int] = None):
        """Reset the random seed."""
        if seed is not None:
            np.random.seed(seed)

class SegmentationRobustnessInjector:

    def __init__(self, over_levels=None, under_levels=None, default_k: int = 512,
                 interior_thresh: float = 3.0, max_hole_iter: int = 12):
        self.over_levels = over_levels or [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                                            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        self.under_levels = under_levels or [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                                             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        self.default_k = default_k
        self.interior_thresh = interior_thresh
        self.max_hole_iter = max_hole_iter
        self.debug = False

    def _resolve_level(self, level, levels) -> float:
        if isinstance(level, (int, np.integer)):
            if level <= 0:
                raise ValueError(f"Level {level} must be >= 1")
            idx = level - 1
            if idx >= len(levels):
                raise ValueError(f"Level {level} out of range (1-{len(levels)})")
            return float(levels[idx])
        level = float(level)
        if level <= 0.0:
            raise ValueError(f"Level {level} must be > 0")
        return level

    def _get_rng(self, rng):
        return rng if rng is not None else np.random

    def _dilate_to_target_overlap(self, obj_mask: np.ndarray, grp_mask: np.ndarray,
                                  target_rho: float, max_dilate_iter: Optional[int] = None):
        obj_mask = obj_mask.astype(bool)
        grp_mask = grp_mask.astype(bool)
        obj_area = int(np.count_nonzero(obj_mask))
        if obj_area == 0 or target_rho <= 0.0 or np.count_nonzero(grp_mask) == 0:
            return np.zeros_like(obj_mask, dtype=bool), 0.0, 0

        h, w = obj_mask.shape[:2]
        if max_dilate_iter is None:
            max_dilate_iter = max(h, w)
        if max_dilate_iter <= 0:
            return np.zeros_like(obj_mask, dtype=bool), 0.0, 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        cur = obj_mask.astype(np.uint8)
        best_added = np.zeros_like(obj_mask, dtype=bool)
        best_ratio = 0.0
        iters = 0

        for _ in range(max_dilate_iter):
            dilated = cv2.dilate(cur, kernel, iterations=1)
            if np.array_equal(dilated, cur):
                break
            iters += 1
            cur = dilated
            dilated_bool = dilated.astype(bool)
            added_mask = dilated_bool & grp_mask & (~obj_mask)
            added_area = int(np.count_nonzero(added_mask))
            ratio = added_area / obj_area if obj_area > 0 else 0.0
            best_added = added_mask
            best_ratio = ratio
            if ratio >= target_rho:
                break

        return best_added, best_ratio, iters

    def _apply_over_segmentation_from_mask(self, obj_mask: np.ndarray, grp_mask: np.ndarray,
                                           depth: np.ndarray, proj_mat: np.ndarray, level,
                                           align_to_view_coordinate: bool = True,
                                           max_dilate_iter: Optional[int] = None):
        rho = self._resolve_level(level, self.over_levels)
        if rho < 0.0:
            raise ValueError(f"Over-segmentation level must be >= 0, got {rho}")

        obj_mask = obj_mask.astype(bool)
        grp_mask = grp_mask.astype(bool)
        debug = self.debug
        if np.count_nonzero(obj_mask) == 0:
            if debug:
                print("[OverSeg] empty obj_mask, skip")
            return get_pointcloud_from_depth_mask(
                depth, proj_mat, obj_mask,
                align_to_view_coordinate=align_to_view_coordinate)
        if rho <= 0.0 or np.count_nonzero(grp_mask) == 0:
            if debug:
                print("[OverSeg] rho<=0 or empty grp_mask, skip")
            return get_pointcloud_from_depth_mask(
                depth, proj_mat, obj_mask,
                align_to_view_coordinate=align_to_view_coordinate)

        if debug:
            obj_area = int(np.count_nonzero(obj_mask))
            grp_area = int(np.count_nonzero(grp_mask))
            print(f"[OverSeg] level={level} target_rho={rho:.3f} obj_area={obj_area} grp_area={grp_area}")

        added_mask, actual_ratio, iters = self._dilate_to_target_overlap(
            obj_mask, grp_mask, rho, max_dilate_iter=max_dilate_iter)

        obj_pc = get_pointcloud_from_depth_mask(
            depth, proj_mat, obj_mask,
            align_to_view_coordinate=align_to_view_coordinate)
        if np.count_nonzero(added_mask) == 0:
            if debug:
                print(f"[OverSeg] added_area=0 actual_ratio={actual_ratio:.3f} iters={iters}")
            return obj_pc

        added_pc = get_pointcloud_from_depth_mask(
            depth, proj_mat, added_mask,
            align_to_view_coordinate=align_to_view_coordinate)
        added_points = np.asarray(added_pc.points)
        if added_points.size == 0:
            if debug:
                print(f"[OverSeg] added_points=0 actual_ratio={actual_ratio:.3f} iters={iters}")
            return obj_pc

        if debug:
            added_area = int(np.count_nonzero(added_mask))
            print(f"[OverSeg] added_area={added_area} actual_ratio={actual_ratio:.3f} iters={iters}")

        obj_points = np.asarray(obj_pc.points)
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(
            np.concatenate([obj_points, added_points], axis=0))
        return merged

    def apply_over_segmentation_mask(self, dep_path: str, seg_path: str, proj_mat: np.ndarray,
                                     obj_seg_id: int, grp_seg_id: int, level,
                                     align_to_view_coordinate: bool = True,
                                     max_dilate_iter: Optional[int] = None):
        depth = load_depth_image(dep_path)
        seg = load_segmentation_image(seg_path)
        obj_mask = (seg == obj_seg_id)
        grp_mask = (seg == grp_seg_id)
        return self._apply_over_segmentation_from_mask(
            obj_mask, grp_mask, depth, proj_mat, level,
            align_to_view_coordinate=align_to_view_coordinate,
            max_dilate_iter=max_dilate_iter)

    def _erode_to_target_removal(self, obj_mask: np.ndarray, target_delta: float,
                                 max_erode_iter: Optional[int] = None):
        obj_mask = obj_mask.astype(bool)
        obj_area = int(np.count_nonzero(obj_mask))
        if obj_area == 0 or target_delta <= 0.0:
            return obj_mask, 0.0, 0

        h, w = obj_mask.shape[:2]
        if max_erode_iter is None:
            max_erode_iter = max(h, w)
        if max_erode_iter <= 0:
            return obj_mask, 0.0, 0

        kernel = np.ones((3, 3), dtype=np.uint8)
        cur = obj_mask.astype(np.uint8)
        best_mask = obj_mask
        best_ratio = 0.0
        iters = 0

        for _ in range(max_erode_iter):
            eroded = cv2.erode(cur, kernel, iterations=1)
            if np.array_equal(eroded, cur):
                break
            iters += 1
            cur = eroded
            eroded_bool = eroded.astype(bool)
            eroded_area = int(np.count_nonzero(eroded_bool))
            ratio = (obj_area - eroded_area) / obj_area if obj_area > 0 else 0.0
            best_mask = eroded_bool
            best_ratio = ratio
            if ratio >= target_delta:
                break

        return best_mask, best_ratio, iters

    def _apply_under_segmentation_from_mask(self, obj_mask: np.ndarray, depth: np.ndarray,
                                            proj_mat: np.ndarray, level,
                                            align_to_view_coordinate: bool = True, rng=None):
        delta = self._resolve_level(level, self.under_levels)
        obj_mask = obj_mask.astype(bool)
        area = int(np.count_nonzero(obj_mask))
        debug = self.debug
        if debug:
            print(f"[UnderSeg] level={level} target_delta={delta:.3f} area={area}")
        if delta < 0.0 or area == 0:
            raise ValueError(f"Under-segmentation level must be >= 0 and object mask non-empty, got {delta}")
        if delta == 0.0:
            return get_pointcloud_from_depth_mask(
                depth, proj_mat, obj_mask,
                align_to_view_coordinate=align_to_view_coordinate)

        obj_mask_under, actual_delta, iters = self._erode_to_target_removal(
            obj_mask, delta, max_erode_iter=None)
        if debug:
            removed = int(area - np.count_nonzero(obj_mask_under))
            print(f"[UnderSeg] removed={removed} actual_delta={actual_delta:.3f} iters={iters}")

        return get_pointcloud_from_depth_mask(
            depth, proj_mat, obj_mask_under,
            align_to_view_coordinate=align_to_view_coordinate)

    def apply_under_segmentation(self, dep_path: str, seg_path: str, proj_mat: np.ndarray,
                                 obj_seg_id: int, level,
                                 align_to_view_coordinate: bool = True, rng=None):
        depth = load_depth_image(dep_path)
        seg = load_segmentation_image(seg_path)
        obj_mask = (seg == obj_seg_id)
        return self._apply_under_segmentation_from_mask(
            obj_mask, depth, proj_mat, level,
            align_to_view_coordinate=align_to_view_coordinate, rng=rng)


def testModel(shape_model:AdaPoinTr, grasp_model:SGSNet, config, num_warmup=10, num_runs=100):
    import time
    from thop import profile
    B, N_v, N_t, C = 1, 2048, 2400, 3  # Batch size, visual points, tactile points, feature dim
    vis_xyz = torch.randn(B, N_v, C)   # Visual point coordinates (B, N_v, 3).
    tac_xyzc = torch.randn(B, N_t, C+1)  # Tactile point coordinates plus contact flag (B, N_t, 4).
    
    shape_model.to('cuda')
    grasp_model.to('cuda')
    shape_model.eval()
    grasp_model.eval()  # Set to eval for inference timing

    vis_xyz = vis_xyz.cuda()
    tac_xyzc = tac_xyzc.cuda()

    # Warmup runs to stabilize GPU
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _, vis_f, vis_coor = shape_model(vis_xyz, return_latent=True)
            if not config.ablation.without_shape_completion:
                _ = grasp_model(vis_f, vis_coor, tac_xyzc)
            else:
                _ = grasp_model(None, vis_xyz, tac_xyzc)

    # Pre-compute shape features for timing measurement (exclude shape model from timing)
    with torch.no_grad():
        _, vis_f, vis_coor = shape_model(vis_xyz, return_latent=True)

    # Measure inference latency for grasp model only
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            # Only time the grasp stability inference
            if not config.ablation.without_shape_completion:
                out = grasp_model(vis_f, vis_coor, tac_xyzc)
            else:
                out = grasp_model(None, vis_xyz, tac_xyzc)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / num_runs
    throughput = num_runs / total_time  # samples per second
    
    print(f"=== Grasp Model Performance Metrics ===")
    print(f"Average Grasp Inference Latency: {avg_latency*1000:.2f} ms")
    print(f"Grasp Throughput: {throughput:.2f} samples/sec")

    # Profile FLOPs and parameters
    if config.ablation.without_shape_completion:
      flops, params = profile(grasp_model, (None, vis_xyz, tac_xyzc))
    else:
      flops, params = profile(grasp_model, (vis_f, vis_coor, tac_xyzc))
    print(f'Grasp FLOPs: {flops/1e9:.2f}G')
    print(f'Grasp Params: {params/1e6:.2f}M')

    print(f"Shape output shape: {vis_f.shape} {vis_coor.shape}")

    # for name, param in shape_model.named_parameters():
    #     if not param.requires_grad:
    #     if param.requires_grad and param.grad is None:

    # for name, param in grasp_model.named_parameters():
    #     if not param.requires_grad:
    #     if param.requires_grad and param.grad is None:

    if torch.isnan(out).any():
        print("Warning: NaN detected in output!")

    # Print output shape.
    print(f"Output shape: {out.shape}")  # Expected: (B, 2).
    
    # Return metrics for further analysis
    return {
        'avg_latency_ms': avg_latency * 1000,
        'throughput_sps': throughput,
        'flops': flops,
        'params': params,
        'output_shape': out.shape
    }
    
def get_shuffled_subset(dataset, ratio=0.2, batch_size=1, num_workers=1, max_sizes=(None, None), shuffle_btw_epoch=(True, False), shuffle_at_first=True, random_seed=42):
    import random
    from torch.utils.data import Subset, DataLoader
    
    num_samples = len(dataset)
    indices = list(range(num_samples))
    if shuffle_at_first:
        # Use a fixed seed for reproducible shuffling
        random.seed(random_seed)
        random.shuffle(indices)
        random.seed()  # Reset to random state
    
    # Compute initial split sizes.
    subset_size = int(ratio * num_samples)
    
    # Apply max_sizes limits.
    max_subset_size, max_rest_size = max_sizes
    if max_subset_size is not None:
        subset_size = min(subset_size, max_subset_size)
    
    # Compute the remaining split size.
    remaining_samples = num_samples - subset_size
    if max_rest_size is not None:
        rest_size = min(remaining_samples, max_rest_size)
    else:
        rest_size = remaining_samples
    
    # Create shuffled indices.
    subset_indices = indices[:subset_size]
    rest_indices = indices[subset_size:subset_size + rest_size]
    
    # Create subsets.
    subset = Subset(dataset, subset_indices)
    rest = Subset(dataset, rest_indices)
    
    # Create DataLoaders.
    subset_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle_btw_epoch[0],
        num_workers=num_workers,
        pin_memory=True
    )
    
    rest_loader = DataLoader(
        rest,
        batch_size=batch_size,
        shuffle=shuffle_btw_epoch[1],
        num_workers=num_workers,
        pin_memory=True
    )
    
    return subset_loader, rest_loader

def load_sampled_gt_pc(obj_id: str, mesh_path_template: str, npoints: int, cache_dir='./data/vtg_gt_cache') -> o3d.geometry.PointCloud:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{obj_id}_{npoints}_sampled_pc.npy")

    if os.path.exists(cache_path):
        # Load from cache.
        points = np.load(cache_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    # Load and sample normally.
    gt_mesh_path = mesh_path_template % obj_id
    mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    sampled_pc = mesh.sample_points_uniformly(number_of_points=npoints)
    points = np.asarray(sampled_pc.points)
    np.save(cache_path, points)

    return sampled_pc

def _getIntrinsicParams(proj_mat: np.ndarray, w: int, h: int):
  fx = proj_mat[0, 0] * w / 2
  fy = proj_mat[1, 1] * h / 2
  cx = w / 2
  cy = h / 2
  return fx, fy, cx, cy

def load_depth_image(dep_path: str) -> np.ndarray:
  dep_uint16 = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
  if dep_uint16 is None:
    raise FileNotFoundError(f"Failed to read depth image: {dep_path}")
  return dep_uint16.astype(np.float32) / 1000.0

def load_segmentation_image(seg_path: str) -> np.ndarray:
  seg_uint8 = cv2.imread(seg_path, cv2.IMREAD_ANYDEPTH)
  if seg_uint8 is None:
    raise FileNotFoundError(f"Failed to read segmentation image: {seg_path}")
  return seg_uint8.astype(np.int32) - 1

def get_pointcloud_from_depth_mask(dep: np.ndarray, proj_mat: np.ndarray, mask: np.ndarray,
                                   align_to_view_coordinate: bool = True):
  if dep is None or mask is None:
    raise ValueError("dep and mask must be provided")
  if dep.shape[:2] != mask.shape[:2]:
    raise ValueError(f"Depth/mask shape mismatch: {dep.shape} vs {mask.shape}")

  dep_masked = dep * mask.astype(np.float32)
  o3d_dep = o3d.geometry.Image(dep_masked)

  w, h = dep.shape[1], dep.shape[0]
  fx, fy, cx, cy = _getIntrinsicParams(proj_mat, w, h)

  intrinsic = o3d.camera.PinholeCameraIntrinsic()
  intrinsic.set_intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

  point_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_dep, intrinsic)

  if align_to_view_coordinate:
    points = np.asarray(point_cloud.points)
    R = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ])
    points = np.dot(points, R.T)
    point_cloud.points = o3d.utility.Vector3dVector(points)

  return point_cloud

def getVisualPointCloud(dep_path: str, proj_mat: np.ndarray, rgb_path: Optional[str], 
                  seg_path: Optional[str], seg_id: Optional[int], align_to_view_coordinate: bool = True):
  """
  Convert depth image to point cloud using Open3D.
  Args:
    dep_path (str): Path to the depth image.
    proj_mat (np.ndarray): Projection matrix.
    rgb_path (str): Path to the RGB image.
    seg_path (str): Path to the segmentation image.
    seg_id (int): Object ID for segmentation. (hand.id always 3, obj.id always 1)
    align_to_view_coordinate (bool): Whether to align the point cloud to view coordinate.
  Returns:
    o3d.geometry.PointCloud: Point cloud object.
  """
  # dep image
  dep_uint16 = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
  dep = dep_uint16.astype(np.float32) / 1000.0
  # rgb image
  if rgb_path is not None:
    rgb = cv2.cvtColor(rgb_path, cv2.COLOR_BGR2RGB)
  # seg image
  if seg_path is not None:
    seg_uint8 = cv2.imread(seg_path, cv2.IMREAD_ANYDEPTH)
    seg = seg_uint8.astype(np.int8) - 1

  # Extract points corresponding to the given obj_id
  if seg is not None and seg_id is not None:
    seg_mask = (seg == seg_id)
    dep = dep * seg_mask

  # convert np.ndarray to open3d.geometry.Image
  o3d_dep = o3d.geometry.Image(dep)

  if rgb_path is not None:
    assert dep.shape == rgb.shape[:2], "Depth and RGB should have the same shape."
    o3d_rgb = o3d.geometry.Image(rgb)
    img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_dep, 
                                 depth_scale=1.0, depth_trunc=5000.0, 
                                 convert_rgb_to_intensity=False)
  else:
    img = o3d_dep

  # get image size and intrinsic parameters
  w, h = dep.shape[1], dep.shape[0]
  fx, fy, cx, cy = _getIntrinsicParams(proj_mat, w, h)

  intrinsic = o3d.camera.PinholeCameraIntrinsic()
  intrinsic.set_intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

  if rgb_path is not None:
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(img, intrinsic)
  else:
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(img, intrinsic)

  if align_to_view_coordinate: # 
    points = np.asarray(point_cloud.points)
    # create a 3x3 rotation matrix that rotate 180 degree around x axis (open3d to openGL standard)
    R = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ])
    points = np.dot(points, R.T)
    point_cloud.points = o3d.utility.Vector3dVector(points)

  return point_cloud  # user needs to check shape (N, 3) or (N, 6)


def getTactilePointCloud(dep_path: str, proj_mat: np.ndarray, distance_cam2gel: float = 0.02315, noise_eps: float = 0.0001,
       align_to_gel_coordinate: bool = True, step:float = 8):
  """
  Generates a point cloud from a depth image using the provided projection matrix.

  Parameters:
    dep_path (str): Path to the depth image.
    proj_mat (np.ndarray): The projection matrix used to convert depth image to point cloud.
    distance_cam2gel (float, optional): Distance from the camera to the gel surface. Default is 0.0235.
    align_to_gel_coordinate (bool, optional): If True, aligns the point cloud to the gel coordinate system. Default is True.

  Returns:
    tuple: A tuple containing:
      - all_points: The point cloud containing all points.
      - gel_mask: A boolean mask indicating the gel surface points.
  """
  # load depth image
  dep_uint8 = cv2.imread(dep_path, cv2.IMREAD_ANYDEPTH)
  dep = dep_uint8.astype(np.float32) / (10.0 * 1000) # the original depth is in 10*mm, we convert to meter

  # restore the depth observed by camera
  dep0 = np.ones((320, 240), dtype=np.float32) * distance_cam2gel
  dep_original = dep0 - dep

  # create a mask to identify the gel surface
  gel_mask = dep_original > (distance_cam2gel - noise_eps)

  # Downsample the depth image by the given step
  mask_step = np.zeros_like(dep_original, dtype=bool)
  mask_step[::int(step), ::int(step)] = True
  dep_original[~mask_step] = 0.0
  gel_mask = gel_mask[::step, ::step]

  # convert np.ndarray to open3d.geometry.Image
  o3d_dep = o3d.geometry.Image(dep_original)
  
  # get image size and intrinsic parameters
  w, h = dep.shape[1], dep.shape[0]
  fx, fy, cx, cy = _getIntrinsicParams(proj_mat, w, h)

  intrinsic = o3d.camera.PinholeCameraIntrinsic(
  o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
  intrinsic.set_intrinsics(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)

  point_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_dep, intrinsic)

  if align_to_gel_coordinate:
    @lru_cache(maxsize=None)  # Adjust cache size here if needed.
    def _get_combined_matrix(distance_cam2gel):
      # create a 3x3 rotation matrix that rotate 180 degree around x axis (open3d to openGL standard)
      R_4 = np.array([
      [1, 0, 0, 0],
      [0, -1, 0, 0],
      [0, 0, -1, 0],
      [0, 0, 0, 1]
      ])
      # now we are at the camera coordinate system, we need to align to the gel coordinate system
      # the gel coordinate system has same position as the left tactile camera, but different orientation
      # gel coordinate system can be represented as [-z_cam, -x_cam, y_cam]
      # so the transform matrix is [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], we then apply on it
      T_Cam2Gel = np.array([
      [0,  0, -1, -distance_cam2gel], # 0.0235 is the distance from camera to gel
      [-1, 0,  0, 0],
      [0,  1,  0, 0],
      [0,  0,  0, 1]
      ])
      return np.dot(T_Cam2Gel, R_4)
    
    points = np.asarray(point_cloud.points)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    combined_mat = _get_combined_matrix(distance_cam2gel)
    transformed_points = np.dot(points, combined_mat.T)
    point_cloud.points = o3d.utility.Vector3dVector(transformed_points[:, :3])

  return point_cloud, gel_mask.flatten()

def downsample_pointcloud_with_indices(pc: o3d.geometry.PointCloud, method: str, num_points: int):
    """
    Downsample a point cloud using random or uniform sampling, and return the sampled point cloud and original indices.

    Parameters:
        pc (o3d.geometry.PointCloud): The input point cloud.
        method (str): 'random' or 'uniform'.
        num_points (int): Number of points to retain.

    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud.

        list(int): The indices of the selected points from the original point cloud.
    """
    total_points = len(pc.points)
    if num_points >= total_points:
        return pc, list(range(total_points))

    if method == 'random':
        indices = np.random.choice(total_points, num_points, replace=False)
        indices.sort()  # optional: keep order consistent
        downpcd = pc.select_by_index(indices)
        return downpcd, indices.tolist()

    elif method == 'uniform':
        step = max(1, total_points // num_points)
        indices = list(range(0, total_points, step))[:num_points]
        downpcd = pc.select_by_index(indices)
        return downpcd, indices

    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'random' or 'uniform'.")

def downsample_by_dist_ratio(vis_pc: np.ndarray, tac_pc_l: np.ndarray, tac_pc_r: np.ndarray,
                             num_points: int, tac_scale: float = 1.0, debug: bool = False) -> np.ndarray:
    def _max_aabb_distance(points: np.ndarray) -> float:
      if len(points) == 0:
        # print("Warning: Point cloud is empty. Returning distance of 0.")
        return 0.0
      max_vals = np.max(points, axis=0)
      min_vals = np.min(points, axis=0)
      
      diff = max_vals - min_vals
      distance = np.linalg.norm(diff)
    
      return distance 
    
    def _safe_random_sample_np(points: np.ndarray, n: int) -> np.ndarray:
        n_total = len(points)
        if n >= n_total:
            return points.copy()
        idx = np.random.choice(n_total, n, replace=False)
        return points[idx]

    # Compute max distances (scales)
    vis_scale = _max_aabb_distance(vis_pc)
    tac_scale_l = _max_aabb_distance(tac_pc_l) * tac_scale
    tac_scale_r = _max_aabb_distance(tac_pc_r) * tac_scale

    total_scale = vis_scale + tac_scale_l + tac_scale_r
    if total_scale == 0:
        raise ValueError("All point clouds have zero scale. Cannot perform downsampling.")

    vis_scale /= total_scale
    tac_scale_l /= total_scale
    tac_scale_r /= total_scale

    if debug:
        pass
        # print(f"[Scale Downsample] VIS scale: {vis_scale * 100:.2f}%, TAC_L scale: {tac_scale_l * 100:.2f}%, TAC_R scale: {tac_scale_r * 100:.2f}%")

    from math import floor
    vis_n = int(floor(vis_scale * num_points))
    tac_l_n = int(floor(tac_scale_l * num_points))
    tac_r_n = int(floor(tac_scale_r * num_points))

    # tac tend to be less than sample, so add to vis to make up
    tac_l_n = min(tac_l_n, len(tac_pc_l))
    tac_r_n = min(tac_r_n, len(tac_pc_r))
    delta = num_points - (vis_n + tac_l_n + tac_r_n) # at best delta equals to vis_n, otherwise larger
    vis_n = min(vis_n + delta, len(vis_pc)) # add to vis

    vis_pc_ds = _safe_random_sample_np(vis_pc, vis_n)
    tac_pc_l_ds = _safe_random_sample_np(tac_pc_l, tac_l_n)
    tac_pc_r_ds = _safe_random_sample_np(tac_pc_r, tac_r_n)

    # concat current samples
    current_number = vis_pc_ds.shape[0] + tac_pc_l_ds.shape[0] + tac_pc_r_ds.shape[0]

    # if still less than num_points, then random re-sample
    while current_number < num_points:
        extra_needed = num_points - current_number
        if debug:
          print(f"\033[93mWarning: Resample number {extra_needed}!\033[0m")
        if vis_pc_ds.shape[0] == 0:
            extra_samples = np.zeros((extra_needed, 3), dtype=np.float32)
        else:
            extra_samples = _safe_random_sample_np(vis_pc_ds, extra_needed)
        vis_pc_ds = np.concatenate([vis_pc_ds, extra_samples], axis=0)
        current_number = vis_pc_ds.shape[0] + tac_pc_l_ds.shape[0] + tac_pc_r_ds.shape[0]

    return vis_pc_ds, tac_pc_l_ds, tac_pc_r_ds


def downsample_pointcloud(pc: o3d.geometry.PointCloud, method, voxel_size=None, num_points=None):
    if method == 'voxel':
        assert voxel_size is not None, "voxel_size must be provided for voxel downsampling."
        return pc.voxel_down_sample(voxel_size)

    elif method == 'random':
        assert num_points is not None, "num_points must be provided for random downsampling."
        if num_points >= len(pc.points):
            return pc
        return pc.random_down_sample(num_points / len(pc.points))

    elif method == 'uniform':
        assert num_points is not None, "num_points must be provided for uniform downsampling."
        if num_points >= len(pc.points):
            return pc
        step = max(1, len(pc.points) // num_points)
        return pc.uniform_down_sample(step)

    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'voxel', 'random', or 'uniform'.")


def _transformPoseToMat(pose:tuple):
    r = R.from_quat(pose[1])
    # create a 4x4 matrix
    poseMat = np.eye(4)
    poseMat[:3, :3] = r.as_matrix()
    poseMat[:3, 3] = pose[0]
    return poseMat

def align_to_world(points:o3d.geometry.PointCloud, pose_tuple:Optional[tuple] = None, pose_mat:Optional[np.ndarray] = None):
    """
    Align the point cloud to the world coordinate system using the given pose.

    Parameters:
      points (o3d.geometry.PointCloud): The point cloud to be aligned.
      pose_tuple (tuple, optional): A tuple containing translation and rotation (translation, rotation).
      pose_mat (np.ndarray, optional): A 4x4 transformation matrix.

    Returns:
      o3d.geometry.PointCloud: The aligned point cloud.
    """
    assert pose_tuple is not None or pose_mat is not None, "Either pose_tuple or pose_mat must be provided."

    if pose_mat is None: # pose_tuple
        pose_mat = _transformPoseToMat(pose_tuple)
    
    points.transform(pose_mat)
    
    return points
  
def get_zero_mean(points_list: list):
  """
  Calculate the zero mean for a list of numpy arrays or torch tensors.

  Parameters:
    points_list (list): A list of numpy arrays or torch tensors.

  Returns:
    mean: The calculated mean as a numpy array.
  """
  assert len(points_list) > 0, "points_list should not be empty"
  assert all(isinstance(points, (np.ndarray, torch.Tensor)) for points in points_list), \
    "All elements in points_list should be numpy arrays or torch tensors"

  # Convert all tensors to numpy arrays
  points_list_np = [points.cpu().numpy() if isinstance(points, torch.Tensor) else points for points in points_list]

  # Calculate the mean for numpy arrays
  mean = np.mean(np.concatenate(points_list_np, axis=0), axis=0)

  return mean

def apply_zero_means(points_list:list[np.ndarray], mean:np.ndarray):
    """
    Apply zero mean for a list of numpy arrays.
    """
    assert len(points_list) > 0, "points_list should not be empty"
    assert all(isinstance(points, np.ndarray) for points in points_list), "All elements in points_list should be numpy arrays"
    
    # subtract the mean from each point cloud
    return [points - mean for points in points_list]
