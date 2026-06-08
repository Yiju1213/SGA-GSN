from re import L
from cv2 import merge
import torch.utils.data as data
import numpy as np
import open3d as o3d
import os, sys, csv, json
from tqdm import tqdm
from multiprocessing import Pool
import torch
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.logger import *
from utils import vtg3d_utils
from datasets.build import DATASETS

def _load_metadata(args):
    data_root, obj_id = args  # Unpack the tuple into two variables
    json_path = os.path.join(data_root, obj_id, '_metadata.json')
    with open(json_path, 'r') as json_file:
        return obj_id, json.load(json_file)

@DATASETS.register_module()
class VTG3D(data.Dataset):
    def __init__(self, config, debug = False):
        # load config
        self.data_root = config.DATA_PATH
        self.tac_dep_pt = config.TAC_DEP_PATH
        self.tac_rgb_pt = config.TAC_RGB_PATH
        self.vis_dep_pt = config.VIS_DEP_PATH
        self.vis_rgb_pt = config.VIS_RGB_PATH
        self.vis_seg_pt = config.VIS_SEG_PATH
        self.complete_mesh_path = config.COMPLETE_MESH_PATH
        self.gt_points = config.GT_POINTS
        self.vis_points = config.VIS_POINTS
        self.tac_points = config.TAC_POINTS
        self.sc_input_points = config.SC_INPUT_POINTS
        self.tac_noise = config.TAC_NOISE
        self.vis_noise = config.VIS_NOISE
        self.subset = config.subset # being merged from config.others.subset
        self.subset_id_file = os.path.join(self.data_root, f'{self.subset}-ids.txt')
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.csv')
        self.debug = debug
        self.only_vis = config.ONLY_VIS
        self.without_spatial_align = getattr(config, 'without_spatial_align', False)  # Default to False if not specified
        
        # alignment robustness (pose perturbation on tactile-aligned points)
        self.align_robust_test = False
        self.align_robust_level = 0
        self.align_robust_injector = vtg3d_utils.RobustnessTestInjector()
        # segmentation robustness (visual)
        self.seg_robust_test = False
        self.seg_robust_mode = None  # "over" or "under"
        self.seg_robust_level = 0
        self.seg_robust_k = 300
        self.seg_robust_seed = None
        self.seg_obj_id = 1
        self.seg_grp_id = 3
        self.seg_injector = vtg3d_utils.SegmentationRobustnessInjector()
        self.return_vis_pc_align_original = False

        print_log(f'[DATASET] Open file {self.data_list_file} to load dataset indice list', logger = f"{config.NAME}")
        self.file_list = []
        with open(self.data_list_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row if it exists
            for row in reader:
                self.file_list.append(tuple(row))

        print_log(f'[DATASET] Open file {self.subset_id_file} to get id list', logger = f"{config.NAME}")
        with open(self.subset_id_file, 'r') as file:
            obj_ids = [line.strip().zfill(3) for line in file.readlines()]
        self.num_classes = len(obj_ids)
        print_log(f'[DATASET] Total {self.num_classes} objects in {self.subset} subset', logger = f"{config.NAME}")
        
        print_log(f'[DATASET] Loading metadata.json of all object', logger = f"{config.NAME}")
        
        self.metadata_dict = {}
        with Pool(processes=6) as pool:
            obj_id_pairs = [(self.data_root, obj_id) for obj_id in obj_ids]
            result = pool.imap_unordered(_load_metadata, obj_id_pairs)
            for obj_id, metadata in tqdm(result, total=len(obj_id_pairs), desc='Loading metadata'):
                self.metadata_dict[obj_id] = metadata

    
    def __getitem__(self, idx):
        ### load data
        sample = self.file_list[idx]
        obj_id, grasp_id = sample
        metadata = self.metadata_dict[obj_id][grasp_id]
        if self.align_robust_test and self.seg_robust_test:
            raise AssertionError("align_robust_test and seg_robust_test cannot both be enabled.")
        view_mat = np.asarray(metadata['viewMat']).reshape(4,4)

        ### sensory data as network input
        # 1. visual point cloud from segmented visual depth
        vis_dep_path = self.vis_dep_pt % (obj_id, grasp_id)
        vis_seg_path = self.vis_seg_pt % (obj_id, grasp_id)
        vis_proj_mat = np.asarray(metadata['visCamProjMat']).reshape(4,4)
        if self.seg_robust_test and self.without_spatial_align:
            raise ValueError("Segmentation robustness requires spatial alignment.")
        if self.seg_robust_test and self.seg_robust_mode not in ('over', 'under'):
            raise ValueError(f"Invalid seg_robust_mode: {self.seg_robust_mode}")

        rng = None
        if self.seg_robust_test and self.seg_robust_seed is not None:
            rng = np.random.RandomState(self.seg_robust_seed + idx)
        # generate visual point cloud under three conditions(normal / under-seg / over-seg)
        if self.seg_robust_test:
            if self.seg_robust_mode == 'under':
                vis_pc = self.seg_injector.apply_under_segmentation(
                    dep_path=vis_dep_path, seg_path=vis_seg_path, proj_mat=vis_proj_mat,
                    obj_seg_id=self.seg_obj_id, level=self.seg_robust_level,
                    align_to_view_coordinate=True, rng=rng)
            else:  # over-seg mode, use mask-level dilation + intersection
                vis_pc = self.seg_injector.apply_over_segmentation_mask(
                    dep_path=vis_dep_path, seg_path=vis_seg_path, proj_mat=vis_proj_mat,
                    obj_seg_id=self.seg_obj_id, grp_seg_id=self.seg_grp_id,
                    level=self.seg_robust_level, align_to_view_coordinate=True)
        else: # default case for normal data loading
            vis_pc = vtg3d_utils.getVisualPointCloud(
                dep_path=vis_dep_path, proj_mat=vis_proj_mat,
                rgb_path=None, seg_path=vis_seg_path, seg_id=self.seg_obj_id,
                align_to_view_coordinate=True)
        # 2. tactile point cloud from tactile depth(tac_pc include gel and contact)
        tac_dep_l_path = self.tac_dep_pt % (obj_id, f'{grasp_id}_l')
        tac_dep_r_path = self.tac_dep_pt % (obj_id, f'{grasp_id}_r')
        tac_proj_mat = np.asarray(metadata['tacCamProjMat']).reshape(4,4)
        tac_pc_l, gel_mask_l = vtg3d_utils.getTactilePointCloud(dep_path=tac_dep_l_path, proj_mat=tac_proj_mat,
                                                  distance_cam2gel=0.02315, align_to_gel_coordinate=True)
        tac_pc_r, gel_mask_r = vtg3d_utils.getTactilePointCloud(dep_path=tac_dep_r_path, proj_mat=tac_proj_mat,
                                                  distance_cam2gel=0.02315, align_to_gel_coordinate=True)
        # 3. sampling points
        vis_pc, _ = vtg3d_utils.downsample_pointcloud_with_indices(vis_pc, method='random', num_points=self.vis_points)
        # actually, the tactile point cloud is already uniformly sampled 
        # in `getTactilePointCloud` to the size of `self.tac_points` for computation efficiency
        # so this is a check to ensure the size is correct
        tac_pc_l, tac_idx_l = vtg3d_utils.downsample_pointcloud_with_indices(tac_pc_l, method='uniform', num_points=self.tac_points)
        tac_pc_r, tac_idx_r = vtg3d_utils.downsample_pointcloud_with_indices(tac_pc_r, method='uniform', num_points=self.tac_points)

        # 4. aligned point cloud from visual and tactile using cams' poses
        if self.without_spatial_align: 
            # if in ablation study, the point clouds will not be aligned to world coordinates
            # but for the rest of the code, we still need to naming these variables as xxx_align
            vis_pc_align = np.asarray(vis_pc.points)
            tac_pc_l_align = np.asarray(tac_pc_l.points)
            tac_pc_r_align = np.asarray(tac_pc_r.points)
        else:
            vis_pc_align = vtg3d_utils.align_to_world(vis_pc, pose_mat=np.linalg.inv(view_mat))
            tac_pc_l_align = vtg3d_utils.align_to_world(tac_pc_l, pose_tuple=metadata['grasping']['left-gel-pose'])
            tac_pc_r_align = vtg3d_utils.align_to_world(tac_pc_r, pose_tuple=metadata['grasping']['right-gel-pose'])
            vis_pc_align = np.asarray(vis_pc_align.points)
            tac_pc_l_align = np.asarray(tac_pc_l_align.points)
            tac_pc_r_align = np.asarray(tac_pc_r_align.points)
        
        # 5. get contact region
        tac_pc_l_align_contact = tac_pc_l_align[~gel_mask_l[tac_idx_l]].copy()
        tac_pc_r_align_contact = tac_pc_r_align[~gel_mask_r[tac_idx_r]].copy()

        ### ground truth
        # 1. complete shape point cloud from .obj file
        gt_pc = vtg3d_utils.load_sampled_gt_pc(obj_id, self.complete_mesh_path, self.gt_points)
        if self.without_spatial_align:
            gt_pc = np.asarray(gt_pc.points)
        else:
            gt_pc = vtg3d_utils.align_to_world(gt_pc, pose_tuple=metadata['grasping']['obj-pose'])
            gt_pc = np.asarray(gt_pc.points)
        assert gt_pc.shape[0] == self.gt_points, f"GT points {gt_pc.shape[0]} is not equal to {self.gt_points}"
        gt_origin = gt_pc.copy()  

        # 2. grasp stabiblity result
        grasp_result = metadata['isPositive']

        if self.align_robust_test:
            tac_pc_l_align, tac_pc_r_align, tac_pc_l_align_contact, tac_pc_r_align_contact \
                = self.align_robust_injector.apply_error_to_points(
                [tac_pc_l_align, tac_pc_r_align, tac_pc_l_align_contact, tac_pc_r_align_contact], 
                self.align_robust_level)

        ### zero-means(get& apply)
        if self.without_spatial_align:
            # For ablation study, use a dummy zero tensor instead of None to avoid DataLoader issues
            zero_mean = torch.zeros(3)  # Use a tensor instead of None
        else:
            zero_mean = vtg3d_utils.get_zero_mean([vis_pc_align, tac_pc_l_align_contact, tac_pc_r_align_contact]) # using only shape information
            tac_pc_l_align, tac_pc_r_align, vis_pc_align, tac_pc_l_align_contact, tac_pc_r_align_contact, gt_pc = \
                vtg3d_utils.apply_zero_means([tac_pc_l_align, tac_pc_r_align, vis_pc_align, tac_pc_l_align_contact, tac_pc_r_align_contact, gt_pc], zero_mean)
            # zero_mean on gt_origin
            gt_origin -= zero_mean  # Apply zero mean to the ground truth origin
            # Convert numpy zero_mean to tensor for consistency
            zero_mean = torch.from_numpy(zero_mean).float()

        ### return data 
        points = {}
        # for grasp stability
        assert tac_pc_l_align.shape[0] + tac_pc_r_align.shape[0] == 2 * self.tac_points, "Tactile points are not equal to 2 * self.tac_points"
        
        # for shape completion, all_pc should be added to same size
        if self.only_vis or self.without_spatial_align:
            # clear tactile points to empty
            tac_pc_l_align_contact = np.zeros((0, 3))
            tac_pc_r_align_contact = np.zeros((0, 3))
            
        vis_pc_sc, tac_pc_l_sc, tac_pc_r_sc = vtg3d_utils.downsample_by_dist_ratio(vis_pc=vis_pc_align,
                                                                                   tac_pc_l=tac_pc_l_align_contact,
                                                                                   tac_pc_r=tac_pc_r_align_contact,
                                                                                   num_points=self.sc_input_points,
                                                                                   debug=self.debug)
        
        total = vis_pc_sc.shape[0] + tac_pc_l_sc.shape[0] + tac_pc_r_sc.shape[0]
        if total != self.sc_input_points:
            raise ValueError(f"Shape completion input points {total} is not equal to {self.sc_input_points}")

        # augmentation on noise & scale, only for training data
        if self.subset == 'train':
            def _sc_augment(pc, scale, noise_std=0.001):
                if pc.shape[0] == 0:
                    return pc
                noise = np.random.normal(scale=noise_std, size=pc.shape)
                return pc * scale + noise
            
            scale = np.random.uniform(0.95, 1.05)
            vis_pc_sc = _sc_augment(vis_pc_sc, scale, noise_std=self.vis_noise)
            tac_pc_l_sc = _sc_augment(tac_pc_l_sc, scale, noise_std=self.tac_noise)
            tac_pc_r_sc = _sc_augment(tac_pc_r_sc, scale, noise_std=self.tac_noise)
            gt_pc = gt_pc * scale
            if self.debug:
                pass
                # print(f"Shape completion scale = {scale:.2f}")
                
        # Add gel-dim for grasp_stability (common for both cases)
        gel_mask_l_float = gel_mask_l.astype(np.float32)
        gel_dim_l = gel_mask_l_float[tac_idx_l][:, None]  # (N, 1)
        tac_pc_l_align_with_gel = np.concatenate([tac_pc_l_align, gel_dim_l], axis=1)  # (N, 4)

        gel_mask_r_float = gel_mask_r.astype(np.float32)
        gel_dim_r = gel_mask_r_float[tac_idx_r][:, None]  # (N, 1)
        tac_pc_r_align_with_gel = np.concatenate([tac_pc_r_align, gel_dim_r], axis=1)  # (N, 4)
        
        vis_pc_align_original = None
        if self.seg_robust_test and self.return_vis_pc_align_original:
            vis_pc_original = vtg3d_utils.getVisualPointCloud(
                dep_path=vis_dep_path, proj_mat=vis_proj_mat,
                rgb_path=None, seg_path=vis_seg_path, seg_id=self.seg_obj_id,
                align_to_view_coordinate=True)
            if self.without_spatial_align:
                vis_pc_align_original = np.asarray(vis_pc_original.points)
            else:
                vis_pc_original = vtg3d_utils.align_to_world(
                    vis_pc_original, pose_mat=np.linalg.inv(view_mat))
                vis_pc_align_original = np.asarray(vis_pc_original.points)
                zero_mean_np = zero_mean.cpu().numpy()
                vis_pc_align_original = vis_pc_align_original - zero_mean_np

        if self.debug:
            points['original'] = (torch.from_numpy(vis_pc_align).float(), 
                      torch.from_numpy(tac_pc_l_align_contact).float(),
                      torch.from_numpy(tac_pc_r_align_contact).float(),
                      torch.from_numpy(gt_origin).float())
            points['grasp_stability'] = (
                torch.from_numpy(tac_pc_l_align_with_gel).float(),
                torch.from_numpy(tac_pc_r_align_with_gel).float(),
                torch.tensor(grasp_result).float()
            )
            points['shape_completion'] = (torch.from_numpy(vis_pc_sc).float(),
                          torch.from_numpy(tac_pc_l_sc).float(),
                          torch.from_numpy(tac_pc_r_sc).float(),
                          torch.from_numpy(gt_pc).float())
            
            if self.return_vis_pc_align_original:
                return points['original'], points['shape_completion'], points['grasp_stability'], zero_mean, sample, vis_pc_align_original
            else:
                return points['original'], points['shape_completion'], points['grasp_stability'], zero_mean, sample
        else:
            points['grasp_stability'] = (
                torch.from_numpy(np.concatenate([tac_pc_l_align_with_gel, tac_pc_r_align_with_gel], axis=0)).float(),
                torch.tensor(grasp_result).float()
            )   
            points['shape_completion'] = (
                torch.from_numpy(np.concatenate([vis_pc_sc, tac_pc_l_sc, tac_pc_r_sc], axis=0)).float(),
                torch.from_numpy(gt_pc).float()
            )
            
            # if self.seg_robust_test and self.seg_robust_mode == 'under':
            #     # we do a ablation study on using pure under-segmented visual point cloud for grasp stability shape info
            #     vis_pc_sc_rs = vis_pc_sc
            #     if vis_pc_sc.shape[0] != self.sc_input_points:
            #         if vis_pc_sc.shape[0] == 0:
            #             vis_pc_sc_rs = np.zeros((self.sc_input_points, 3), dtype=np.float32)
            #         elif vis_pc_sc.shape[0] > self.sc_input_points:
            #             idx = np.random.choice(vis_pc_sc.shape[0], self.sc_input_points, replace=False)
            #             vis_pc_sc_rs = vis_pc_sc[idx]
            #         else:
            #             extra_idx = np.random.choice(
            #                 vis_pc_sc.shape[0],
            #                 self.sc_input_points - vis_pc_sc.shape[0],
            #                 replace=True,
            #             )
            #             vis_pc_sc_rs = np.concatenate([vis_pc_sc, vis_pc_sc[extra_idx]], axis=0)
            #     points['shape_completion'] = (
            #         torch.from_numpy(vis_pc_sc_rs).float(),
            #         torch.from_numpy(gt_pc).float()
            #     )
            # else:
            #     points['shape_completion'] = (torch.from_numpy(np.concatenate([vis_pc_sc, tac_pc_l_sc, tac_pc_r_sc], axis=0)).float(),
            #                               torch.from_numpy(gt_pc).float())
                
            
            if self.return_vis_pc_align_original:
                return points['shape_completion'], points['grasp_stability'], zero_mean, sample, vis_pc_align_original
            return points['shape_completion'], points['grasp_stability'], zero_mean, sample

    def __len__(self):
        return len(self.file_list)
