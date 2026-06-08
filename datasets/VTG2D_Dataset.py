import torch.utils.data as data
import numpy as np
import os, sys, csv, json
from tqdm import tqdm
from multiprocessing import Pool
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
import cv2
import random
from datasets.build import DATASETS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def _load_metadata(args):
    data_root, obj_id = args  # Unpack the tuple into two variables
    json_path = os.path.join(data_root, obj_id, '_metadata.json')
    with open(json_path, 'r') as json_file:
        return obj_id, json.load(json_file)

@DATASETS.register_module()
class VTG2D(data.Dataset):
    def __init__(self, config, debug=False):
        # load config
        self.data_root = config.DATA_PATH
        self.vis_rgb_pt = config.VIS_RGB_PATH
        self.vis_seg_pt = config.VIS_SEG_PATH
        self.tac_rgb_pt = config.TAC_RGB_PATH
        self.subset = config.subset
        self.subset_id_file = os.path.join(self.data_root, f'{self.subset}-ids.txt')
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.csv')
        self.debug = debug
        self.gelbackground = config.BACKGROUND_PATH
        self.concat_two_tactile = config.ConcatTwoTactile if hasattr(config, 'ConcatTwoTactile') else False
        
        # Load and cache the single tactile background image
        self._load_tactile_background()
        
        # Image processing parameters
        self.crop_size = 256
        self.final_size = 224
        
        print(f'[DATASET] Open file {self.data_list_file} to load dataset indice list')
        self.file_list = []
        with open(self.data_list_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row if it exists
            for row in reader:
                self.file_list.append(tuple(row))

        print(f'[DATASET] Open file {self.subset_id_file} to get id list')
        with open(self.subset_id_file, 'r') as file:
            obj_ids = [line.strip().zfill(3) for line in file.readlines()]
        
        print(f'[DATASET] Loading metadata.json of all object')
        
        self.metadata_dict = {}
        with Pool(processes=6) as pool:
            obj_id_pairs = [(self.data_root, obj_id) for obj_id in obj_ids]
            result = pool.imap_unordered(_load_metadata, obj_id_pairs)
            for obj_id, metadata in tqdm(result, total=len(obj_id_pairs), desc='Loading metadata'):
                self.metadata_dict[obj_id] = metadata
        
        # Define data augmentation transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup data augmentation transforms
        
        SYNCHRONIZED TRANSFORM STRATEGY:
        
        1. Visual RGB-Visual SEG: Must fully synchronize geometric transforms
        2. Left tactile background-Left tactile grasp: Synchronize geometric transforms
        3. Right tactile background-Right tactile grasp: Synchronize geometric transforms
        
        Preserve ColorJitter and GaussianBlur for color/quality enhancement
        """
        self.crop_size = 256
        self.final_size = 224
        
        if self.subset == 'train':
            # We will apply synchronized transforms in the __getitem__ method
            pass
        else:
            # Validation/test transforms: direct resize to final_size, no cropping
            self.vis_transform = transforms.Compose([
                transforms.Resize((self.final_size, self.final_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.vis_seg_transform = transforms.Compose([
                transforms.Resize((self.final_size, self.final_size)),
                transforms.ToTensor()
            ])
            
            self.tac_transform = transforms.Compose([
                transforms.Resize((self.final_size, self.final_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _apply_synchronized_transforms_vis(self, rgb_img, seg_img):
        """Apply synchronized geometric transforms to RGB and segmentation images
        
        Ensure RGB and segmentation images use the same geometric transform parameters 
        to maintain pixel-level correspondence
        """
        if self.subset != 'train':
            # Validation/test: use simple resize
            rgb_transformed = self.vis_transform(rgb_img)
            seg_transformed = self.vis_seg_transform(seg_img)
            return rgb_transformed, seg_transformed
        
        # Training: apply synchronized transforms
        # 1. Resize (deterministic transform)
        rgb_img = transforms.Resize((self.crop_size, self.crop_size))(rgb_img)
        seg_img = transforms.Resize((self.crop_size, self.crop_size))(seg_img)
        
        # 2. RandomCrop (generate parameters once, use same for both images)
        i, j, h, w = transforms.RandomCrop.get_params(rgb_img, output_size=(self.final_size, self.final_size))
        rgb_img = transforms.functional.crop(rgb_img, i, j, h, w)
        seg_img = transforms.functional.crop(seg_img, i, j, h, w)
        
        # 3. RandomHorizontalFlip (generate random once, use same decision for both images)
        if random.random() < 0.5:
            rgb_img = transforms.functional.hflip(rgb_img)
            seg_img = transforms.functional.hflip(seg_img)
        
        # 4. RandomRotation (generate angle once, use same angle for both images)
        angle = transforms.RandomRotation.get_params(degrees=[-10, 10])
        rgb_img = transforms.functional.rotate(rgb_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        seg_img = transforms.functional.rotate(seg_img, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        # 5. RGB-specific transforms (ColorJitter, GaussianBlur)
        # ColorJitter
        if random.random() < 0.8:  # 80% probability to apply ColorJitter
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            rgb_img = transforms.functional.adjust_brightness(rgb_img, brightness)
            rgb_img = transforms.functional.adjust_contrast(rgb_img, contrast)
            rgb_img = transforms.functional.adjust_saturation(rgb_img, saturation)
            rgb_img = transforms.functional.adjust_hue(rgb_img, hue)
        
        # GaussianBlur
        if random.random() < 0.3:  # 30% probability to apply Gaussian blur
            rgb_img = transforms.GaussianBlur(kernel_size=3, sigma=random.uniform(0.1, 2.0))(rgb_img)
        
        # 6. Final conversion to tensor and normalization
        rgb_tensor = transforms.ToTensor()(rgb_img)
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb_tensor)
        
        seg_tensor = transforms.ToTensor()(seg_img)
        
        return rgb_tensor, seg_tensor
    
    def _apply_synchronized_transforms_tactile(self, bg_img, grasp_img):
        """Apply synchronized geometric transforms to tactile background and grasp images
        
        Ensure tactile background and grasp images use the same geometric transform parameters
        """
        if self.subset != 'train':
            # Validation/test: use simple resize
            bg_transformed = self.tac_transform(bg_img)
            grasp_transformed = self.tac_transform(grasp_img)
            return bg_transformed, grasp_transformed
        
        # Training: apply synchronized transforms
        # 1. Resize (deterministic transform)
        bg_img = transforms.Resize((self.crop_size, self.crop_size))(bg_img)
        grasp_img = transforms.Resize((self.crop_size, self.crop_size))(grasp_img)
        
        # 2. RandomCrop (generate parameters once, use same for both images)
        i, j, h, w = transforms.RandomCrop.get_params(bg_img, output_size=(self.final_size, self.final_size))
        bg_img = transforms.functional.crop(bg_img, i, j, h, w)
        grasp_img = transforms.functional.crop(grasp_img, i, j, h, w)
        
        # 3. RandomHorizontalFlip (generate random once, use same decision for both images)
        if random.random() < 0.5:
            bg_img = transforms.functional.hflip(bg_img)
            grasp_img = transforms.functional.hflip(grasp_img)
        
        # 4. RandomRotation (generate angle once, use same angle for both images, smaller angle for tactile)
        angle = transforms.RandomRotation.get_params(degrees=[-5, 5])
        bg_img = transforms.functional.rotate(bg_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        grasp_img = transforms.functional.rotate(grasp_img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        
        # 5. Tactile-specific light ColorJitter (synchronized application)
        if random.random() < 0.6:  # 60% probability to apply ColorJitter
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.9, 1.1)
            hue = random.uniform(-0.05, 0.05)
            
            bg_img = transforms.functional.adjust_brightness(bg_img, brightness)
            bg_img = transforms.functional.adjust_contrast(bg_img, contrast)
            bg_img = transforms.functional.adjust_saturation(bg_img, saturation)
            bg_img = transforms.functional.adjust_hue(bg_img, hue)
            
            grasp_img = transforms.functional.adjust_brightness(grasp_img, brightness)
            grasp_img = transforms.functional.adjust_contrast(grasp_img, contrast)
            grasp_img = transforms.functional.adjust_saturation(grasp_img, saturation)
            grasp_img = transforms.functional.adjust_hue(grasp_img, hue)
        
        # 6. GaussianBlur (synchronized application)
        if random.random() < 0.2:  # 20% probability to apply Gaussian blur
            sigma = random.uniform(0.1, 1.0)
            bg_img = transforms.GaussianBlur(kernel_size=3, sigma=sigma)(bg_img)
            grasp_img = transforms.GaussianBlur(kernel_size=3, sigma=sigma)(grasp_img)
        
        # 7. Final conversion to tensor and normalization
        bg_tensor = transforms.ToTensor()(bg_img)
        bg_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(bg_tensor)
        
        grasp_tensor = transforms.ToTensor()(grasp_img)
        grasp_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(grasp_tensor)
        
        return bg_tensor, grasp_tensor
    
    def _crop_table_region(self, image, metadata):
        """
        Crop the image with a bounding box containing the table that holds the objects,
        padded vertically to provide a view of the robot's gripper.
        """
        # This is a placeholder implementation - you may need to adjust based on your metadata
        # or implement table detection logic
        height, width = image.shape[:2]
        
        # Example cropping logic - adjust based on your specific setup
        # Assuming the table is in the lower portion of the image
        crop_top = int(height * 0.2)  # Start from 20% of image height
        crop_bottom = height  # Go to bottom
        crop_left = int(width * 0.1)  # 10% margin from left
        crop_right = int(width * 0.9)  # 10% margin from right
        
        cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
        return cropped_image
    
    def _load_rgb_image(self, image_path):
        """Load RGB image and convert to PIL format"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        return image
    
    def _load_tactile_background(self):
        """Load and cache the single tactile background image"""
        print(f'[DATASET] Loading tactile background image')
        self.cached_tac_bg = self._load_rgb_image(self.gelbackground)
        print(f'[DATASET] Tactile background image cached successfully')
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation to input data and labels
        
        Mixup is a powerful regularization technique that creates virtual training 
        examples by mixing pairs of examples and their labels. This should be 
        applied in the training loop, not in the dataset preprocessing.
        
        Usage in training loop:
            if use_mixup:
                inputs, targets_a, targets_b, lam = dataset.mixup_data(inputs, targets, alpha=1.0)
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
        Args:
            x: Input batch tensor
            y: Target batch tensor  
            alpha: Mixup interpolation strength (1.0 recommended)
            
        Returns:
            mixed_x: Mixed input batch
            y_a, y_b: Original target pairs
            lam: Mixing coefficient
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def _concatenate_images_horizontal(self, img1, img2):
        """
        Concatenate two PIL images horizontally (side by side)
        
        Args:
            img1: Left PIL image
            img2: Right PIL image
            
        Returns:
            concatenated_img: PIL image with img1 on left and img2 on right
        """
        # Ensure both images have the same height
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Use the minimum height to avoid distortion
        target_height = min(height1, height2)
        
        # Resize both images to the same height if needed
        if height1 != target_height:
            img1 = img1.resize((int(width1 * target_height / height1), target_height), Image.LANCZOS)
        if height2 != target_height:
            img2 = img2.resize((int(width2 * target_height / height2), target_height), Image.LANCZOS)
        
        # Get new dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Create new image with combined width
        concat_width = width1 + width2
        concat_img = Image.new('RGB', (concat_width, target_height))
        
        # Paste images side by side
        concat_img.paste(img1, (0, 0))
        concat_img.paste(img2, (width1, 0))
        
        return concat_img
    
    def __getitem__(self, idx):
        ### Load data
        sample = self.file_list[idx]
        obj_id, grasp_id = sample
        metadata = self.metadata_dict[obj_id][grasp_id]
        
        ### Load visual RGB image
        vis_rgb_path = self.vis_rgb_pt % (obj_id, grasp_id)
        vis_rgb = self._load_rgb_image(vis_rgb_path)
        
        ### Load visual segmentation image
        vis_seg_path = self.vis_seg_pt % (obj_id, grasp_id)
        vis_seg = self._load_rgb_image(vis_seg_path)
        
        ### Apply synchronized transforms to RGB and segmentation images
        vis_rgb_tensor, vis_seg_tensor = self._apply_synchronized_transforms_vis(vis_rgb, vis_seg)
        
        ### Load tactile RGB images (left and right)
        tac_rgb_l_path = self.tac_rgb_pt % (obj_id, f'{grasp_id}_l')
        tac_rgb_r_path = self.tac_rgb_pt % (obj_id, f'{grasp_id}_r')
        tac_rgb_l = self._load_rgb_image(tac_rgb_l_path)
        tac_rgb_r = self._load_rgb_image(tac_rgb_r_path)
        
        ### Load tactile background image (shared for all grasps and both sensors)
        tac_bg = self.cached_tac_bg  # Use cached background image
        
        ### Branch: Handle tactile processing based on concatenation setting
        if self.concat_two_tactile:
            # Concatenate left and right tactile images horizontally
            tac_rgb_concat = self._concatenate_images_horizontal(tac_rgb_l, tac_rgb_r)
            tac_bg_concat = self._concatenate_images_horizontal(tac_bg, tac_bg)  # Duplicate background
            
            # Apply synchronized transforms to concatenated tactile pair (only once)
            tac_bg_concat_tensor, tac_rgb_concat_tensor = self._apply_synchronized_transforms_tactile(tac_bg_concat, tac_rgb_concat)
            
        else:
            # Apply synchronized transforms to tactile pairs separately (original behavior)
            # Left tactile background-left tactile grasp synchronized transforms
            tac_bg_l_tensor, tac_rgb_l_tensor = self._apply_synchronized_transforms_tactile(tac_bg, tac_rgb_l)
            
            # Right tactile background-right tactile grasp synchronized transforms (apply new random transforms)
            tac_bg_r_tensor, tac_rgb_r_tensor = self._apply_synchronized_transforms_tactile(tac_bg, tac_rgb_r)
        
        ### Get grasp stability result
        grasp_result = metadata['isPositive']
        grasp_result_tensor = torch.tensor(grasp_result).float()
        
        ### Return data based on concatenation setting and debug mode
        if self.debug:
            if self.concat_two_tactile:
                return {
                    'visual_rgb': vis_rgb_tensor,
                    'visual_seg': vis_seg_tensor,
                    'tactile_rgb_concat': tac_rgb_concat_tensor,  # Concatenated tactile
                    'tactile_bg_concat': tac_bg_concat_tensor,    # Concatenated background
                    'grasp_result': grasp_result_tensor,
                    'sample_info': sample,
                    'metadata': metadata
                }
            else:
                return {
                    'visual_rgb': vis_rgb_tensor,
                    'visual_seg': vis_seg_tensor,
                    'tactile_rgb_left': tac_rgb_l_tensor,
                    'tactile_rgb_right': tac_rgb_r_tensor,
                    'tactile_bg_left': tac_bg_l_tensor,
                    'tactile_bg_right': tac_bg_r_tensor,
                    'grasp_result': grasp_result_tensor,
                    'sample_info': sample,
                    'metadata': metadata
                }
        else:
            if self.concat_two_tactile:
                return {
                    'visual_rgb': vis_rgb_tensor,
                    'visual_seg': vis_seg_tensor,
                    'tactile_rgb_concat': tac_rgb_concat_tensor,  # Concatenated tactile
                    'tactile_bg_concat': tac_bg_concat_tensor,    # Concatenated background 
                    'grasp_result': grasp_result_tensor,
                    'sample_info': sample
                }
            else:
                return {
                    'visual_rgb': vis_rgb_tensor,
                    'visual_seg': vis_seg_tensor,
                    'tactile_rgb_left': tac_rgb_l_tensor,
                    'tactile_rgb_right': tac_rgb_r_tensor,
                    'tactile_bg_left': tac_bg_l_tensor,
                    'tactile_bg_right': tac_bg_r_tensor,
                    'grasp_result': grasp_result_tensor,
                    'sample_info': sample
                }
    
    def __len__(self):
        return len(self.file_list)
    
    def get_sample_info(self, idx):
        """Get sample information without loading images (useful for debugging)"""
        sample = self.file_list[idx]
        obj_id, grasp_id = sample
        metadata = self.metadata_dict[obj_id][grasp_id]
        
        return {
            'obj_id': obj_id,
            'grasp_id': grasp_id,
            'grasp_result': metadata['isPositive'],
            'vis_rgb_path': self.vis_rgb_pt % (obj_id, grasp_id),
            'vis_seg_path': self.vis_seg_pt % (obj_id, grasp_id),
            'tac_rgb_l_path': self.tac_rgb_pt % (obj_id, f'{grasp_id}_l'),
            'tac_rgb_r_path': self.tac_rgb_pt % (obj_id, f'{grasp_id}_r'),
            'tac_bg_path': self.gelbackground  # Single background path for all samples
        }
