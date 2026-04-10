import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)
# sys.path.append('/raid/dtle/deepfake/DeepfakeBench/training/networks')

from metrics.registry import BACKBONE

from .dinov2    import DINOv2
from .dinov2_clip_fusion import DINOv2_CLIP_Fusion