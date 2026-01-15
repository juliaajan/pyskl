import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class KeypointTo2D:
    """Convert 3D keypoints (x, y, z) to 2D keypoints (x, y)."""
    
    def __call__(self, results):
        if 'keypoint' in results:
            kp = results['keypoint']
            if kp.shape[-1] == 3:  #if keypoint has 3 coordinates (x,y,z), remove z
                results['keypoint'] = kp[..., :2]
        return results