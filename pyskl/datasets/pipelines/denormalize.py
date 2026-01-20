import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class DeNormalizeKeypoints:
    """Convert normalized keypoints to original image shape."""    

    def __call__(self, results):
        h, w = results['img_shape']
        keypoint = results['keypoint']
        
        max_val = np.max(keypoint[..., :2])
        print("Max keypoint value BEFORE denormalization:", max_val)

        #multiply normalized keypoints with image width and heigth
        keypoint = keypoint.copy()
        keypoint[..., 0] *= w  #x
        keypoint[..., 1] *= h  #y

        max_val = np.max(keypoint[..., :2])
        print("Max keypoint value AFTER denormalization:", max_val)
        
        
        results['keypoint'] = keypoint
        return results
    