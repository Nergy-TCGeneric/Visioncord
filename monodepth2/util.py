import numpy as np
import matplotlib
import matplotlib.cm as cm
import PIL.Image

FOCAL_LENGTH_IN_MM = 3.04
# Experimental value to adjust estimated depth
BASELINE_IN_MM = 1120 

# https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/layers.py#L16
def estimate_distance_from_disp(disp: np.ndarray, min_depth: float = 0.1, max_disp: float = 100) -> np.ndarray:
    # CAVEAT: monodepth2 is trained using KITTI dataset to yield consistent results.
    # Indeed, the disp is not metric but a relative one.
    min_disp = 1 / max_disp
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp * FOCAL_LENGTH_IN_MM * BASELINE_IN_MM

def save_disparity_map_to_image(disp: np.ndarray, image_name: str) -> None:
    vmax = np.percentile(disp, 95)
    normalizer = matplotlib.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    im = PIL.Image.fromarray(colormapped_im)
    im.save(f'{image_name}.jpg')