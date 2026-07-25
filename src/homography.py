import cv2
import numpy as np


def _to_np_points(pts):
    arr = np.asarray(pts, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError(f"Expected shape (4,2), got {arr.shape}")
    return arr


class HomographyProjector:
    def __init__(self, H_img2gnd, H_gnd2img):
        self.H_img2gnd = np.asarray(H_img2gnd, dtype=np.float64)
        self.H_gnd2img = np.asarray(H_gnd2img, dtype=np.float64)

    @classmethod
    def from_points(cls, image_points, ground_points):
        img = _to_np_points(image_points)
        gnd = _to_np_points(ground_points)
        H_img2gnd = cv2.getPerspectiveTransform(img, gnd)
        H_gnd2img = cv2.getPerspectiveTransform(gnd, img)
        return cls(H_img2gnd, H_gnd2img)

    def image_to_ground(self, x, y):
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)  # (1,1,2)
        out = cv2.perspectiveTransform(p, self.H_img2gnd)[0, 0]
        return float(out[0]), float(out[1])

    def ground_to_image(self, x, y):
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)  # (1,1,2)
        out = cv2.perspectiveTransform(p, self.H_gnd2img)[0, 0]
        return float(out[0]), float(out[1])
