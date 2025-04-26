import sys
pyngp_path = "/content/instant-ngp/build/"

sys.path.append(pyngp_path)
import pyngp as ngp  # noqa
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class ngp_render():
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.testbed = ngp.Testbed()
        self.testbed.load_snapshot(weight_path)
        self.screenshot_spp = 32
        self.resolution = None
        self.flip_mat = np.array([
                                    [1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]
                                ])

    def load_snapshot(self, snapshot_path):
        self.testbed.load_snapshot(snapshot_path)

    def set_renderer_mode(self, mode):
        if mode == 'Depth':
            self.testbed.render_mode = ngp.RenderMode.Depth
        elif mode == 'Normals':
            self.testbed.render_mode = ngp.RenderMode.Normals
        elif mode == 'Shade':
            self.testbed.render_mode = ngp.RenderMode.Shade

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_fov(self, K):
        fov_x = np.degrees(2 * np.arctan2(self.resolution[0], 2 * K[0,0]))
        fov_y = np.degrees(2 * np.arctan2(self.resolution[1], 2 * K[1,1]))
        self.testbed.screen_center = np.array([1-(K[0,2]/self.resolution[0]), 1-(K[1,2] /self.resolution[1])])
        self.testbed.fov_xy = np.array([fov_y, fov_y])

    def set_exposure(self, exposure):
        self.testbed.exposure = exposure

    def get_image_from_tranform(self, mode):
        self.set_renderer_mode(mode)
        image = self.testbed.render(self.resolution[0], self.resolution[1], self.screenshot_spp, True)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = np.array(image) * 255.0
        return image

    def get_image_raw(self, mode):
        self.set_renderer_mode(mode)
        image = self.testbed.render(self.resolution[0], self.resolution[1], self.screenshot_spp, True)
        return image

    def set_camera_matrix(self,Extrinsics):
        camera_matrix = Extrinsics[:3, :4]
        self.testbed.set_nerf_camera_matrix(camera_matrix)
