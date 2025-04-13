from typing import Optional
import numpy as np
from scipy import interpolate
import cv2
from gymnasium import spaces
from envs.bbo import BBO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
import os
import tempfile

ImgDim = namedtuple('ImgDim', 'width height')

class Shape(BBO):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, naive=False, step_size=1e-2, state_dim=64, max_num_step=20, render_mode='rgb_array'):
        # Superclass setup
        super(Shape, self).__init__(naive, step_size, max_num_step)

        # State and action info
        self.state_dim = state_dim
        self.max_val = 4; self.min_val = -4
        self.max_act = 1; self.min_act = -1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.state = None

        # Shape interpolation info
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        self.xg, self.yg = np.mgrid[-1:1:50j, -1:1:50j]
        self.viewer = ImgDim(width=self.xg.shape[0], height=self.yg.shape[1])
        
        # Rendering parameters
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 600
        self.frame_count = 0
        self.isopen = True
        
        # Create temporary directory for frames
        self.temp_dir = tempfile.mkdtemp()
        
        # Track current value
        self.current_value = None
        self.current_area = None
        self.current_perimeter = None

    def step(self, action):
        self.state += self.step_size * action

        # Update number of step
        self.num_step += 1

        # Calculate value
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        done = (area == 0 or peri == 0)
        if not done:
            val = peri/np.sqrt(area)
            self.current_value = val
            self.current_area = area
            self.current_perimeter = peri
            done = self.num_step >= self.max_num_step
        else:
            val = 1e9
            self.current_value = val
            self.current_area = 0
            self.current_perimeter = 0

        reward = self.calculate_final_reward(val, action)
        return np.array(self.state), reward, done, False, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        self.frame_count = 0
        return self.reset_at(mode='random'), {}
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        width = int(np.sqrt(self.state_dim))
        self.state = np.ones((width, width))
        if mode=='hole':
            self.state[1:8, 1:8] = 0
            self.state += self.rng.random((width, width))
            self.state = np.clip(self.state, 0, 1)
        elif mode=='random':
            self.state = self.rng.random((width, width))
        elif mode=='random_with_padding':
            # Random with zero padding
            self.state[1:(width-1), :(width-1)] = self.rng.random((width-2, width-1))
        self.state -= .5
        self.state = self.state.reshape(-1)
        
        # Initialize current value
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        self.current_value = peri/np.sqrt(area) if area > 0 and peri > 0 else 1e9
        self.current_area = area
        self.current_perimeter = peri
        
        return np.array(self.state)
    
    def render(self):
        """Enhanced rendering with additional information"""
        # Get the basic shape image
        basic_img = 255-spline_interp(self.state.reshape(self.xk.shape[0], self.yk.shape[0]), 
                                      self.xk, self.yk, self.xg, self.yg)
        
        # Create a larger canvas for the enhanced visualization
        canvas_height = 700  # Increased height to accommodate information
        canvas = np.ones((canvas_height, self.screen_width, 3), dtype=np.uint8) * 255
        
        # Calculate current shape metrics
        area, perimeter = geometry_info_from_img(basic_img)
        isoperimetric_quotient = perimeter/np.sqrt(area) if area > 0 and perimeter > 0 else float('inf')
        
        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor(basic_img, cv2.COLOR_GRAY2BGR)
        
        # Add colored contours
        contours, _ = cv2.findContours(basic_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)  # Red contours
        
        # Resize to fit our canvas width if needed
        if img_rgb.shape[1] != self.screen_width:
            img_rgb = cv2.resize(img_rgb, (self.screen_width, self.screen_width))
        
        # Place the shape image on the canvas
        canvas[:img_rgb.shape[0], :img_rgb.shape[1]] = img_rgb
        
        # Create info text
        info_texts = [
            f"Step: {self.num_step}/{self.max_num_step}",
            f"Isoperimetric Quotient: {self.current_value:.4f}",
            f"Area: {self.current_area:.2f}",
            f"Perimeter: {self.current_perimeter:.2f}"
        ]
        
        # Add text to the bottom of the canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 0, 0)  # Black
        thickness = 2
        line_height = 30
        
        for i, text in enumerate(info_texts):
            position = (20, img_rgb.shape[0] + 30 + i * line_height)
            cv2.putText(canvas, text, position, font, font_scale, font_color, thickness)
        
        # Save frame
        self.frame_count += 1
        
        return canvas
    
    def close(self):
        self.isopen = False
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            for frame_file in os.listdir(self.temp_dir):
                try:
                    os.remove(os.path.join(self.temp_dir, frame_file))
                except:
                    pass
            try:
                os.rmdir(self.temp_dir)
            except:
                pass


## Helper functions ##
# Spline interpolation for 2D density problem
def spline_interp(z, xk, yk, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    img = np.array(zint*C+C).astype('uint8')
    # Thresholding give binary image, which gives better contour
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def geometry_info_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    
    return area, peri

def geometry_info(z, xk, yk, xg, yg):
    img = spline_interp(z, xk, yk, xg, yg)
    return geometry_info_from_img(img)