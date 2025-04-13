from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
from gymnasium import spaces
import pygame
from pygame import gfxdraw
from envs.bbo import BBO
import cv2
import tempfile
import os

MAX_ACT = 1e4

class ShapeBoundary(BBO):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(self, naive=False, step_size=1e-2, state_dim=16, max_num_step=20, render_mode='rgb_array'):
        # Superclass setup
        super(ShapeBoundary, self).__init__(naive, step_size, max_num_step)

        # State and action info
        self.state_dim = state_dim
        self.min_val = -4; self.max_val = 4
        self.observation_space = spaces.box.Box(low=self.min_val, high=self.max_val, shape=(state_dim,), dtype=np.float32)
        self.min_act = -1; self.max_act = 1
        self.action_space = spaces.box.Box(low=self.min_act, high=self.max_act, shape=(state_dim,), dtype=np.float32)
        self.state = None
    
        # Geometry
        self.num_coef = self.state_dim//2
        self.ts = np.linspace(0, 1, 80)
        self.verts = None
        self.current_value = None
        self.current_area = None
        self.current_perimeter = None

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 700  # Increased to accommodate info
        self.screen = None
        self.clock = None
        self.isopen = True
        self.frame_count = 0
        
        # Create temporary directory for frames
        self.temp_dir = tempfile.mkdtemp()

    def step(self, action):
        self.state += self.step_size * action
        
        # Use cubic spline to smooth out the new state parametric curve
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        polygon = Polygon(zip(coords[:,0], coords[:,1]))
        coords = coords/np.max(np.abs(coords))*100 + 300
        self.verts = list(zip(coords[:,0], coords[:,1]))
        done = (polygon.area == 0 or polygon.length == 0)
        
        # Update number of step
        self.num_step += 1

        # Calculate value
        if not done:
            self.current_area = polygon.area
            self.current_perimeter = polygon.length
            val = polygon.length/np.sqrt(polygon.area)
            self.current_value = val
            done = self.num_step >= self.max_num_step
        else:
            val = 1e9
            self.current_value = val
            self.current_area = 0
            self.current_perimeter = 0

        # Calculate final reward
        reward = self.calculate_final_reward(val, action)
            
        return np.array(self.state), reward, done, False, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        self.frame_count = 0

        return self.reset_at(mode='half_random'), {}
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        self.state = np.zeros(self.state_dim)
        t = np.arange(self.num_coef)/self.num_coef
        if mode == 'ellipse':
            self.state[:self.num_coef] = 0.2*np.sin(2*np.pi*t)
            self.state[self.num_coef:] = np.cos(2*np.pi*t)
        elif mode == 'rect':
            # Assume n%4 == 0
            n = self.num_coef//4
            # x-coord
            self.state[:n] = np.arange(n)/n
            self.state[n:2*n] = 1
            self.state[2*n:3*n] = 1 - (np.arange(n)/n)
            self.state[3*n:4*n] = 0
            # y-coord
            self.state[4*n:5*n] = 0
            self.state[5*n:6*n] = np.arange(n)/n
            self.state[6*n:7*n] = 1
            self.state[7*n:8*n] = 1 - (np.arange(n)/n)
        elif mode == 'half_random':
            # Assume n%2 == 0
            n = self.num_coef//2
            # x-coord
            self.state[0:n] = 0.8*self.rng.random(n) + 0.2
            self.state[n:2*n] = -0.8*self.rng.random(n) - 0.2

            # y-coord
            self.state[2*n:3*n] = np.arange(n)/n
            self.state[3*n:4*n] = np.arange(n)/n
        elif mode == 'random':
            self.state = self.rng.random(self.state_dim) - 0.5
            
        cs = CubicSpline(np.linspace(0,1,self.num_coef), self.state.reshape(2, self.num_coef).T)
        coords = cs(self.ts)
        
        # Initialize current metrics
        polygon = Polygon(zip(coords[:,0], coords[:,1]))
        self.current_area = polygon.area
        self.current_perimeter = polygon.length
        self.current_value = polygon.length/np.sqrt(polygon.area) if polygon.area > 0 else 1e9
        
        coords = coords/np.max(np.abs(coords))*100 + 300
        self.verts = list(zip(coords[:,0], coords[:,1]))
        return np.array(self.state)
    
    def render(self):
        """Enhanced rendering with additional information using cv2 instead of pygame fonts"""
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode is "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Base image - larger to include info section
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        
        # Draw the shape (in the top part only)
        shape_surface = pygame.Surface((self.screen_width, self.screen_width))
        shape_surface.fill((255, 255, 255))
        gfxdraw.aapolygon(shape_surface, self.verts, (0, 0, 0))
        gfxdraw.filled_polygon(shape_surface, self.verts, (0, 0, 255))  # Fill with blue
        
        # Draw the outline with a different color
        gfxdraw.aapolygon(shape_surface, self.verts, (255, 0, 0))  # Red outline
        
        # Flip for correct orientation
        shape_surface = pygame.transform.flip(shape_surface, False, True)
        
        # Blit shape to main surface
        self.surf.blit(shape_surface, (0, 0))
        
        # Blit to screen
        self.screen.blit(self.surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        
        # For rgb_array mode - convert to numpy array and add text with cv2
        if self.render_mode == "rgb_array":
            rgb_array = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
            # Add text with cv2 instead of pygame font
            info_texts = [
                f"Step: {self.num_step}/{self.max_num_step}",
                f"Isoperimetric Quotient: {self.current_value:.4f}",
                f"Area: {self.current_area:.2f}",
                f"Perimeter: {self.current_perimeter:.2f}"
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 0, 0)  # Black
            thickness = 2
            
            for i, text in enumerate(info_texts):
                position = (20, self.screen_width + 30 + i * 30)
                cv2.putText(rgb_array, text, position, font, font_scale, font_color, thickness)
            
            self.frame_count += 1
            return rgb_array
     
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
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