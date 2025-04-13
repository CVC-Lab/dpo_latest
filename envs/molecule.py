from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *
from envs.bbo import BBO
import tempfile
import os
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio

pyrosetta.init()
MAX_ABS = 1e18

### Generic continuous environment for reduced Hamiltonian dynamics framework
class Molecule(BBO):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }
    
    def __init__(self, pose, naive=False, reset_scale=1e-2, step_size=1e-1, max_num_step=6, render_mode='human'):
        # Superclass setup
        super(Molecule, self).__init__(naive, step_size, max_num_step)

        # Molecule info
        self.pose = pose
        self.num_residue = pose.total_residue()
        self.sfxn = get_fa_scorefxn() # Score function

        # State and action info
        self.state_dim = self.num_residue*2
        self.min_val = -180; self.max_val = 180
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.min_act = -90; self.max_act = 90 
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)      
        self.state = None

        # Reset scale
        self.reset_scale = reset_scale
        
        # Rendering
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.isopen = True
        
        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)
        
        # Create temporary directory for PyMOL images
        self.temp_dir = tempfile.mkdtemp()
        self.frame_count = 0
        
        # Check if PyMOL is available
        self.pymol_available = self._check_pymol_available()
    
    def _check_pymol_available(self):
        """Check if PyMOL is available on the system."""
        try:
            # Try to find the pymol executable using the 'which' command
            result = subprocess.run(["which", "pymol"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"PyMOL found at: {result.stdout.strip()}")
                return True
            else:
                print("PyMOL not found in PATH. Using matplotlib fallback.")
                return False
        except Exception as e:
            print(f"Error checking for PyMOL: {e}")
            return False
    
    def step(self, action):
        self.state += self.step_size * action
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        val = self.sfxn(self.pose)

        # Update number of step
        self.num_step += 1

        done = self.num_step >= self.max_num_step

        # Calculate final reward
        reward = self.calculate_final_reward(val, action)
        reward = np.clip(reward, -MAX_ABS, MAX_ABS)
        
        return np.array(self.state), reward, done, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        self.frame_count = 0
        return self.reset_at(mode='random'), {}
    
    def reset_at(self, mode='random'):
        if mode == 'random':
            self.state = self.reset_scale*(self.rng.random(self.state_dim)-.5)
            
        # Update pose with initial state
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])

        return np.array(self.state)

    def render(self):
        """Render the current state of the molecule"""
        # Update pose with current state
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        
        # Apply to PyMOL - this sends the current pose to PyMOL
        self.pmm.apply(self.pose)
        
        # Generate frame name
        frame_path = os.path.join(self.temp_dir, f'frame_{self.frame_count:04d}')
        
        # Try to use PyMOL for rendering if available
        if self.pymol_available:
            try:
                self.save_image_pymol(frame_path)
                img_path = f"{frame_path}.png"
                if os.path.exists(img_path):
                    img = imageio.imread(img_path)
                    # Convert to RGB if needed (remove alpha channel)
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    self.frame_count += 1
                    return img
            except Exception as e:
                print(f"PyMOL rendering failed: {e}")
                print("Falling back to matplotlib rendering")
                self.pymol_available = False
        
        # Fallback to matplotlib rendering
        return self.generate_image_matplotlib()
    
    def save_image_pymol(self, filename):
        """Save current pose as an image using PyMOL"""
        # Save the current pose to a PDB file
        pdb_filename = f"{filename}.pdb"
        self.pose.dump_pdb(pdb_filename)
        
        # Use PyMOL command to save image with ball-and-stick representation
        cmd = f"""
        import pymol
        pymol.finish_launching()
        cmd.load("{pdb_filename}", "molecule")
        cmd.show("sticks", "molecule")
        cmd.show("spheres", "molecule")
        cmd.set("sphere_scale", 0.3, "molecule")
        cmd.set("stick_radius", 0.1, "molecule")
        cmd.set("ray_shadows", 0)
        cmd.bg_color("white")
        cmd.png("{filename}.png", width={self.screen_width}, height={self.screen_height}, dpi=100, ray=1)
        cmd.quit()
        """
        
        # Look for PyMOL executable in different possible locations
        pymol_paths = ["pymol", "/usr/local/bin/pymol", "/opt/homebrew/bin/pymol"]
        success = False
        
        for pymol_path in pymol_paths:
            try:
                subprocess.run([pymol_path, "-cq", "-d", cmd], check=True, capture_output=True)
                success = True
                break
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Failed to run PyMOL at {pymol_path}: {e}")
        
        if not success:
            raise RuntimeError("Failed to run PyMOL using any of the available paths")
    
    def generate_image_matplotlib(self):
        """Generate a matplotlib-based visualization of the molecule"""
        # Get coordinates of all atoms
        coords = []
        colors = []
        
        for i in range(1, self.pose.total_residue() + 1):
            residue = self.pose.residue(i)
            for j in range(1, residue.natoms() + 1):
                coords.append(residue.atom(j).xyz())
                
                # Color by atom type
                atom_name = residue.atom_name(j).strip()
                if atom_name.startswith('C'):
                    colors.append('gray')
                elif atom_name.startswith('N'):
                    colors.append('blue')
                elif atom_name.startswith('O'):
                    colors.append('red')
                elif atom_name.startswith('S'):
                    colors.append('yellow')
                elif atom_name.startswith('P'):
                    colors.append('orange')
                else:
                    colors.append('green')
        
        if not coords:
            # Fallback if no coordinates are available
            img = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
            return img
            
        coords = np.array(coords)
        
        # Create figure with specific dimensions to match screen size
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms as points
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=30, c=colors, alpha=0.8)
        
        # Add bonds between nearby atoms (simplified approach)
        # In a real implementation, you would use bond information from PyRosetta
        max_bond_distance = 2.0  # Å, typical covalent bond length
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                # Calculate distance between atoms
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < max_bond_distance:
                    ax.plot([coords[i, 0], coords[j, 0]],
                           [coords[i, 1], coords[j, 1]],
                           [coords[i, 2], coords[j, 2]], 'k-', alpha=0.5, linewidth=2)
        
        # Set limits and labels
        max_range = np.max(np.array([
            coords[:, 0].max() - coords[:, 0].min(),
            coords[:, 1].max() - coords[:, 1].min(),
            coords[:, 2].max() - coords[:, 2].min()
        ]))
        
        mid_x = (coords[:, 0].max() + coords[:, 0].min()) / 2
        mid_y = (coords[:, 1].max() + coords[:, 1].min()) / 2
        mid_z = (coords[:, 2].max() + coords[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Remove background grid for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        
        # Set title with score information
        score = self.sfxn(self.pose)
        ax.set_title(f"Score: {score:.2f}")
        
        # Ensure tight layout
        fig.tight_layout()
        
        # Save to a temporary file and read back as RGB
        frame_path = os.path.join(self.temp_dir, f'frame_{self.frame_count:04d}.png')
        plt.savefig(frame_path, format='png', dpi=100, transparent=False)
        plt.close(fig)
        
        # Read the image and ensure it's RGB
        img = imageio.imread(frame_path)
        # Convert to RGB if needed (remove alpha channel)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        
        self.frame_count += 1
        return img
     
    def close(self):
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
        self.isopen = False