import numpy as np
import imageio
import os

# Run num_traj trajectories using policy given by model on env.
def test_model_through_vals(seeds, env, model, num_traj, num_step_per_traj,
                                 benchmark_model=False):
    # List of all vals across seed & trajectory of shape (seed, num_traj, num_step_per_traj).
    all_vals = [[] for _ in range(len(seeds))]
    for i, seed in enumerate(seeds):
        env.rng = np.random.default_rng(seed=seed)
        for _ in range(num_traj):
            obs, _ = env.reset() # obs same as state in our case.
            num_iteration = 0
            action = np.zeros(obs.shape)
            vals_cur_traj = []
            for _ in range(num_step_per_traj):
                if benchmark_model:
                    action, _ = model.predict(obs)
                else:
                    action = model.get_action(obs, action)
                obs, reward, done, _, _ = env.step(action)
                val = env.get_val(reward, action)
                vals_cur_traj.append(val)
                num_iteration += 1
                if done:
                    break
            all_vals[i].append(np.array(vals_cur_traj))
    all_vals = np.array(all_vals)

    return all_vals.reshape(-1, num_step_per_traj)


# Visualize a particular trajectories from given model's policy.
def visualize(env, model, num_step=100, benchmark_model=False,
              extra_args='random', img_path=None):
    vals = []; images = []
    num_iteration = 0
    obs = env.reset_at(mode=extra_args)
    action = np.zeros(obs.shape)
    
    # Create output directory for individual frames
    if img_path is not None:
        # Extract base path without extension
        base_path = img_path.rsplit('.', 1)[0]
        frames_dir = f"{base_path}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving individual frames to: {frames_dir}")
    
    for step in range(num_step):
        if benchmark_model:
            action, _ = model.predict(obs)
        else:
            action = model.get_action(obs, action)
        obs, reward, done, _, _ = env.step(action)
        img = env.render()
        
        # Ensure image is RGB
        if img is not None:
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]  # Remove alpha channel
            images.append(img)
        
        # Save individual frame
        if img_path is not None and img is not None:
            frame_path = os.path.join(frames_dir, f"frame_{step:03d}.png")
            imageio.imwrite(frame_path, img)
        
        vals.append(env.get_val(reward, action))
        num_iteration += 1
        if done:
            break
    env.close()
    
    # Create and save a grid visualization
    if img_path is not None and len(images) > 0:
        grid_path = f"{base_path}_grid.png"
        grid = create_image_grid(images)
        imageio.imwrite(grid_path, grid)
        print(f"Grid visualization saved to: {grid_path}")
    
    # Print summary information
    print(f"Simulation complete: {num_iteration} frames saved")
    if len(vals) > 0:
        print(f"Final value: {vals[-1]}")

def create_image_grid(images, rows=None, cols=None):
    """Create a grid of images for easy visualization"""
    # Calculate grid dimensions if not provided
    n = len(images)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    if rows is None and cols is None:
        cols = min(4, n)  # Max 4 columns
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    
    # Get dimensions of images and ensure they're all RGB
    h, w = images[0].shape[:2]
    channels = 3  # Force RGB
    
    # Create grid
    grid = np.ones((h * rows, w * cols, channels), dtype=np.uint8) * 255
    
    # Fill grid with images
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        r = i // cols
        c = i % cols
        
        # Ensure image is RGB format
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Convert RGBA to RGB
        elif len(img.shape) == 2:
            # Convert grayscale to RGB
            img = np.stack([img, img, img], axis=2)
            
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
    
    return grid
# Visualize a particular trajectories from given model's policy.
# def visualize(env, model, num_step=100, benchmark_model=False,
#               extra_args='random', img_path=None):
#     vals = []; images = []
#     num_iteration = 0
#     obs = env.reset_at(mode=extra_args)
#     action = np.zeros(obs.shape)
#     for _ in range(num_step):
#         if benchmark_model:
#             action, _ = model.predict(obs)
#         else:
#             action = model.get_action(obs, action)
#         obs, reward, done ,_, _ = env.step(action)
#         img = env.render()
#         images.append(img)
#         vals.append(env.get_val(reward, action))
#         num_iteration += 1
#         if done:
#             break
#     env.close()
    
#     # Save simulation.
#     if img_path is not None:
#         imageio.mimsave(img_path, [np.array(img) for img in images], format='wmv', fps=20)
