from test_utils import visualize
from utils import get_environment, setup_dpo_model
from benchmarks.sb3_utils import setup_benchmark_model
from envs import Shape, ShapeBoundary, Molecule
import os
import pdb

def visualize_util(method, env, env_name, num_step, extra_args):
    if method.startswith('DPO'):
        model = setup_dpo_model(method, env, env_name)
        benchmark_model = False
    else:
        model_path = os.path.join('benchmarks', 'models', f'{env_name}_{method}.zip')
        model = setup_benchmark_model(method, env, model_path)
        benchmark_model = True
    benchmark_str = '_benchmark' if benchmark_model else ''
    img_path = 'output/videos/' + env_name + benchmark_str + method + '_result.wmv'
    # if 'molecule' in env_name:
    #     img_path = None
    visualize(env, model, num_step=num_step, benchmark_model=benchmark_model,
              extra_args=extra_args, img_path=img_path)


# Visualization material deformation
env = ShapeBoundary(render_mode='rgb_array')
visualize_util('DPO_zero_order', env, 'shape_boundary', 20, 'half_random')
visualize_util('TRPO_0_99', env, 'shape_boundary', 20, 'half_random')
print('done with ShapeBoundary - DPO, TRPO viz')
# # Visualization topological material deformation
visualize_util('DPO_zero_order', Shape(), 'shape', 20, 'hole')
visualize_util('TRPO_0_99', Shape(naive=True), 'naive_shape', 20, 'hole')
print('done with Shape - DPO, TRPO viz')

# Visualization molecular dynamics
# env_name = 'molecule'
# env = get_environment(env_name)
# visualize_util('DPO_zero_order', get_environment('molecule'), 'molecule', 6, 'random')
# visualize_util('SAC_0_99', get_environment('naive_molecule'), 'naive_molecule', 6, 'random')
# print('done with molecule - DPO, TRPO viz')
 