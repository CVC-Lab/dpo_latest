import torch
from utils import get_environment, get_train_params, setup_main_net, DEVICE
from query_system import QuerySystem
from policy import Policy
from memory import ReplayMemory

MAX_CLIP_VAL = 1e2

def optimize_net(memory: ReplayMemory, main_net, num_iter, lr,
                 batch_size=32, log_interval=100):
    total_loss = 0
    cnt = 0
    # Setup optimizers
    optim = torch.optim.Adam(main_net.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    # Training main net.
    for i in range(num_iter):
        batch = memory.get_samples(batch_size)
        states_batch = torch.cat([torch.Tensor(s).reshape(1, -1) for s, _ in batch]).to(DEVICE)
        val_predicted = main_net(states_batch)
        val_expected = torch.cat([torch.Tensor(v).reshape(1, -1) for _, v in batch]).to(DEVICE)
        loss = criterion(val_predicted, val_expected)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(main_net.parameters(), MAX_CLIP_VAL)
        optim.step()
        total_loss += loss.item()
        # Logging.
        cnt += 1
        if (i+1) % log_interval == 0:
            print('Iter {}: loss {:.3f}.'.format(i+1, total_loss/cnt))
            cnt = 0
            total_loss = 0

def train(env_name, num_optimize_iters, zero_order=True):
    # Get hyperparams from file.
    rate, num_traj, step_size,\
        lr, batch_size, log_interval = get_train_params(env_name)
    print('Hyperparams: Rate: {:.3f}, num traj: {}, step size: {:.5f}, '
          'lr: {:.5f}, batch_size: {}, log_interval:{}'.format(
        rate, num_traj, step_size, lr, batch_size, log_interval
    ))

    # Setup environment and query system
    env = get_environment(env_name)
    query_system = QuerySystem(env)
    state_dim = query_system.state_dim

    # Setup memory
    memory = ReplayMemory(query_system, zero_order)

    # Setup main net
    main_net = setup_main_net(env_name, zero_order, state_dim)
    
    # Setup policy
    policy = Policy(zero_order, main_net, rate, step_size)
    memory.set_policy(policy)
    
    # Main training loop over time step.
    max_step = len(num_optimize_iters)
    for stage in range(max_step):
        print('\n\nCurrently at stage {}'.format(stage))
        # Sample
        memory.add_samples(num_traj, max_step=stage)
            
        # Optimize net for next stage from replay memory
        optimize_net(memory, main_net, num_optimize_iters[stage], lr, batch_size, log_interval)

        # Refresh memory
        memory.refresh()

    # Save main net.
    zero_order_str = 'zero_order' if zero_order else 'first_order'
    torch.save(main_net.state_dict(), 'models/' + env_name + '_OCF_' + zero_order_str + '.pth')
    print('\nDone OCF training.')
