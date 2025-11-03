import numpy as np
import matplotlib.pyplot as plt

def theta_D_function(D, alpha, beta):
    return alpha * (1 - np.exp(-beta * D / 100))

def Q2_function(k, Q1):
    return k * Q1

def deterministic_part(state, params):
    x, y, z = state
    Gs2, Gs1, F, Cg2 = params['Gs2'], params['Gs1'], params['F'], params['Cg2']
    L1, L2, L3 = params['L1'], params['L2'], params['L3']
    Q1, k = params['Q1'], params['k']
    s1, s2 = params['s1'], params['s2']
    pp1, pp2 = params['pp1'], params['pp2']
    Cp2, Tp, Tc = params['Cp2'], params['Tp'], params['Tc']
    i1, i2 = params['i1'], params['i2']
    pc1, pc2 = params['pc1'], params['pc2']
    Cc, Ec, Ep = params['Cc'], params['Ec'], params['Ep']
    D, alpha, beta = params['D'], params['alpha'], params['beta']
    mc, mp = params['mc'], params['mp']
    theta_D = theta_D_function(D, alpha, beta)
    Q2 = Q2_function(k, Q1)

    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    z = np.clip(z, 0, 1)

    dx_dt = x * (1 - x) * (-z * (L3 + Gs2 - mc) - y * (L2 + Gs1 - mp) + (F + L2 + L3 - Cg2 + L1 - mp - mc))
    dy_dt = y * (1 - y) * (x * (L2 + Gs1) + z * ((s1 + s2 - pp1 - pp2) * (Q1 - Q2)) +
                           ((s1 - pp1) * Q2 - (s2 - pp2) * Q1 - Cp2 + Tp * Ep * theta_D))
    dz_dt = z * (1 - z) * (x * (Gs2 + L2) + y * (2 * (i1 * Q1 - i2 * Q2) - (pc1 + pc2) * (Q1 - Q2)) +
                           ((i2 * Q2 - i1 * Q1) - Cc - pc2 * (Q2 - Q1) + Tc * Ec * theta_D))
    return np.array([dx_dt, dy_dt, dz_dt])

def sde_milstein_multiplicative(x0, t_span, dt, params, num_simulations=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t_start, t_end = t_span
    t_points = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_points)
    trajectories = np.zeros((num_simulations, n_steps, 3))
    for sim in range(num_simulations):
        state = np.array(x0, dtype=float)
        trajectories[sim, 0, :] = state
        for i in range(1, n_steps):
            drift = deterministic_part(state, params)
            diffusion_x = params['sigma_x'] * state[0]
            diffusion_y = params['sigma_y'] * state[1]
            diffusion_z = params['sigma_z'] * state[2]
            dW_x = np.random.normal(0, np.sqrt(dt))
            dW_y = np.random.normal(0, np.sqrt(dt))
            dW_z = np.random.normal(0, np.sqrt(dt))
            milstein_corr_x = 0.5 * diffusion_x * params['sigma_x'] * ((dW_x ** 2) - dt)
            milstein_corr_y = 0.5 * diffusion_y * params['sigma_y'] * ((dW_y ** 2) - dt)
            milstein_corr_z = 0.5 * diffusion_z * params['sigma_z'] * ((dW_z ** 2) - dt)
            state[0] += drift[0] * dt + diffusion_x * dW_x + milstein_corr_x
            state[1] += drift[1] * dt + diffusion_y * dW_y + milstein_corr_y
            state[2] += drift[2] * dt + diffusion_z * dW_z + milstein_corr_z
            state = np.clip(state, 0, 1)
            trajectories[sim, i, :] = state
    return t_points, trajectories

params_base = {
    # 政府
    'Cg1': 5, 'Cg2': 3, 'F': 5, 'L1': 2,
    'Gs1': 2, 'Gs2': 1,
    'mp': 0.6, 'mc': 0.4,
    'L2': 2, 'L3': 0.2,

    # 生产方
    'Cp1': 5, 'Cp2': 3,
    's1': 4, 's2': 2,
    'pp1': 3, 'pp2': 1,
    'Tp': 6, 'Ep': 2,

    # 消费者
    'Cc': 2,
    'pc1': 5, 'pc2': 4,
    'i1': 6,  'i2': 5,
    'Tc': 3, 'Ec': 1,

    # 其它
    'D': 80, 'alpha': 0.6, 'beta': 1, 'Q1': 10, 'k': 0.8,
}

initial_state = [0.5, 0.5, 0.5]
t_span = [0, 10]
dt = 0.01
num_sim = 5
sigma_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 不同随机干扰强度


fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# a-政府策略x
for sigma in sigma_list:
    params = params_base.copy()
    params.update({'sigma_x': sigma, 'sigma_y': sigma, 'sigma_z': sigma})
    t, trajectories = sde_milstein_multiplicative(
        initial_state, t_span, dt, params, num_simulations=num_sim)
    mean_traj = np.mean(trajectories, axis=0)
    axes[0].plot(t, mean_traj[:, 0], lw=1, label=f'$\sigma$={sigma:.1f}')

axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('Strategy proportion (x)', fontsize=12)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].text(0.05, 0.95, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# b-生产方策略y
for sigma in sigma_list:
    params = params_base.copy()
    params.update({'sigma_x': sigma, 'sigma_y': sigma, 'sigma_z': sigma})
    t, trajectories = sde_milstein_multiplicative(
        initial_state, t_span, dt, params, num_simulations=num_sim)
    mean_traj = np.mean(trajectories, axis=0)
    axes[1].plot(t, mean_traj[:, 1], lw=1, label=f'$\sigma$={sigma:.1f}')

axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Strategy proportion (y)', fontsize=12)
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].text(0.05, 0.95, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# c-消费者策略z
for sigma in sigma_list:
    params = params_base.copy()
    params.update({'sigma_x': sigma, 'sigma_y': sigma, 'sigma_z': sigma})
    t, trajectories = sde_milstein_multiplicative(
        initial_state, t_span, dt, params, num_simulations=num_sim)
    mean_traj = np.mean(trajectories, axis=0)
    axes[2].plot(t, mean_traj[:, 2], lw=1, label=f'$\sigma$={sigma:.1f}')

axes[2].set_xlabel('Time', fontsize=12)
axes[2].set_ylabel('Strategy proportion (z)', fontsize=12)
axes[2].set_ylim(-0.05, 1.05)
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].text(0.05, 0.95, '(c)', transform=axes[2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.show()
