import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gtsam
from gtsam.symbol_shorthand import X
import os

# ============================================================
# PARAMETRY PROBLEMU
# ============================================================

N = 25
DT = 1.0  # Krok czasowy między stanami (Delta t)

# [x, y, vx, vy] - start i cel z zerową prędkością
START = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
GOAL = np.array([10.0, 8.0, 0.0, 0.0], dtype=np.float64)

OBSTACLES = []
OUTPUT_DIR = "."

SAFETY_DISTANCE = 0.45

# Parametry
SIGMA_START = 0.001
SIGMA_GOAL = 0.001
Q_C = 0.05  # Gęstość widmowa szumu procesu Gaussa (smoothness)
SIGMA_OBSTACLE = 0.15

# ============================================================
# MACIERZE PROCESU GAUSSA
# ============================================================

# Definicja macierzy przejścia stanu Phi dla modelu Constant Velocity
PHI = np.array([
    [1.0, 0.0, DT, 0.0],
    [0.0, 1.0, 0.0, DT],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)


# Konstrukcja macierzy kowariancji Qi zintegrowanego szumu procesu
def d_val(pow_val):
    return pow_val * Q_C


Q_I = np.array([
    [d_val(DT ** 3 / 3.0), 0.0, d_val(DT ** 2 / 2.0), 0.0],
    [0.0, d_val(DT ** 3 / 3.0), 0.0, d_val(DT ** 2 / 2.0)],
    [d_val(DT ** 2 / 2.0), 0.0, d_val(DT), 0.0],
    [0.0, d_val(DT ** 2 / 2.0), 0.0, d_val(DT)]
], dtype=np.float64)


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def create_initial_values():
    values = gtsam.Values()
    for i in range(N):
        alpha = i / (N - 1)
        pos = (1.0 - alpha) * START[:2] + alpha * GOAL[:2]
        vel = (GOAL[:2] - START[:2]) / (N * DT)
        state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)
        values.insert(X(i), state)
    return values


def values_to_trajectory(values):
    trajectory = []
    for i in range(N):
        state = values.atVector(X(i))
        trajectory.append([state[0], state[1]])
    return np.array(trajectory, dtype=np.float64)


def obstacle_penetration(point, obstacle):
    center = obstacle["center"]
    radius = obstacle["radius"]
    dist_to_center = np.linalg.norm(point - center)
    min_allowed_dist = radius + SAFETY_DISTANCE
    return max(0.0, min_allowed_dist - dist_to_center)


# ============================================================
# CZYNNIKI DO GRAFU
# ============================================================

def gp_prior_error_func(this, values, H):
    keys = this.keys()
    theta_prev = values.atVector(keys[0])
    theta_curr = values.atVector(keys[1])

    error = PHI @ theta_prev - theta_curr

    if H is not None:
        H[0] = np.asarray(PHI, dtype=np.float64, order="F")
        H[1] = np.asarray(-np.eye(4), dtype=np.float64, order="F")

    return np.asarray(error, dtype=np.float64)


def make_obstacle_error_func(center, radius):
    center = np.asarray(center, dtype=np.float64)

    def obstacle_error_func(this, values, H):
        key = this.keys()[0]
        state = values.atVector(key)
        p = state[:2]

        diff = p - center
        dist = np.linalg.norm(diff)
        min_allowed_dist = radius + SAFETY_DISTANCE
        penetration = min_allowed_dist - dist

        if penetration <= 0.0:
            error = np.array([0.0], dtype=np.float64)
            if H is not None:
                H[0] = np.asarray(np.zeros((1, 4)), dtype=np.float64, order="F")
            return error

        error = np.array([penetration], dtype=np.float64)

        if H is not None:
            direction = np.array([1.0, 0.0]) if dist < 1e-9 else diff / dist
            jacobian = np.array([[-direction[0], -direction[1], 0.0, 0.0]], dtype=np.float64)
            H[0] = np.asarray(jacobian, dtype=np.float64, order="F")

        return error

    return obstacle_error_func


def prior_factor_error_func(target_state):
    def func(this, values, H):
        key = this.keys()[0]
        state = values.atVector(key)
        error = state - target_state
        if H is not None:
            H[0] = np.asarray(np.eye(4), dtype=np.float64, order="F")
        return error

    return func


# ============================================================
# BUDOWA GRAFU I OPTYMALIZACJA
# ============================================================

def build_graph():
    graph = gtsam.NonlinearFactorGraph()

    start_noise = gtsam.noiseModel.Isotropic.Sigma(4, SIGMA_START)
    goal_noise = gtsam.noiseModel.Isotropic.Sigma(4, SIGMA_GOAL)
    obstacle_noise = gtsam.noiseModel.Isotropic.Sigma(1, SIGMA_OBSTACLE)
    gp_noise = gtsam.noiseModel.Gaussian.Covariance(Q_I)

    graph.add(gtsam.CustomFactor(start_noise, [X(0)], prior_factor_error_func(START)))
    graph.add(gtsam.CustomFactor(goal_noise, [X(N - 1)], prior_factor_error_func(GOAL)))

    for i in range(1, N):
        graph.add(gtsam.CustomFactor(gp_noise, [X(i - 1), X(i)], gp_prior_error_func))

    for i in range(1, N - 1):
        for obs in OBSTACLES:
            graph.add(
                gtsam.CustomFactor(obstacle_noise, [X(i)], make_obstacle_error_func(obs["center"], obs["radius"])))

    return graph


def optimize_trajectory():
    graph = build_graph()
    initial_values = create_initial_values()

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    return initial_values, result


# ============================================================
# FUNKCJE GENERUJĄCE WYKRESY
# ============================================================

def plot_trajectory(initial_values, result_values):
    initial_traj = values_to_trajectory(initial_values)
    result_traj = values_to_trajectory(result_values)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(initial_traj[:, 0], initial_traj[:, 1], "--", label="trajektoria początkowa")
    ax.plot(result_traj[:, 0], result_traj[:, 1], "-o", markersize=4, label="trajektoria po optymalizacji")
    ax.scatter(START[0], START[1], s=100, label="start", color="blue", zorder=5)
    ax.scatter(GOAL[0], GOAL[1], s=100, label="cel", color="green", zorder=5)

    for idx, obs in enumerate(OBSTACLES):
        ax.add_patch(plt.Circle(obs["center"], obs["radius"], fill=False, linewidth=1.5, color='red',
                                label="przeszkoda" if idx == 0 else ""))
        ax.add_patch(
            plt.Circle(obs["center"], obs["radius"] + SAFETY_DISTANCE, fill=False, linestyle=":", linewidth=1.2,
                       color='orange', label="strefa bezpieczeństwa" if idx == 0 else ""))

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Optymalizacja trajektorii przy użyciu GP Prior")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_trajektoria.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_obstacle_cost_field(result_values):
    result_traj = values_to_trajectory(result_values)

    x_min, x_max = -1.0, 11.0
    y_min, y_max = -1.0, 9.0

    xs = np.linspace(x_min, x_max, 150)
    ys = np.linspace(y_min, y_max, 150)

    X_grid, Y_grid = np.meshgrid(xs, ys)
    cost_grid = np.zeros_like(X_grid)

    for iy in range(Y_grid.shape[0]):
        for ix in range(X_grid.shape[1]):
            p = np.array([X_grid[iy, ix], Y_grid[iy, ix]], dtype=np.float64)
            total_penetration = 0.0
            for obs in OBSTACLES:
                total_penetration += obstacle_penetration(p, obs)
            cost_grid[iy, ix] = total_penetration

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(X_grid, Y_grid, cost_grid, levels=30, cmap='inferno')
    plt.colorbar(contour, ax=ax, label="koszt bliskości przeszkód")

    ax.plot(result_traj[:, 0], result_traj[:, 1], "-o", markersize=4, color="cyan", label="zoptymalizowana ścieżka")
    ax.scatter(START[0], START[1], s=100, color="blue", label="start", zorder=5)
    ax.scatter(GOAL[0], GOAL[1], s=100, color="green", label="cel", zorder=5)

    for obs in OBSTACLES:
        ax.add_patch(plt.Circle(obs["center"], obs["radius"], fill=False, linewidth=1.2, color='white', linestyle="--"))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Mapa kosztu przeszkód")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_mapa_kosztow.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_factor_graph_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))

    state_y = 0.0
    factor_y = 0.7
    obs_y = -0.7

    xs = np.arange(N)

    # Węzły zmiennych stanów x_i (Większe jasne koła)
    for i in range(N):
        ax.scatter(xs[i], state_y, s=250, marker="o", color="lightblue", edgecolors="black", zorder=3)
        if i % 2 == 0 or i in [0, N - 1]:
            ax.text(xs[i], state_y - 0.22, f"x{i}", ha="center", va="top", fontsize=8, fontweight="bold")

    # Czynnik Prior Start (Czarny kwadrat przypięty do x0)
    ax.scatter(xs[0], factor_y, s=120, marker="s", color="black", zorder=4)
    ax.text(xs[0], factor_y + 0.15, "prior\nstart", ha="center", va="bottom", fontsize=8)
    ax.plot([xs[0], xs[0]], [state_y, factor_y], linewidth=1.2, color="black")

    # Czynnik Prior Goal (Czarny kwadrat przypięty do x_N-1)
    ax.scatter(xs[-1], factor_y, s=120, marker="s", color="black", zorder=4)
    ax.text(xs[-1], factor_y + 0.15, "prior\ncel", ha="center", va="bottom", fontsize=8)
    ax.plot([xs[-1], xs[-1]], [state_y, factor_y], linewidth=1.2, color="black")

    # Czynniki binarne GP Prior (Brązowe kwadraty łączące sąsiednie stany)
    for i in range(1, N):
        # Umieszczamy czynnik dokładnie pomiędzy stanem i-1 a i
        fx = xs[i] - 0.5
        fy = 0.4
        ax.scatter(fx, fy, s=60, marker="s", color="darkred", zorder=4)

        if i % 5 == 0 or i == 1:
            ax.text(fx, fy + 0.15, "GP prior", ha="center", va="bottom", fontsize=7, color="darkred")

        ax.plot([fx, xs[i - 1]], [fy, state_y], linewidth=1.0, color="darkred")
        ax.plot([fx, xs[i]], [fy, state_y], linewidth=1.0, color="darkred")

    # Czynniki unarne przeszkód (Pomarańczowe romby podpięte bezpośrednio pod każdy stan wewnętrzny)
    for i in range(1, N - 1):
        fx = xs[i]
        ax.scatter(fx, obs_y, s=80, marker="D", color="orange", edgecolors="black", zorder=4)

        if i % 5 == 0:
            ax.text(fx, obs_y - 0.15, "obs", ha="center", va="top", fontsize=7, color="darkorange")

        ax.plot([fx, xs[i]], [obs_y, state_y], linewidth=1.0, linestyle=":", color="orange")

    ax.text(
        N / 2,
        1.3,
        "Struktura grafu czynników",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(-1.0, state_y, "Zmienne stanu", ha="right", va="center", fontsize=9)
    ax.text(-1.0, factor_y - 0.15, "GP Prior", ha="right", va="center", fontsize=9, color="darkred")
    ax.text(-1.0, obs_y, "Czynniki przeszkód", ha="right", va="center", fontsize=9, color="orange")

    ax.set_xlim(-2.5, N)
    ax.set_ylim(-1.3, 1.5)
    ax.axis("off")

    filepath = os.path.join(OUTPUT_DIR, "03_graf_czynnikow.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def generate_random_obstacles():
    obstacles = []
    for _ in range(3):
        cx = np.random.uniform(2.0, 8.0)
        cy = np.random.uniform(1.0, 7.0)
        r = np.random.uniform(0.6, 1.4)
        obstacles.append({
            "center": np.array([cx, cy], dtype=np.float64),
            "radius": r
        })
    return obstacles


# ============================================================
# PETLA GŁÓWNA SYMULACJI
# ============================================================

if __name__ == "__main__":
    base_dir = "Wyniki_symulacji"
    os.makedirs(base_dir, exist_ok=True)

    total_simulations = 25

    for sim_idx in range(1, total_simulations + 1):
        print(f"Symulacja: {sim_idx}/{total_simulations}")

        OUTPUT_DIR = os.path.join(base_dir, f"sim_{sim_idx:02d}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Generowanie losowych przeszkód dla każdego przebiegu
        OBSTACLES = generate_random_obstacles()

        # Optymalizacja oparta na ciągłym modelu GP
        initial_values, result_values = optimize_trajectory()

        # Generowanie i zapisywanie żądanych wykresów
        plot_trajectory(initial_values, result_values)
        plot_obstacle_cost_field(result_values)
        plot_factor_graph_diagram()

        print(f"Zapisano wyniki w: {OUTPUT_DIR}")
