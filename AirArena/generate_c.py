"""
generate_solver.py - Robust, non-singular CasADi binary generator (<30ms solve time).
"""
import casadi as ca
from simulator import MAX_ANG_SPEED, vehicle_dynamics

TOL_ANGLE = 0.05
TOL_SPEED = 0.07
MIN_DOT = float(ca.cos(TOL_ANGLE / 2.0))
TOL_SPEED_SQ = float(TOL_SPEED ** 2)


def build_solver(N=25):
    print(f"Building robust CasADi graph for N={N}...")
    opti = ca.Opti()

    # Inputs in WORLD frame
    p_x0 = opti.parameter(7)            # [q0(4), w0(3)]
    p_q_target = opti.parameter(4)      # Target quaternion (signed)

    # Decision variables
    Tf = opti.variable()
    X = opti.variable(7, N + 1)
    U = opti.variable(3, N)

    dt = Tf / N
    # Objective: time-optimal + small control regularization
    opti.minimize(Tf + 1e-4 * ca.sumsqr(U) * dt)

    # Initial and Time Bounds
    opti.subject_to(X[:, 0] == p_x0)
    opti.subject_to(opti.bounded(0.1, Tf, 6.0))

    # Terminal tolerance constraints
    opti.subject_to(ca.dot(X[0:4, -1], p_q_target) >= MIN_DOT)
    opti.subject_to(ca.sumsqr(X[4:7, -1]) <= TOL_SPEED_SQ)

    # RK4 dynamics integration & manifold integrity
    for k in range(N):
        x_k, u_k = X[:, k], U[:, k]

        # Actuator and velocity bounds
        opti.subject_to(opti.bounded(-1.0, u_k, 1.0))
        opti.subject_to(ca.sumsqr(x_k[4:7]) <= MAX_ANG_SPEED ** 2)

        # Enforce quaternion unit length at each node (prevents Inf explosion in RK4)
        opti.subject_to(ca.sumsqr(x_k[0:4]) == 1.0)

        def f_rk(x, u):
            dq, dw = vehicle_dynamics(x[0:4], x[4:7], u)
            return ca.vertcat(dq, dw)

        k1 = f_rk(x_k, u_k)
        k2 = f_rk(x_k + 0.5 * dt * k1, u_k)
        k3 = f_rk(x_k + 0.5 * dt * k2, u_k)
        k4 = f_rk(x_k + dt * k3, u_k)
        opti.subject_to(X[:, k + 1] == x_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4))

    # Strict, fast-converging IPOPT configuration
    opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 100,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 3,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.bound_frac": 1e-4,
        "ipopt.bound_push": 1e-4,
        "print_time": False
    }
    opti.solver("ipopt", opts)

    solver_fn = opti.to_function(
        "solve_reorient_world",
        [p_x0, p_q_target, Tf, X],
        [Tf, U],
        {"error_on_fail": False}
    )

    save_path = "casadi_solver.casadi"
    solver_fn.save(save_path)
    print(f"Successfully saved robust solver to '{save_path}'!")


if __name__ == "__main__":
    build_solver(N=25)
