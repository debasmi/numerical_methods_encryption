import numpy as np
import m
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------

h = 0.01
steps = 20000

# initial conditions
x0, y0, z0 = 1, 1, 1

# small perturbation
delta = 1e-8


# ---------------------------
# Lyapunov Exponent Function
# ---------------------------

def lyapunov(method_function):

    traj1 = method_function(x0, y0, z0, h, steps)
    traj2 = method_function(x0 + delta, y0, z0, h, steps)

    d0 = np.linalg.norm(traj1[0] - traj2[0])

    distances = []

    for i in range(1, steps):

        d = np.linalg.norm(traj1[i] - traj2[i])

        if d > 0:
            distances.append(np.log(d / d0))

    lyap = np.mean(distances) / (h * steps)

    return lyap, traj1


# ---------------------------
# Compute Lyapunov Values
# ---------------------------

lyap_euler, traj_euler = lyapunov(m.euler_method)

lyap_heun, traj_heun = lyapunov(m.heun_method)

lyap_rk4, traj_rk4 = lyapunov(m.rk4_method)

lyap_taylor, traj_taylor = lyapunov(m.taylor_method)

lyap_picard, traj_picard = lyapunov(m.picard_method)


# ---------------------------
# Stability Analysis
# ---------------------------

def stability_measure(traj):

    variance = np.var(traj[:,0])

    return variance


stab_euler = stability_measure(traj_euler)
stab_heun = stability_measure(traj_heun)
stab_rk4 = stability_measure(traj_rk4)
stab_taylor = stability_measure(traj_taylor)
stab_picard = stability_measure(traj_picard)


# ---------------------------
# Print Results
# ---------------------------

print("\nLyapunov Exponent Comparison")
print("--------------------------------")

print("Euler Method:", lyap_euler)
print("Heun Method:", lyap_heun)
print("RK4 Method:", lyap_rk4)
print("Taylor Method:", lyap_taylor)
print("Picard Method:", lyap_picard)



print("\nStability Measure (Variance of x)")
print("--------------------------------")

print("Euler:", stab_euler)
print("Heun:", stab_heun)
print("RK4:", stab_rk4)
print("Taylor:", stab_taylor)
print("Picard:", stab_picard)


# ---------------------------
# Plot Trajectory Comparison
# ---------------------------

plt.figure(figsize=(10,6))

plt.plot(traj_euler[:,0], label="Euler", alpha=0.6)
plt.plot(traj_heun[:,0], label="Heun", alpha=0.6)
plt.plot(traj_rk4[:,0], label="RK4", alpha=0.6)
plt.plot(traj_taylor[:,0], label="Taylor", alpha=0.6)
plt.plot(traj_picard[:,0], label="Picard (3 iters)", alpha=0.6)

plt.title("Chaotic Trajectory Stability Comparison")
plt.xlabel("Time Step")
plt.ylabel("x(t)")

plt.legend()

plt.show()