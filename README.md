
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# IC ENGINE CYCLE SIMULATION
# Otto, Diesel and Dual cycles
# Air-standard, constant specific heats
# ==========================================

# Air properties
gamma = 1.4
R = 287.0              # J/(kg K)
cv = R / (gamma - 1)
cp = gamma * cv

# Initial conditions
P1 = 100e3             # Pa
T1 = 300.0             # K
compression_ratio = 8.0

# Heat supplied per kg of air
q_in = 800e3           # J/kg

# Dual cycle: fraction of heat added at constant volume
dual_cv_fraction = 0.5

# State calculations use specific volume (m^3/kg)
v1 = R * T1 / P1
v2 = v1 / compression_ratio

def efficiency(work, heat_input):
    return work / heat_input

# ==========================================
# OTTO CYCLE
# 1-2: Isentropic compression
# 2-3: Constant-volume heat addition
# 3-4: Isentropic expansion
# 4-1: Constant-volume heat rejection
# ==========================================

T2_o = T1 * compression_ratio**(gamma - 1)
P2_o = P1 * compression_ratio**gamma

T3_o = T2_o + q_in / cv
P3_o = P2_o * T3_o / T2_o

T4_o = T3_o / compression_ratio**(gamma - 1)
P4_o = P3_o / compression_ratio**gamma

eta_otto = 1 - 1 / compression_ratio**(gamma - 1)
w_otto = eta_otto * q_in

# ==========================================
# DIESEL CYCLE
# Heat addition at constant pressure
# ==========================================

T2_d = T2_o
P2_d = P2_o

# Cutoff ratio determined by heat input
cutoff_ratio = 1 + q_in / (cp * T2_d)

v3_d = v2 * cutoff_ratio
T3_d = T2_d * cutoff_ratio
P3_d = P2_d

# Isentropic expansion to v4 = v1
T4_d = T3_d * (v3_d / v1)**(gamma - 1)
P4_d = P3_d * (v3_d / v1)**gamma

eta_diesel = 1 - (
    (cutoff_ratio**gamma - 1)
    / (gamma * compression_ratio**(gamma - 1)
       * (cutoff_ratio - 1))
)
w_diesel = eta_diesel * q_in

# ==========================================
# DUAL CYCLE
# Part constant-volume, part constant-pressure
# ==========================================

T2_du = T2_o
P2_du = P2_o

q_cv = dual_cv_fraction * q_in
q_cp = (1 - dual_cv_fraction) * q_in

# Constant-volume heat addition: 2-3
T3_du = T2_du + q_cv / cv
P3_du = P2_du * T3_du / T2_du
v3_du = v2

# Constant-pressure heat addition: 3-4
T4_du = T3_du + q_cp / cp
P4_du = P3_du
v4_du = v3_du * T4_du / T3_du

# Isentropic expansion: 4-5
T5_du = T4_du * (v4_du / v1)**(gamma - 1)
P5_du = P4_du * (v4_du / v1)**gamma

# Thermal efficiency from heat rejected
q_out_du = cv * (T5_du - T1)
eta_dual = 1 - q_out_du / q_in
w_dual = eta_dual * q_in

# ==========================================
# P-V CURVE HELPERS
# ==========================================

def isentropic_curve(v_start, v_end, p_start, n=100):
    v = np.linspace(v_start, v_end, n)
    p = p_start * (v_start / v)**gamma
    return v, p

def constant_pressure(v_start, v_end, pressure, n=50):
    return (
        np.linspace(v_start, v_end, n),
        np.full(n, pressure)
    )

def constant_volume(volume, p_start, p_end, n=50):
    return (
        np.full(n, volume),
        np.linspace(p_start, p_end, n)
    )

def draw_otto(ax):
    v12, p12 = isentropic_curve(v1, v2, P1)
    v23, p23 = constant_volume(v2, P2_o, P3_o)
    v34, p34 = isentropic_curve(v2, v1, P3_o)
    v41, p41 = constant_volume(v1, P4_o, P1)

    ax.plot(v12, p12 / 1000, label="1-2 Compression")
    ax.plot(v23, p23 / 1000, label="2-3 Heat addition")
    ax.plot(v34, p34 / 1000, label="3-4 Expansion")
    ax.plot(v41, p41 / 1000, label="4-1 Heat rejection")

def draw_diesel(ax):
    v12, p12 = isentropic_curve(v1, v2, P1)
    v23, p23 = constant_pressure(v2, v3_d, P2_d)
    v34, p34 = isentropic_curve(v3_d, v1, P3_d)
    v41, p41 = constant_volume(v1, P4_d, P1)

    ax.plot(v12, p12 / 1000, label="1-2 Compression")
    ax.plot(v23, p23 / 1000, label="2-3 Heat addition")
    ax.plot(v34, p34 / 1000, label="3-4 Expansion")
    ax.plot(v41, p41 / 1000, label="4-1 Heat rejection")

def draw_dual(ax):
    v12, p12 = isentropic_curve(v1, v2, P1)
    v23, p23 = constant_volume(v2, P2_du, P3_du)
    v34, p34 = constant_pressure(v3_du, v4_du, P3_du)
    v45, p45 = isentropic_curve(v4_du, v1, P4_du)
    v51, p51 = constant_volume(v1, P5_du, P1)

    ax.plot(v12, p12 / 1000, label="1-2 Compression")
    ax.plot(v23, p23 / 1000, label="2-3 Constant volume")
    ax.plot(v34, p34 / 1000, label="3-4 Constant pressure")
    ax.plot(v45, p45 / 1000, label="4-5 Expansion")
    ax.plot(v51, p51 / 1000, label="5-1 Heat rejection")

# ==========================================
# PRINT RESULTS
# ==========================================

print("\nIDEAL IC ENGINE CYCLE COMPARISON")
print("=" * 48)
print(f"Compression ratio: {compression_ratio:.1f}")
print(f"Heat input: {q_in / 1000:.1f} kJ/kg")
print(f"Initial pressure: {P1 / 1000:.1f} kPa")
print(f"Initial temperature: {T1:.1f} K")

print("\nThermal efficiency:")
print(f"Otto cycle:  {eta_otto * 100:.2f}%")
print(f"Diesel cycle: {eta_diesel * 100:.2f}%")
print(f"Dual cycle:  {eta_dual * 100:.2f}%")

print("\nNet work output:")
print(f"Otto cycle:  {w_otto / 1000:.2f} kJ/kg")
print(f"Diesel cycle: {w_diesel / 1000:.2f} kJ/kg")
print(f"Dual cycle:  {w_dual / 1000:.2f} kJ/kg")

print("\nPeak cycle temperatures:")
print(f"Otto: {T3_o:.1f} K")
print(f"Diesel: {T3_d:.1f} K")
print(f"Dual: {max(T3_du, T4_du):.1f} K")

# ==========================================
# PLOT P-V DIAGRAMS
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

draw_otto(axes[0])
axes[0].set_title("Otto Cycle")
axes[0].legend(fontsize=7)

draw_diesel(axes[1])
axes[1].set_title("Diesel Cycle")
axes[1].legend(fontsize=7)

draw_dual(axes[2])
axes[2].set_title("Dual Cycle")
axes[2].legend(fontsize=7)

for ax in axes:
    ax.set_xlabel("Specific volume (m³/kg)")
    ax.set_ylabel("Pressure (kPa)")
    ax.grid(True)

plt.tight_layout()
plt.savefig("ic_engine_pv_diagrams.png", dpi=300)
plt.show()

# ==========================================
# EFFICIENCY COMPARISON
# ==========================================

names = ["Otto", "Diesel", "Dual"]
efficiencies = [
    eta_otto * 100,
    eta_diesel * 100,
    eta_dual * 100
]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    names, efficiencies,
    color=["royalblue", "darkorange", "seagreen"]
)

plt.ylabel("Thermal efficiency (%)")
plt.title("Ideal Engine Cycle Efficiency Comparison")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, efficiencies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("ic_engine_efficiency.png", dpi=300)
plt.show()
