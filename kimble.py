"""
    Docstring for kimble.py

This file uses our formalism to compute the coupling between a quantum emitter
and a photonic crystal. We made use of a tight-binding model to simulate the
crystal and assume the emitter as a two-level system.

Additionally, we compute the time evolution of both the emitter and the photonic
bath for different polarizations of the emitter and compare them and include an
additional emitter to probe emission in the bath and the concurrence between both
emitters.

NOTE: this file assumes that the fields are already computed for the points where
the emitters are located. If not, consider running `./miscellaneous/mpb_field_computation.py`.
"""

# ------------------------------------------------------------------------------
# Import packages and some tweaks for plottings
# ------------------------------------------------------------------------------

# Import necessary packages
import numpy as np
import meep as mp  # needed for the numerical simulation of the photonic crystal
from meep import mpb  # same case as the previous package
import pythtb as tb  # for easy simulation of the tight-binding model
from scipy.sparse.linalg import (
    expm_multiply,
)  # for solving the time-dependent Schrödinger equation

# some imports for plotting utilities
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import src as mf  # our package, see `./src/requirements.txt` for dependencies

# Some set-ups for the plotting of the figures
mpl.rcParams.update(
    {
        # Figure
        "figure.dpi": 300,  # higher resolution
        # Font
        "font.family": "serif",  # switch to serif (journal-style)
        "font.size": 11,  # base font size
        "mathtext.fontset": "stix",  # match math to Times-like font
        # Axes
        "axes.labelsize": 11,  # axis label font size
        "axes.titlesize": 12,  # title font size
        "axes.linewidth": 1,  # thickness of axes spines
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
        # Save
        "savefig.bbox": "tight",
    }
)

# ------------------------------------------------------------------------------
# Define the parameters necessary for the simulation
# ------------------------------------------------------------------------------

# Parameters of the MPB simulation
N_BANDS = 2
BAND_INDEX = 0  # index of the band to consider (0-based)
ε_INPUT_FILE = "data/kimble/epsilon.h5"

N_K = 64  # number of points per direction in the Monkhorst-Pack grid,
#           also used to identify the file where the fields are stored

# Parameters of the emitters that we are going to consider. In this case, we are
# going to consider two emitters at the same position r₀ but with different orientations
θ_1 = (90) * np.pi / 180
θ_2 = (135) * np.pi / 180
DIP_STRENGTH = 1e-2
N_EMITTERS = 1  # per simulation.
# NOTE: In this case there are two simulations (one per orientation)
#       and one emitter per simulation
R_0 = [0.0, 0.0]  # position(s) of the emitter. List of positions is also allowed

# wrap the emitter orientations and dipoles in lists
θs = [θ_1, θ_2]
dipoles = [DIP_STRENGTH * np.array([np.cos(θ), np.sin(θ), 0]) for θ in θs]
dip_colors = ["#1f77b4", "#ff7f0e"]  # colors associated to each dipole

# Parameters for the tight-binding model
LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]  # position of the orbital in lattice units
N_CELLS = 51  # number of unit cells in each direction

# coefficients for the tight-binding model, considered until 3-rd nearest neighbors
J_0 = 0.07755639
J_1 = -0.01993698
J_2 = 0.00053966

# nodes in the path inside the Brillouin zone for band structure calculation
K_PATH = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
# define high-symmetry points labels
K_LABEL = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]

# Parameters for the dynamic simulation
T_0 = 0  # starting time
T_F = 500  # final time
T_STEPS = 100  # number of time steps considered
SNAP_INDEX = (
    50  # time index to consider for the field snapshots, can go from 0 to `T_STEPS`
)

# Read some data from the epsilon.h5 file
ε_data = mf.read_epsilon_file(ε_INPUT_FILE)
resolution = ε_data["resolution"]
lattice_vecs = ε_data["lattice_vectors"]
lattice_size = ε_data["lattice_size"]

# Print summary of parameters
print("\n")
print("Number of bands      = " + str(N_BANDS))
print("Epsilon file used    = " + ε_INPUT_FILE)
print("Resolution           = " + str(resolution))
print("Lattice size         = " + str(lattice_size))
print("Lattice vectors      = " + str(lattice_vecs))

# ------------------------------------------------------------------------------
# Read epsilon data for plotting
# ------------------------------------------------------------------------------
md = mpb.MPBData(rectify=False, periods=2)
ε_converted = mf.convert_epsilon("./data/kimble/epsilon.h5", md)

# retrieve data for plotting
x = ε_converted["positions-lattice"][:, 0]
y = ε_converted["positions-lattice"][:, 1]
z = ε_converted["epsilon_values"]

# --- Plot the dielectric function
fig, ax = plt.subplots()

sc = ax.scatter(x, y, c=z, marker="s", cmap="copper_r", s=mf.marker_size(len(x)))
fig.colorbar(sc, ax=ax, label=r"$\epsilon(\mathbf{r})$")
ax.set(xlabel=r"$x/a$", ylabel=r"$y/a$", aspect="equal")

for i, θ in enumerate(θs):
    mf.draw_arrow(
        ax,
        R_0,
        θ,
        color=dip_colors[i],
        label=rf"$\theta_{i+1}$",
        length=0.4,
        label_offset=0.09,
        arrow_offset=0.05,
    )

fig.savefig(f"./figures/kimble/dielectric_function_dipoles_at_{R_0}.png")
plt.show()

# ------------------------------------------------------------------------------
# Compute the TB bands and the MPB bands for comparison
# ------------------------------------------------------------------------------

# Initialize tight-binding model
infinite_model = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)

# Onsite terms
infinite_model.set_onsite([J_0])

# First nearest neighbors (1st nn)
first_nn_vectors = np.array([[1, 0], [0, 1]])
for vec in first_nn_vectors:
    infinite_model.set_hop(J_1, 0, 0, vec)

# Second nearest neighbors (2nd nn)
second_nn_vectors = np.array([[1, 1], [1, -1]])
for vec in second_nn_vectors:
    infinite_model.set_hop(J_2, 0, 0, vec)

# --- Compute bands for the tight-binding model

### Compute bands along high-symmetry path

# creates a path between the high-symmetry points
n_k_points = 100
k_vec, k_dist, k_node = infinite_model.k_path(K_PATH, n_k_points, report=False)

# solve the infinite model in the considered path
tb_evals = infinite_model.solve_all(k_vec)

## Select the detuning frequency
# we are picking the value of the frequency at the X point where there is the
# saddle point of the band and a high density of states
ω_0 = np.sqrt(tb_evals[BAND_INDEX][k_node[1] == k_dist].item())
print(f"Detuning frequency selected at X point: {ω_0:.6f} (2πc/a)")

# --- Compute the bands with MPB for comparison

geometry_lattice = mp.Lattice(
    size=lattice_size, basis1=lattice_vecs[0], basis2=lattice_vecs[1]
)

ms = mpb.ModeSolver(
    num_bands=N_BANDS,
    epsilon_input_file=ε_INPUT_FILE,
    geometry_lattice=geometry_lattice,
    k_points=mf.array_to_Vector3(k_vec),
)

ms.run_te()

# all frequency bands
freqs = ms.all_freqs

# --- Plot the bands
fig, ax = plt.subplots()

for tb_eval in tb_evals:
    tb_eval_clipped = np.clip(
        np.real(tb_eval), 0, None
    )  # ensure no negative values due to numerical errors
    ax.plot(k_dist, np.sqrt(tb_eval_clipped), color="r", linestyle="--")
ax.plot(k_dist, freqs, color="k")
ax.hlines(ω_0, xmin=k_dist[0], xmax=k_dist[-1], color="k", linestyle="--", alpha=0.5)

ax.set(
    xlim=(k_node[0], k_node[-1]),
    xticks=k_node,
    xticklabels=K_LABEL,
    ylabel=r"$\omega \, [c/a]$",
    title="TE modes",
)
ax.grid(axis="x")

fig.savefig(f"./figures/kimble/MPBvTB_bandstructure.svg")
plt.show()

# ------------------------------------------------------------------------------
# Compute the emitter-bath coupling in real space g_r(r₀)
# ------------------------------------------------------------------------------

# Load the fields, frequencies and k-points from the MPB calculation
# NOTE that all this data is provided in the units MPB uses!
fields_data = mf.load(f"./data/kimble/fields/{N_K}_meshgrid_at_{R_0}.pkl")

# Extract relevant quantities
ω_ks_n = np.asarray(fields_data["freqs"])  # Angular frequencies (N_k, N_b)
frs_n_ks = np.asarray(fields_data["Eks"])  # Electric fields, shape: (N_r, N_b, N_k, 3)
ks = np.asarray(
    fields_data["kpts"]
)  # k-points in Cartesian coordinates, shape: (N_k, 3)

# Select the field at position r₀. Here we computed the field for only that point
fr0_n_ks = frs_n_ks[0, :, :, :]  # Shape: (N_b, N_k, 3)

# Define the real-space grid. It should correspond to a set of lattice sites
n_r = 13
xs = np.arange(-(n_r // 2), n_r // 2 + 1, 1)
ys = np.arange(-(n_r // 2), n_r // 2 + 1, 1)
rs = mf.meshgrid_points(xs, ys, indexing="xy")

# Compute g_r(r₀) for each dipole moment, shape (Nr,)
gr0_rs_list = [
    mf.compute_real_space_coupling(
        fr0_n_ks=fr0_n_ks,
        dipole_moment=dipole,
        ω_ks_n=ω_ks_n,
        ks=ks,
        rs=rs,
    )[BAND_INDEX, :].reshape(n_r, n_r)
    for dipole in dipoles
]

# --- Plot both g_r(r₀)

cbar_min = np.abs(gr0_rs_list).min()  # set a minimum value for the colorbars
cbar_max = np.abs(gr0_rs_list).max()  # set a maximum value for the colorbars

for i, θ in enumerate(θs):
    fig, ax = plt.subplots()

    sc = ax.pcolormesh(
        xs,
        ys,
        np.abs(gr0_rs_list[i]),  # take absolute value for plotting
        norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
    )

    cbar = fig.colorbar(sc, ax=ax, label=r"$|g_\mathbf{R}(\mathbf{r}_0)| \, [c/a]$")

    mf.draw_arrow(ax, R_0, θ, color=dip_colors[i], length=2, arrow_offset=0.3)

    ax.set(
        title=rf"$\theta_{i+1} = {np.rad2deg(θ):.0f}º$",
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
    )
    ax.title.set_color(dip_colors[i])

    fig.savefig(f"./figures/kimble/g_r({R_0})_for_θ_{np.rad2deg(θ):.0f}º.svg")
    plt.show()

# ------------------------------------------------------------------------------
# Compute the LDOS to see how the emitters couple to the photonic bath
# ------------------------------------------------------------------------------
ω_ks = ω_ks_n[:, BAND_INDEX]  # Shape: (Nk,)

# Frequency range for LDOS
ωs = np.linspace(np.min(ω_ks), np.max(ω_ks), n_k_points)

# Fields for the selected band
fr0_ks = fr0_n_ks[BAND_INDEX, :, :]  # Shape: (Nk, 3)

# Compute LDOS for each dipole
ldos_list = [
    mf.compute_ldos(ωs=ωs, ω_ks=ω_ks, fr0_ks=fr0_ks, dipole_moment=dipole)
    for dipole in dipoles
]

# --- Plot LDOS
for i, ldos in enumerate(ldos_list):
    fig, ax = plt.subplots()

    ax.scatter(ωs, ldos, c=dip_colors[i])
    ax.axvline(
        x=ω_0,
        color="k",
        linestyle="--",
        label=r"$\omega_0$",
        linewidth=1,
    )

    ax.legend()

    ax.set(
        xlabel=r"$\omega \, [c/a]$",
        ylabel=r"$\mathrm{LDOS} \, [c/a]$",
        title=rf"$R_0/a$={R_0}, $\theta_{i+1}={np.degrees(θs[i]):.0f}º$",
    )

    fig.savefig(f"./figures/kimble/ldos_at_{R_0}_θ_{np.rad2deg(θs[i]):.0f}.svg")
    plt.show()

# ------------------------------------------------------------------------------
# --- Compute the time evolution of the emitter and the bath
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Computed the normalized isofrequency contour
# ------------------------------------------------------------------------------

# --- Generate 2D k-point grid in the Brillouin zone
kx = np.linspace(-0.5, 0.5, nk_points)
ky = np.linspace(-0.5, 0.5, nk_points)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

# Flatten grid for evaluation
k_points = np.column_stack((kx_grid.ravel(), ky_grid.ravel()))

# --- Compute band energies for the selected band
band_energies = np.array(
    [infiniteModel.solve_one(k)[BAND_INDEX] for k in k_points],
)

# Clip all negative values (remove numerical noise)
band_energies_clipped = np.clip(band_energies, a_min=0.0, a_max=None)

# Reshape and compute omega = sqrt(E)
omega = np.sqrt(band_energies_clipped).reshape(nk_points, nk_points)

# --- Plot the isofrequency contour
fig, ax = plt.subplots()

c = ax.contourf(kx_grid, ky_grid, omega, levels=12)
cbar = fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k} \, [c/a]$")
cbar.formatter.set_powerlimits((0, 0))

ax.contour(
    kx_grid,
    ky_grid,
    omega,
    levels=[detuning_freq],
    colors="k",
    linewidths=2,
)

ax.set(
    xlabel=r"$k_x \, [2\pi/a]$",
    ylabel=r"$k_y \, [2\pi/a]$",
    aspect="equal",
)

fig.savefig(f"./figures/kimble/tb_isofrequency_contour.svg")
plt.show()

# ------------------------------------------------------------------------------
# Finite model definition and time evolution
# ------------------------------------------------------------------------------

# Cut the system
temporalModel = infiniteModel.cut_piece(
    N_CELLS, fin_dir=0, glue_edgs=False
)  # cutout along 1st direction
finiteModel = temporalModel.cut_piece(
    N_CELLS, fin_dir=1, glue_edgs=False
)  # cutout along 2nd direction

# --- Compute g_rj(r0) on the finite grid

if N_CELLS % 2 == 0:
    xjg = np.arange(-N_CELLS / 2, N_CELLS / 2, 1)
    yjg = np.arange(-N_CELLS / 2, N_CELLS / 2 + 1, 1)
else:
    xjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
    yjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

gr0rjs = [
    mf.compute_gr0r(
        E_k_r0=E_ks_r0,
        dipole_moment=dipole,
        omega_k=omegas,
        kpts=kpts,
        rpts=rjpts,
    )
    for dipole in dipoles
]

# we want to normalize also the couplings by the 1-st nn coupling strength
gr0rjs_norm = [gr0rj for gr0rj in gr0rjs]

# --- Build the light-matter Hamiltonian

couplings = [gr0rj_norm[BAND_INDEX, :] for gr0rj_norm in gr0rjs_norm]

# small check before proceeding
assert len(couplings[0]) == len(
    couplings[1]
), "Unexpected different lengths for the couplings!"

indexes = np.arange(0, len(couplings[0]), 1)
interaction_1 = [[list(couplings[0]), list(indexes)]]
interaction_2 = [[list(couplings[1]), list(indexes)]]

# build the light-matter Hamiltonian
H_total_1, omegas_total_1 = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq, interaction_1
)
H_total_2, omegas_total_2 = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq, interaction_2
)

# --- Compute time evolution

tvals = np.linspace(T0, TF, T_STEPS, endpoint=True)  # for later plotting

# Small check before computing the time evolution
assert (
    H_total_1.shape == H_total_2.shape
), "Unexpected different shapes for the Hamiltonians!"

# Initial condition
psi0 = np.zeros(H_total_1.shape[0])
psi0[0] = 1

# Compute time evolution
psit_1 = expm_multiply(-1j * H_total_1, psi0, T0, TF, T_STEPS, endpoint=True)
psit_2 = expm_multiply(-1j * H_total_2, psi0, T0, TF, T_STEPS, endpoint=True)

# --- Prepare data to be plotted

# get emitter component of the overall wavefunction
mod_Ce_squared_1 = np.abs(psit_1[:, 0]) ** 2
mod_Ce_squared_2 = np.abs(psit_2[:, 0]) ** 2

# compute the entanglement for both cases
entropy_1 = mf.binary_entropy(mod_Ce_squared_1)
entropy_2 = mf.binary_entropy(mod_Ce_squared_2)

# get positions in Cartesian coordinates
positions = mf.find_positions(finiteModel._orb, finiteModel._lat)

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
mod_Cr_squared_1 = np.abs(psit_1[TIME_INDEX, N_EMITTERS:]) ** 2
mod_Cr_squared_2 = np.abs(psit_2[TIME_INDEX, N_EMITTERS:]) ** 2

# shift the crystal to place emitter at the center
x = np.unique(positions[:, 0]) - N_CELLS // 2
y = np.unique(positions[:, 1]) - N_CELLS // 2
z1 = mod_Cr_squared_1.reshape(len(x), len(y))
z2 = mod_Cr_squared_2.reshape(len(x), len(y))

# --- Plot the emitter's time evolution

fig, ax = plt.subplots()

(l1,) = ax.plot(
    tvals,
    mod_Ce_squared_1,
    lw=3,
    color=dip_colors[0],
    label=r"$|C_e(t)|^2$, $\theta_1 = 90º$",
)
(l2,) = ax.plot(
    tvals,
    mod_Ce_squared_2,
    lw=2,
    color=dip_colors[1],
    alpha=0.9,
    label=r"$|C_e(t)|^2$, $\theta_2 = 135º$",
)

ax.set(xlabel=r"$t \, [a/c]$", ylabel=r"$|C_e(t)|^2$", yscale="log")

ax2 = ax.twinx()
(l3,) = ax2.plot(
    tvals,
    entropy_1,
    ls="--",
    color=dip_colors[0],
    lw=3,
    label=r"$S(t)$, $\theta_1 = 90º$",
)
(l4,) = ax2.plot(
    tvals,
    entropy_2,
    ls="--",
    color=dip_colors[1],
    alpha=0.9,
    lw=2.0,
    label=r"$S(t)$, $\theta_2 = 135º$",
)
ax2.set(ylabel=r"$S(t)$")

# combined legend
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels)

fig.savefig(f"./figures/kimble/emitter_timeevolution_entropy_at_{R0}.svg")
plt.show()

# --- Plot a bath population snapshot in time

cbar_max = max(z1.max(), z2.max())

for i, (theta, z) in enumerate(zip(thetas, [z1, z2])):
    fig, ax = plt.subplots()

    sc = ax.pcolormesh(
        x,
        y,
        z,
        vmin=0,
        vmax=cbar_max,
    )

    mf.draw_arrow(ax, R0, theta, color=dip_colors[i], length=8, arrow_offset=2)

    cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{R}(t)|^2$")
    cbar.formatter.set_powerlimits((0, 0))

    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title=rf"$\mathbf{{r}}_0/a = {R0}$",
        aspect="equal",
    )

    fig.tight_layout()
    fig.savefig(
        f"./figures/kimble/bath_snapshot_at_{R0}_θ_{np.rad2deg(theta):.0f}º.svg"
    )
    plt.show()

# ------------------------------------------------------------------------------
# compute the photon-photon entropy for each cuadrant to show that the one with
# four chanels is equal to log(4) and the one with two chanels is equatl to log(2)
# ------------------------------------------------------------------------------
amplitudes_1 = psit_1[:, N_EMITTERS:]
amplitudes_2 = psit_2[:, N_EMITTERS:]

S_1, p_1 = mf.quadrant_entropy(rjpts[:, 0], rjpts[:, 1], amplitudes_1)
S_2, p_2 = mf.quadrant_entropy(rjpts[:, 0], rjpts[:, 1], amplitudes_2)

check_1 = np.allclose(S_1[1:], np.log(4), atol=1e-3)
check_2 = np.allclose(S_2[1:], np.log(2), atol=1e-3)

print(f"Is S_dir equal to log(4) for θ = {np.rad2deg(THETA_1):.0f}º?", check_1)

print(f"Is S_dir equal to log(2) for θ = {np.rad2deg(THETA_2):.0f}º?", check_2)

# ------------------------------------------------------------------------------
# --- Place another emitter to probe directional emission
# ------------------------------------------------------------------------------

# We are going to place another emitter at position [-4.5,4.5] with orientation
# of 45º to check that this emitter should be excited when the first excited
# emitter has orientation 90º and not when 135º
new_n_emitters = 2
new_R = [-4.5, 4.5]
new_theta = np.deg2rad(45)
new_dipole = 1e-1 * np.array([np.cos(new_theta), np.sin(new_theta), 0])
new_dip_color = "#2ca02c"
my_red = "#d62728"

# we also need to provide a detuning frequency, we are going to pick the one
# associated to a state localized near the position of the emitter
target = [-5, 5]
idx = np.where(np.all(np.isclose(rjpts, target), axis=1))[0][0]
new_detuning_1 = omegas_total_1[idx]
new_detuning_2 = omegas_total_2[idx]

# for this study it is not necessary to go for times as big as the previous ones,
# so we are going to cut the simulations at previous times
new_tf = 50
new_tvals = np.linspace(T0, new_tf, T_STEPS, endpoint=True)
new_time_index = (
    20  # approximate time snapshot when the bath's excitation hits the emitter probe
)

add_data = mf.load(f"./data/kimble/fields/{NK}_meshgrid_at_{new_R}.pkl")
new_omegas = np.asarray(add_data["freqs"])  # (Nk, Nb)
E_ks_R = np.asarray(add_data["Eks"])[0, :, :, :]  # (Nb, Nk, 3)
new_kpts = np.asarray(add_data["kpts"])

gRrjs_norm = mf.compute_gr0r(E_ks_R, new_dipole, new_omegas, new_kpts, rjpts) / np.sqrt(
    np.abs(TAU_1)
)  # (Nb, Nr)

new_couplings = gRrjs_norm[BAND_INDEX, :]

new_interaction_1 = [
    [list(couplings[0]), list(indexes)],
    [list(new_couplings), list(indexes)],
]
new_interaction_2 = [
    [list(couplings[1]), list(indexes)],
    [list(new_couplings), list(indexes)],
]

# --- Build the light-matter Hamiltonian
new_H_1, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, [detuning_freq_norm, new_detuning_1], new_interaction_1
)
new_H_2, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, [detuning_freq_norm, new_detuning_2], new_interaction_2
)

new_psi0 = np.zeros(new_H_1.shape[0])
new_psi0[0] = 1

# --- Compute time evolution
new_psit_1 = expm_multiply(-1j * new_H_1, new_psi0, T0, new_tf, T_STEPS, endpoint=True)
new_psit_2 = expm_multiply(-1j * new_H_2, new_psi0, T0, new_tf, T_STEPS, endpoint=True)

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
new_mod_Cr_1 = np.abs(new_psit_1[new_time_index, new_n_emitters:]) ** 2
new_mod_Cr_2 = np.abs(new_psit_2[new_time_index, new_n_emitters:]) ** 2

z1 = new_mod_Cr_1.reshape(len(x), len(y))
z2 = new_mod_Cr_2.reshape(len(x), len(y))

# get all the emitters dynamics
new_mod_Ces = [np.abs(new_psit_1[:, 0]) ** 2, np.abs(new_psit_2[:, 0]) ** 2]
new_mod_Ce_probes = [np.abs(new_psit_1[:, 1]) ** 2, np.abs(new_psit_2[:, 1]) ** 2]

# compute the concurrence
concurrences = [
    2 * np.abs(new_mod_Ces[0] * np.conj(new_mod_Ce_probes[0])),
    2 * np.abs(new_mod_Ces[1] * np.conj(new_mod_Ce_probes[1])),
]

# ------------------------------------------------------------------------------
# Plot the previous emission for visualizing the situation
# ------------------------------------------------------------------------------
cbar_max = max(z1.max(), z2.max())

for i, (theta, z) in enumerate(zip(thetas, [z1, z2])):
    fig, ax = plt.subplots()

    sc = ax.pcolormesh(
        x,
        y,
        z,
        vmin=0,
        vmax=cbar_max,
    )

    mf.draw_arrow(ax, new_R, new_theta, new_dip_color, length=2, arrow_offset=0.2)
    mf.draw_arrow(ax, R0, theta, dip_colors[i], length=2, arrow_offset=0.2)

    cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}(t)|^2$")
    cbar.formatter.set_powerlimits((0, 0))

    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        aspect="equal",
        title=rf"$\theta_{i+1} = {np.rad2deg(theta):.0f}º$",
        xlim=(-7, 7),
        ylim=(-7, 7),
    )
    ax.title.set_color(dip_colors[i])

    fig.tight_layout()
    fig.savefig(
        f"./figures/kimble/bath_snapshot_at_{R0}_θ_{np.rad2deg(theta):.0f}º_with_probe.svg"
    )
    plt.show()

# ------------------------------------------------------------------------------
# Plot the emitters time evolution to test the previous hypothesis
# ------------------------------------------------------------------------------

concurrence_max = max(concurrences[0].max(), concurrences[1].max())

for i, theta in enumerate(thetas):
    fig, ax = plt.subplots()

    ax.plot(new_tvals, new_mod_Ces[i], c=dip_colors[i])
    ax.plot(new_tvals, new_mod_Ce_probes[i], c=new_dip_color)

    ax2 = ax.twinx()
    ax2.plot(new_tvals, concurrences[i], c=my_red, ls="--")

    ax.set(
        title=rf"$\theta_p = {np.rad2deg(new_theta):.0f}º$",
        xlabel=r"$t [\tau_1^{-1/2}]$",
        ylabel=r"$|C_e(t)|^2$",
        yscale="log",
    )
    ax.title.set_color(new_dip_color)
    ax2.set_ylabel(r"$C(t)$", color=my_red)
    ax2.tick_params(axis="y", colors=my_red)
    ax2.spines["right"].set_color(my_red)
    ax2.set_ylim((-1e-5, concurrence_max + 1e-5))
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(
        f"./figures/kimble/emitter_timeevolution_concurrence_at_{R0}_θ_{np.rad2deg(theta):.0f}º_and_{new_R}.svg"
    )
    plt.show()
