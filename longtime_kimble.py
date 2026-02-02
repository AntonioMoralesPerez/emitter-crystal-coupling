import numpy as np

import meep as mp
from meep import mpb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import src as mf

# ...
import pythtb as tb

mpl.rcParams.update(
    {
        # Figure
        "figure.figsize": (7, 6),  # default figure size in inches, for one-column width
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
        # Tick length/width
        "xtick.major.size": 4,
        "xtick.major.width": 1,
        "ytick.major.size": 4,
        "ytick.major.width": 1,
        # Grid (off by default, but can be enabled)
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,
    }
)


# INPUTS --------------------------------------------------------------------- #
NUM_BANDS = 2

EPSILON_INPUT_FILE = "data/kimble/epsilon.h5"
DIRNAME_OUTPUT = "data/kimble/"  # directory to save outputs

TOLERANCE = 1e-12  # tolerance for the MPB calculations

N_CELLS = 61  # number of unit cells in each direction

# .. define number of points per direction for the Monkhorst-Pack grid
NK = 64

# angles for the dipole moments of the quantum emitters and its dimensionless value
THETA_1 = (90) * np.pi / 180
THETA_2 = (135) * np.pi / 180
P_DIMLESS = (
    1e-2  # TODO: this needs to be corrected in the paper as soon as you fix one value
)

# Position of the quantum emitter in units of the lattice constant 'a'
# list of positions are also allowed
R0 = [0.0, 0.0]

# small inputs for the tight-binding model
LAT = [[1, 0], [0, 1]]
ORB = [[0, 0]]

BAND_INDEX = 0  # index of the band to consider (0-based)

# coefficients for the tight-binding model
TAU_0 = 0.07755639
TAU_1 = -0.01993698
TAU_2 = 0.00053966

# path in the Brillouin zone for band structure calculation
K_PATH = [[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]]
# define HSPs labels
K_LABEL = [r"$\Gamma$", r"X", r"M", r"$\Gamma$"]

# TODO: you need to adjust this depending on the coupling strength
T0 = 0  # initial time for the field snapshots
TF = 500  # final time for the field snapshots
T_STEPS = 1000  # number of time steps for the field snapshots
TIME_INDEX = 500  # time index to consider for the field snapshots

# ---------------------------------------------------------------------------- #

# Read some data from the epsilon.h5 file
d = mf.read_epsilon_file(EPSILON_INPUT_FILE)
resolution = d["resolution"]
lattice_vecs = d["lattice_vectors"]
lattice_size = d["lattice_size"]

# define the tight-binding model for the infinite lattice
infiniteModel = tb.tb_model(2, 2, LAT, ORB)

# onsite term
infiniteModel.set_onsite([TAU_0])

# 1st nn
infiniteModel.set_hop(TAU_1, 0, 0, [1, 0])
infiniteModel.set_hop(TAU_1, 0, 0, [0, 1])

# 2nd nn
infiniteModel.set_hop(TAU_2, 0, 0, [1, 1])
infiniteModel.set_hop(TAU_2, 0, 0, [1, -1])

# %% Compute bands for the infinite model

### Compute bands along high-symmetry path ...

# creates a path between the HSPs defined before with 500 points between them
k_vec, k_dist, k_node = infiniteModel.k_path(K_PATH, 500, report=False)

# solve the infinite model in the considered path
evals = infiniteModel.solve_all(k_vec)

## .. Select the detuning frequency
# select the emitters detunings, we are picking the value of the frequency at the X point
# where there is the saddle point of the band and a high density of states
detuning_freq = np.sqrt(evals[BAND_INDEX][k_node[1] == k_dist].item())
print(f"Detuning frequency selected at X point: {detuning_freq:.6f} (2πc/a)")

# compute the bands with MPB for comparison

geometry_lattice = mp.Lattice(
    size=lattice_size, basis1=lattice_vecs[0], basis2=lattice_vecs[1]
)

ms = mpb.ModeSolver(
    num_bands=NUM_BANDS,
    resolution=mp.Vector3(*resolution),  # type: ignore[arg-type]
    epsilon_input_file=EPSILON_INPUT_FILE,
    tolerance=TOLERANCE,
    mesh_size=1,
    geometry_lattice=geometry_lattice,
    k_points=mf.array_to_Vector3(k_vec),
)

ms.run_te()

# all frequency bands
freqs = ms.all_freqs

# %% Read epsilon data for plotting
md = mpb.MPBData(rectify=False, periods=2)
d = mf.convert_epsilon("./data/kimble/epsilon.h5", md)

x1 = d["positions-lattice"][:, 0]
y1 = d["positions-lattice"][:, 1]
z1 = d["epsilon_values"]

# ---------------------------------------------------------------------------
# PERFORM THE FOURIER TRANSFORMATION OF THE COUPLING FUNCTION g(r0,r)
# ---------------------------------------------------------------------------

# Load the fields and k-points from the MPB calculation
# This file was generated previously in `kimble.ipynb`
data = mf.load("./data/kimble/fields/64_meshgrid_data.pkl")

# Extract relevant quantities
omegas = np.asarray(data["freqs"])  # Frequencies (Nk, Nb)
Eks = np.asarray(data["Eks"])  # Electric fields, shape: (Nr, Nb, Nk, 3)
kpts = np.asarray(data["kpts"])  # k-points in cartesian coordinates, shape: (Nk, 3)

# Define the dipole moments for the emitters
dipole_moment_1 = P_DIMLESS * np.array([np.cos(THETA_1), np.sin(THETA_1), 0])
dipole_moment_2 = P_DIMLESS * np.array([np.cos(THETA_2), np.sin(THETA_2), 0])

# ---------------------------------------------------------------------------
# Select the band of interest and position r0
# ---------------------------------------------------------------------------

E_k_r0 = Eks[0, :, :, :]  # Shape: (Nb, Nk, 3)
omega_k = omegas  # Shape: (Nk, Nb)

# ---------------------------------------------------------------------------
# Define the real-space grid for interpolation
# ---------------------------------------------------------------------------
xg = np.linspace(-5, 5, 101)
yg = np.linspace(-5, 5, 101)
rpts = mf.meshgrid_points(xg, yg)  # Shape: (Nx*Ny, 2)

# ---------------------------------------------------------------------------
# Compute g(r0,r) for each dipole moment, shape (Nb, Nr)
# ---------------------------------------------------------------------------
gr0rs_1 = mf.compute_gr0r(E_k_r0, dipole_moment_1, omega_k, kpts, rpts)
gr0rs_2 = mf.compute_gr0r(E_k_r0, dipole_moment_2, omega_k, kpts, rpts)

# ---------------------------------------------------------------------------
# Pick the band of interest from g(r0,r), shape (Nr,)
# ---------------------------------------------------------------------------
gr0r_1 = gr0rs_1[BAND_INDEX, :]
gr0r_2 = gr0rs_2[BAND_INDEX, :]

fig, axs = plt.subplots(2, 2)  # two-column width
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

# ----------------------------------------------
# Global dipole settings
# ----------------------------------------------
x0, y0 = R0
dip_colors = ["#1f77b4", "#ff7f0e"]  # [dipole 1, dipole 2]
arrow_length = 0.4

# =====================================================
# --- First quadrant: dielectric structure (a)
# =====================================================
ax = axs[0, 0]
sc1 = ax.scatter(x1, y1, c=z1, marker="o", cmap="copper_r", s=mf.marker_size(len(x1)))
fig.colorbar(sc1, ax=ax, label=r"$\epsilon(\mathbf{r})$")
ax.set_title(r"$\epsilon(\mathbf{r})$")
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.set_aspect("equal")
ax.text(-0.2, 1.1, panel_labels[0], transform=ax.transAxes)

mf.draw_arrow(ax, R0, THETA_1, dip_colors[0], r"$\theta_1$", length=arrow_length)
mf.draw_arrow(ax, R0, THETA_2, dip_colors[1], r"$\theta_2$", length=arrow_length)

# =====================================================
# --- Second quadrant: TE bands (b)
# =====================================================
ax = axs[0, 1]
for n in evals:
    # Ensure no negative values due to numerical errors
    n_safe = np.clip(np.real(n), 0, None)
    ax.plot(k_dist, np.sqrt(n_safe), color="r", linestyle="--")
ax.plot(k_dist, freqs, color="k")
ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(K_LABEL)
for k in k_node:
    ax.axvline(x=k, linewidth=0.5, color="k")
ax.set_ylabel(r"$\omega \, [c/a]$")
ax.set_title("TE modes")
ax.text(-0.2, 1.05, panel_labels[1], transform=ax.transAxes)

# =====================================================
# --- Third quadrant: g(r0,r) dipole 1 (c)
# =====================================================
# define a maximum for the color scale
vmax = max(np.abs(gr0r_1).max(), np.abs(gr0r_2).max())
ax = axs[1, 0]
sc2 = ax.scatter(
    rpts[:, 0],
    rpts[:, 1],
    c=np.abs(gr0r_1),  # take absolute value for plotting
    s=mf.marker_size(len(rpts)),
    marker="o",
    cmap="inferno",
    vmin=0,
    vmax=vmax,
)
cbar1 = fig.colorbar(sc2, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{r}_0)| \, [c/a]$")
cbar1.formatter.set_powerlimits((0, 0))  # type: ignore

ax.set_title(
    r"$\theta_1 = {:.0f}^\circ$".format(THETA_1 * 180 / np.pi), color=dip_colors[0]
)
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.set_aspect("equal")
ax.text(-0.2, 1.1, panel_labels[2], transform=ax.transAxes)

# =====================================================
# --- Fourth quadrant: g(r0,r) dipole 2 (d)
# =====================================================
ax = axs[1, 1]
sc3 = ax.scatter(
    rpts[:, 0],
    rpts[:, 1],
    c=np.abs(gr0r_2),  # take absolute value for plotting
    s=mf.marker_size(len(rpts)),
    marker="o",
    cmap="inferno",
    vmin=0,
    vmax=vmax,
)
cbar2 = fig.colorbar(sc3, ax=ax, label=r"$|g_\mathbf{r}(\mathbf{r}_0)| \, [c/a]$")
cbar2.formatter.set_powerlimits((0, 0))  # type: ignore

ax.set_title(
    r"$\theta_2 = {:.0f}^\circ$".format(THETA_2 * 180 / np.pi), color=dip_colors[1]
)
ax.set_xlabel(r"$x/a$")
ax.set_ylabel(r"$y/a$")
ax.set_aspect("equal")
ax.text(-0.2, 1.1, panel_labels[3], transform=ax.transAxes)

# --- Layout and save ---
fig.tight_layout()
fig.savefig("./figures/paper_figure_2.png", dpi=300)

# -------------------------------
# Normalize parameters
# -------------------------------
alpha_norm = TAU_0 / np.abs(TAU_1)
j1_norm = TAU_1 / np.abs(TAU_1)
j2_norm = TAU_2 / np.abs(TAU_1)
detuning_freq_norm = detuning_freq / np.sqrt(np.abs(TAU_1))

# -------------------------------
# Initialize tight-binding model
# -------------------------------
infiniteModel = tb.tb_model(dim_k=2, dim_r=2, lat=LAT, orb=ORB)

# -------------------------------
# Onsite term
# -------------------------------
infiniteModel.set_onsite([alpha_norm])

# -------------------------------
# First nearest neighbors (1st nn)
# -------------------------------
first_nn_vectors = np.array([[1, 0], [0, 1]])
for vec in first_nn_vectors:
    infiniteModel.set_hop(j1_norm, 0, 0, vec)

# -------------------------------
# Second nearest neighbors (2nd nn)
# -------------------------------
second_nn_vectors = np.array([[1, 1], [1, -1]])
for vec in second_nn_vectors:
    infiniteModel.set_hop(j2_norm, 0, 0, vec)

# -------------------------------------------------
# --- Generate 2D k-point grid in the Brillouin zone ---
# -------------------------------------------------
kx = np.linspace(-0.5, 0.5, NK)
ky = np.linspace(-0.5, 0.5, NK)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

# Flatten grid for evaluation
k_points = np.column_stack((kx_grid.ravel(), ky_grid.ravel()))

# -------------------------------------------------
# --- Compute band energies for the selected band ---
# -------------------------------------------------
band_energies = np.array(
    [infiniteModel.solve_one(k)[BAND_INDEX] for k in k_points],  # type: ignore[index]
    dtype=float,
)

# -------------------------------------------------
# --- Clip all negative values (remove numerical noise) ---
# -------------------------------------------------
band_energies_clipped = np.clip(band_energies, a_min=0.0, a_max=None)

# -------------------------------------------------
# --- Reshape and compute omega = sqrt(E) ---
# -------------------------------------------------
omega = np.sqrt(band_energies_clipped).reshape(NK, NK)

# load the data from a file
data_fields = mf.load("./data/kimble/fields/64_meshgrid_data.pkl")
Eks = data_fields["Eks"]  # shape: (Nr, Nb, Nk, 3)
Eks_r0 = Eks[0, :, :, :]  # shape: (Nb, Nk, 3)
omega_ks = data_fields["freqs"]  # shape: (Nk, Nb)
kpts = data_fields["kpts"]  # shape: (Nk, 3)

# .. Select the dipole moment
thetas = [THETA_1, THETA_2]
dipoles = [P_DIMLESS * np.array([np.cos(theta), np.sin(theta), 0]) for theta in thetas]

# %% Finite model definition

# Cut the system
temporalModel = infiniteModel.cut_piece(
    N_CELLS, fin_dir=0, glue_edgs=False
)  # cutout along 1st direction
finiteModel = temporalModel.cut_piece(
    N_CELLS, fin_dir=1, glue_edgs=False
)  # cutout along 2nd direction

# %% Compute g(r0,rj) on the finite grid
# NOTE: here we make an interpolation of g(r0,r) on the finite grid defined by pythonTB
#       in principle, Nk should be equal to N_CELLS, but here we take Nk < N_CELLS

if N_CELLS % 2 == 0:
    xjg = np.arange(-N_CELLS / 2, N_CELLS / 2, 1)
    yjg = np.arange(-N_CELLS / 2, N_CELLS / 2 + 1, 1)
else:
    xjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
    yjg = np.arange(-(N_CELLS // 2), N_CELLS // 2 + 1, 1)
rjpts = mf.meshgrid_points(xjg, yjg, indexing="xy")

gr0rjs = [
    mf.compute_gr0r(
        E_k_r0=Eks_r0,
        dipole_moment=dipole,
        omega_k=omega_ks,
        kpts=kpts,
        rpts=rjpts,
    )
    for dipole in dipoles
]

# we want to normalize the couplings by the 1-st nn coupling strength, since TAU_1 has units
# of frequency² while g(r0,rj) has units of frequency, we need to divide by sqrt(|TAU_1|)
gr0rjs_norm = [gr0rj / np.sqrt(np.abs(TAU_1)) for gr0rj in gr0rjs]

# %% Build the light-matter Hamiltonian

# ... non-local couplings
values = [gr0rj_norm[BAND_INDEX, :] for gr0rj_norm in gr0rjs_norm]

# small check before proceeding
assert len(values[0]) == len(
    values[1]
), "Unexpected different lengths for the couplings!"

indexes = np.arange(0, len(values[0]), 1)
couplings_1 = [[list(values[0]), list(indexes)]]
couplings_2 = [[list(values[1]), list(indexes)]]

# build the light-matter Hamiltonian
H_total_1, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_1
)
H_total_2, _ = mf.make_light_matter_Hamiltonian(
    finiteModel, detuning_freq_norm, couplings_2
)

# %% Compute time evolution

from scipy.sparse.linalg import expm_multiply

tvals = np.linspace(T0, TF, T_STEPS)

# Define initial condition
assert (
    H_total_1.shape == H_total_2.shape
), "Unexpected different shapes for the Hamiltonians!"

psi0 = np.zeros(H_total_1.shape[0])  # type: ignore
psi0[0] = 1

# Compute time evolution
psit_1 = expm_multiply(-1j * H_total_1, psi0, T0, TF, T_STEPS, endpoint=True)
psit_2 = expm_multiply(-1j * H_total_2, psi0, T0, TF, T_STEPS, endpoint=True)

# Prepare data to be plotted ...

# get emitter component of the overall wavefunction
mod_Ce_squared_1 = np.abs(psit_1[:, 0]) ** 2
mod_Ce_squared_2 = np.abs(psit_2[:, 0]) ** 2

# compute the entanglement for both cases
entropy_1 = mf.binary_entropy(mod_Ce_squared_1)
entropy_2 = mf.binary_entropy(mod_Ce_squared_2)

# get positions in Cartesian coordinates (requires find_positions() defined above)
positions = mf.find_positions(finiteModel._orb, finiteModel._lat)

# get photonic component of the overall wave function at selected time, i.e, tvals[time_index]
mod_Cr_squared_1 = np.abs(psit_1[TIME_INDEX, 1 : np.shape(psit_1)[1]]) ** 2
mod_Cr_squared_2 = np.abs(psit_2[TIME_INDEX, 1 : np.shape(psit_2)[1]]) ** 2

x = positions[:, 0]
y = positions[:, 1]
z1 = mod_Cr_squared_1
z2 = mod_Cr_squared_2

fig, axs = plt.subplots(2, 2)  # no figsize unless absolutely needed
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

ax = axs[0, 0]
c = ax.contourf(kx_grid, ky_grid, omega, levels=12, cmap="viridis")
fig.colorbar(c, ax=ax, label=r"$\omega_\mathbf{k}\,[J_1^{-1}]$", pad=0.07)

ax.contour(
    kx_grid, ky_grid, omega, levels=[detuning_freq_norm], colors="k", linewidths=2.0
)

ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title=rf"$\omega_0/J_1 = {detuning_freq:.2f}$")
ax.set_aspect("equal")
ax.text(-0.2, 1.1, panel_labels[0], transform=ax.transAxes)

ax1 = axs[0, 1]

colors = ["#1f77b4", "#ff7f0e"]  # consistent with figure 2

(l1,) = ax1.plot(
    tvals,
    mod_Ce_squared_1,
    lw=3,
    color=colors[0],
    label=r"$|C_e(t)|^2,\;\theta=90^\circ$",
)
(l2,) = ax1.plot(
    tvals,
    mod_Ce_squared_2,
    lw=2,
    color=colors[1],
    alpha=0.9,
    label=r"$|C_e(t)|^2,\;\theta=135^\circ$",
)

ax1.set(
    xlabel=r"$t\,[J_1^{-1}]$",
    ylabel=r"$|C_e(t)|^2$",
    # ylim=(0, 1.05),
    title=rf"$\omega_0/J_1 = {detuning_freq:.2f}$",
)

ax2 = ax1.twinx()
(l3,) = ax2.plot(
    tvals, entropy_1, ls="--", color=colors[0], lw=3, label=r"$S(t),\;\theta=90^\circ$"
)
(l4,) = ax2.plot(
    tvals,
    entropy_2,
    ls="--",
    color=colors[1],
    alpha=0.9,
    lw=2.0,
    label=r"$S(t),\;\theta=135^\circ$",
)
ax2.set_ylabel(r"$S(t)$")

# --- combined legend ---
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]

ax1.legend(lines, labels, loc="upper right", fontsize=7, frameon=False)
ax1.text(-0.2, 1.05, panel_labels[1], transform=ax1.transAxes)

vmax = max(z1.max(), z2.max())

for ax, z, theta, color, label in zip(
    axs[1], [z1, z2], [90, 135], colors, panel_labels[2:]
):
    sc = ax.scatter(
        x,
        y,
        c=z,
        s=0.5 * mf.marker_size(len(x)),
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        marker="o",
    )
    ax.set(xlabel=r"$x/a$", ylabel=r"$y/a$", title=rf"$\theta = {theta}^\circ$")
    ax.title.set_color(color)
    ax.set_aspect("equal")
    ax.text(-0.2, 1.1, label, transform=ax.transAxes)
    cbar = fig.colorbar(sc, ax=ax, label=r"$|C_\mathbf{r}(t)|^2$")
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore

# Layout adjustment and save
fig.tight_layout()
fig.savefig("./figures/paper_figure_3.png", dpi=300)

# plot the population dynamics in logarithmic scale to see the decay better
fig, ax = plt.subplots()

colors = ["#1f77b4", "#ff7f0e"]

ax.plot(
    tvals,
    mod_Ce_squared_1,
    lw=3,
    color=colors[0],
    label=r"$|C_e(t)|^2,\;\theta=90^\circ$",
)

ax.plot(
    tvals,
    mod_Ce_squared_2,
    lw=2,
    color=colors[1],
    alpha=0.9,
    label=r"$|C_e(t)|^2,\;\theta=135^\circ$",
)

ax.set(
    xlabel=r"$t\,[J_1^{-1}]$",
    ylabel=r"$|C_e(t)|^2$",
    title=rf"$\omega_0/J_1 = {detuning_freq:.2f}$",
    yscale="log",
    xscale="log",
)

ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("./figures/emitter_population_log.png", dpi=300)
