import numpy as np
from scipy.sparse import csc_matrix  # for sparse matrices


def make_light_matter_Hamiltonian(
    model, emitter_detunings=[0], couplings=[[[0.5], [0]]]
):
    # ...
    #
    # The idea is to create a sparce matrix (this is relevant to speed up calculations or even made
    # them attainable). To do that you need to provide the non-zero matrix elements H_{ij} and their
    # corresponding the indexes i,j.
    #
    # First, I create the bath Hamiltonian H_bath, given by the sum of H_onsite and H_hoppings. To do
    # that, I use the ouptuts from a finite pythonTB model. The emitter's and interaction's Hamiltonians
    # all together using the `emitter_detuning` and `couplings` inputs.
    #
    # ...

    ###  Definitions
    N_orbitals = model.get_num_orbitals()
    N_emitters = len(couplings)

    H_dimension = N_orbitals + N_emitters

    if np.size(emitter_detunings) == 1 and N_emitters > 1:
        emitter_detunings = emitter_detunings * N_emitters
    # ...
    # NOTE: if a single emitter_detunings is given, but more than one emitter is considered, we assume all  to
    #       emitters to feature the same detuning value.
    # ...

    ### Bath Hamiltonian ......................................................

    ## On-site energies ...

    # define the on-site Hamiltonian as a sparce matrix
    v_onsite = model._site_energies
    i_onsite = np.arange(0, N_orbitals, 1) + N_emitters
    j_onsite = np.arange(0, N_orbitals, 1) + N_emitters

    H_onsite = csc_matrix(
        (v_onsite, (i_onsite, j_onsite)), shape=(H_dimension, H_dimension)
    )

    ## Hopping terms ...

    # define the hopping Hamiltonian as a sparce matrix
    hopping_vals_aux1 = np.array(model._hoppings)[:, 0]  # get the hopping values
    hopping_indx_aux1 = np.array(model._hoppings)[:, [1, 2]]  # get the hopping indexes
    hopping_vals_aux2 = np.conjugate(
        hopping_vals_aux1
    )  # conjugate of the hopping values
    hopping_indx_aux2 = hopping_indx_aux1[:, [1, 0]]  # exchange the i and j indexes

    hopping_vals = np.hstack([hopping_vals_aux1, hopping_vals_aux2])
    hopping_indx = np.vstack([hopping_indx_aux1, hopping_indx_aux2]) + N_emitters

    # ...
    # NOTE_1: The aux quantities are defined to make the Hamiltonian Hermitian. The `model._hoppings`
    #         output accounts for the coupling between sites in one direction, but the conjugate term
    #         must be considered as well.
    # NOTE_2: The "+ N_emitters" in `hopping_indx` is there to shift the indexes so that H_bath occupies
    #         the lower right corner in the total Hamiltonian.
    # ...

    # define sparse hoppings matrix
    v_hopping = hopping_vals
    i_hopping = hopping_indx[:, 0]
    j_hopping = hopping_indx[:, 1]

    H_hoppings = csc_matrix(
        (v_hopping, (i_hopping, j_hopping)), shape=(H_dimension, H_dimension)
    )

    ## Sum both parts and take the square root of the total Hamiltonian
    H_bath = H_onsite + H_hoppings

    ### Emitters + coupling Hamiltonian .......................................

    # define the emitters and interaction Hamiltonians as a sparse matrix

    # define the non-zero values and indexes for the emitters' Hamiltonian
    v_emitters = np.array(emitter_detunings)
    i_emitters = np.arange(0, N_emitters, 1)
    j_emitters = np.arange(0, N_emitters, 1)

    # define the non-zero values and indexes for the interactions' Hamiltonian
    coupling_vals_list = []
    coupling_indx_i_list = []
    coupling_indx_j_list = []
    for i in range(N_emitters):
        coupling_vals_list.append(couplings[i][0])
        coupling_indx_i_list.append(
            list(i * np.ones(np.shape(couplings[i][1]), dtype=int))
        )
        coupling_indx_j_list.append(list(np.array(couplings[i][1]) + N_emitters))

    coupling_vals_aux1 = np.hstack(coupling_vals_list)
    coupling_indx_i_aux1 = np.hstack(coupling_indx_i_list)
    coupling_indx_j_aux1 = np.hstack(coupling_indx_j_list)

    coupling_vals_aux2 = np.conjugate(coupling_vals_aux1)
    coupling_indx_i_aux2 = coupling_indx_j_aux1
    coupling_indx_j_aux2 = coupling_indx_i_aux1

    v_coupling = np.hstack([coupling_vals_aux1, coupling_vals_aux2])
    i_coupling = np.hstack([coupling_indx_i_aux1, coupling_indx_i_aux2])
    j_coupling = np.hstack([coupling_indx_j_aux1, coupling_indx_j_aux2])

    # ...
    # NOTE: I use similar procedure to the one use to create the H_hoppings matrix. This depends on
    #       the particular way the coupling object is defined though. Therefore, if we change that,
    #       we will need to revisit this part.
    # ...

    # merge the emitters' and interactions' values and indexes
    v_all = np.hstack((v_emitters, v_coupling))
    i_all = np.hstack((i_emitters, i_coupling))
    j_all = np.hstack((j_emitters, j_coupling))

    H_emitters_and_coupling = csc_matrix(
        (v_all, (i_all, j_all)), shape=(H_dimension, H_dimension)
    )

    ### Total Hamiltonian
    H_total = H_bath + H_emitters_and_coupling

    ### Output ...
    return H_total
