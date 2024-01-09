
##### Code example 1 #####

displacement_vectors_norm = np.asarray([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 3]])
displacement_vectors_factored = np.multiply(displacement_vectors_norm, 0.5)

dim[0] *= si_si_distance * AREA_SCALING
dim[1] *= si_si_distance * AREA_SCALING

dim_x *= si_si_distance * AREA_SCALING
dim_y *= si_si_distance * AREA_SCALING
triangle_raft_si_crds = np.multiply(monolayer_crds, si_si_distance * AREA_SCALING)
dict_sio = {}
for i in range(int(n_nodes * 3 / 2), int(n_nodes * 5 / 2)):
    dict_sio['{:}'.format(i)] = []
for i in range(monolayer_harmpairs.shape[0]):
    atom_1 = int(monolayer_harmpairs[i, 0])
    atom_2 = int(monolayer_harmpairs[i, 1])
    atom_1_crds = triangle_raft_si_crds[atom_1, :]
    atom_2_crds = triangle_raft_si_crds[atom_2, :]

    v = pbc_v(atom_1_crds, atom_2_crds)
    norm_v = np.divide(v, np.linalg.norm(v))

    grading = [abs(np.dot(norm_v, displacement_vectors_norm[i, :])) for i in range(displacement_vectors_norm.shape[0])]
    selection = grading.index(min(grading))
    if abs(grading[selection]) < 0.1:

        unperturbed_oxygen_0_crds = np.add(atom_1_crds, np.divide(v, 2))
        oxygen_0_crds = np.add(unperturbed_oxygen_0_crds, displacement_vectors_factored[selection, :])

    else:
        oxygen_0_crds = np.add(atom_1_crds, np.divide(v, 2))

    if oxygen_0_crds[0] > dim_x:
        oxygen_0_crds[0] -= dim_x
    elif oxygen_0_crds[0] < 0:
        oxygen_0_crds[0] += dim_x
    if oxygen_0_crds[1] > dim_y:
        oxygen_0_crds[1] -= dim_y
    elif oxygen_0_crds[1] < 0:
        oxygen_0_crds[1] += dim_y

    if i == 0:
        triangle_raft_o_crds = np.asarray(oxygen_0_crds)
        triangle_raft_harmpairs = np.asarray([[i, atom_1 + n_nodes * 3 / 2],
                                              [i, atom_2 + n_nodes * 3 / 2]])
        dict_sio['{:}'.format(int(atom_1 + n_nodes * 3 / 2))].append(i)
        dict_sio['{:}'.format(int(atom_2 + n_nodes * 3 / 2))].append(i)
    else:
        triangle_raft_o_crds = np.vstack((triangle_raft_o_crds, oxygen_0_crds))
        triangle_raft_harmpairs = np.vstack((triangle_raft_harmpairs, np.asarray([[i, atom_1 + n_nodes * 3 / 2],
                                                                                  [i, atom_2 + n_nodes * 3 / 2]])))
        dict_sio['{:}'.format(int(atom_1 + n_nodes * 3 / 2))].append(i)
        dict_sio['{:}'.format(int(atom_2 + n_nodes * 3 / 2))].append(i)

for i in range(int(n_nodes * 3 / 2), int(n_nodes * 5 / 2)):
    for j in range(2):
        for k in range(j + 1, 3):
            if i == int(n_nodes * 3 / 2) and j == 0 and k == 1:
                triangle_raft_o_harmpairs = np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])
            else:
                triangle_raft_o_harmpairs = np.vstack((triangle_raft_o_harmpairs, np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])))
            triangle_raft_harmpairs = np.vstack((triangle_raft_harmpairs, np.array([dict_sio['{:}'.format(i)][j], dict_sio['{:}'.format(i)][k]])))

triangle_raft_crds = np.vstack((triangle_raft_o_crds, triangle_raft_si_crds))

##### Code example 2 #####

DISPLACEMENT_VECTORS_NORM = np.array([[1, 0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
DISPLACEMENT_VECTORS_FACTORED = DISPLACEMENT_VECTORS_NORM * 0.5

triangle_raft_lammps_data = LAMMPSData.from_netmc_data(netmc_data, NetworkType.A, atom_label="Si", atomic_mass=28.1, atom_style="atomic")
triangle_raft_lammps_data.scale_coords(SI_SI_DISTANCE_BOHR * AREA_SCALING)
dimension_ranges = triangle_raft_lammps_data.dimensions[:, 1] - triangle_raft_lammps_data.dimensions[:, 0]
triangle_raft_lammps_data.add_atom_label("O")
triangle_raft_lammps_data.add_mass("O", 15.995)
for si_atom in triangle_raft_lammps_data.atoms:
    bonded_si_atoms = triangle_raft_lammps_data.get_bonded_atoms(si_atom)
    for bonded_si_atom in bonded_si_atoms:
        vector_between_si_atoms = pbc_vector(si_atom.coords, bonded_si_atom.coords, dimension_ranges)
        normalized_vector = vector_between_si_atoms / np.linalg.norm(vector_between_si_atoms)
        dot_product_grades = np.abs(np.dot(normalized_vector, DISPLACEMENT_VECTORS_NORM.T))
        selected_vector_index = np.argmin(dot_product_grades)
        midpoint = (si_atom.coords + vector_between_si_atoms / 2) % dimension_ranges

        if dot_product_grades[selected_vector_index] < 0.1:
            oxygen_coord = midpoint + DISPLACEMENT_VECTORS_FACTORED[selected_vector_index] % dimension_ranges
        else:
            oxygen_coord = midpoint
        triangle_raft_lammps_data.add_atom(LAMMPSAtom("O", oxygen_coord))
        triangle_raft_lammps_data.add_structure(LAMMPSBond(si_atom, triangle_raft_lammps_data.atoms[-1]))
        triangle_raft_lammps_data.add_structure(LAMMPSBond(bonded_si_atom, triangle_raft_lammps_data.atoms[-1]))
        triangle_raft_lammps_data.add_structure(LAMMPSAngle(si_atom, triangle_raft_lammps_data.atoms[-1], bonded_si_atom))
