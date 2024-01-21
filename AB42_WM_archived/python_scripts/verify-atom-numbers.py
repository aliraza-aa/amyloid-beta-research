import pandas as pd


ca_list = [5, 17, 27, 42, 62, 86, 103, 115, 126, 133, 154, 169, 185, 202, 219, 236, 258, 277, 293, 313, 333,
           343, 358, 370, 386, 393, 404, 418, 440, 447, 457, 476, 495, 502, 521, 538, 554, 561, 568, 584, 600, 619]


# checking that ca numbers match with gro files used to produce the run input (tpr) file for production run
# for i in range(48):
#     data = pd.read_table(f"../system/simulations/replica{i}/npt_output.gro",
#                          delim_whitespace=True, skiprows=2, header=None, nrows=627)

#     atom_type = data[1]

#     print(f"Reading replica number: {i}")

#     for atom_number in ca_list:
#         if atom_type[atom_number-1] != "CA":
#             print(f"error with atom number: {atom_number}")


# checking cbeta atom numbers for hydrophobic residues i.e. should match CVs.dat file

# cb_list = [19, 44, 171, 260, 279, 295, 315, 335, 372,
#            449, 459, 478, 504, 523, 540, 570, 586, 602, 621]
# hydrophobic_res = ["ALA", "GLY", "VAL", "LEU", "ILE", "MET", "PHE", "PRO"]

# for i in range(48):
#     data = pd.read_table(f"../system/simulations/replica{i}/npt_output.gro",
#                          delim_whitespace=True, skiprows=2, header=None, nrows=627)

#     atom_type = data[1]
#     res_names = data[0]

#     print(f"Reading replica number: {i}")

#     cb_verification_list = []
#     for i in range(627):
#         res_name = res_names[i][-3:]
#         if res_name in hydrophobic_res and atom_type[i] == "CB":
#             if (i+1) not in cb_list:
#                 print(f"error with atom_number: {i+1}")
#             else:
#                 cb_verification_list.append(i+1)
#     if cb_list != cb_verification_list:
#         print(f"error")


# checking salt bridges atom numbers

groupa = [10, 11, 12, 35, 36, 37, 108, 109, 110,
          162, 163, 164, 351, 352, 353, 363, 364, 365]
groupb = [75, 76, 77, 78, 79, 80, 81, 250, 251, 252, 253, 432, 433, 434, 435]

neg = ["ASP", "GLU"]
pos = ["ARG", "LYS"]

# for i in range(48):
#     data = pd.read_table(f"../system/simulations/replica{i}/npt_output.gro",
#                          delim_whitespace=True, skiprows=2, header=None, nrows=627)

#     atom_types = data[1]
#     res_names = data[0]

#     print(f"Reading replica number: {i}")

#     for atm in groupa:
#         res_name = res_names[atm-1][-3:]
#         atom_type = atom_types[atm-1][0]

#         if res_name in neg and atom_type in ["C", "O"]:
#             pass
#         else:
#             print(f"error with atom number: {atm}")

arg_atom_types = ["CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22"]
lys_atom_types = ["NZ", "HZ1", "HZ2", "HZ3"]

for i in range(48):
    data = pd.read_table(f"../system/simulations/replica{i}/npt_output.gro",
                         delim_whitespace=True, skiprows=2, header=None, nrows=627)

    atom_types = data[1]
    res_names = data[0]

    print(f"Reading replica number: {i}")

    for atm in groupb:
        res_name = res_names[atm-1][-3:]
        atom_type = atom_types[atm-1]

        if res_name == "ARG" and atom_type in arg_atom_types:
            pass
        elif res_name == "LYS" and atom_type in lys_atom_types:
            pass
        else:
            print(f"error with atom number: {atm}")
