import subprocess

for i in range(1, 48):
    # subprocess.run(
    #     f"gmx_mpi_d grompp -f ../system/ions.mdp -c ../system/gro_files/topol{i}.gro -p system/topol.top -o system/simulations/replica{i}/genion_input.tpr", shell=True)
    print(f"working on replica{i}")
    p = subprocess.Popen(f"gmx_mpi_d pdb2gmx -f ../system/gro_files/topol{i}.gro -o ../system/simulations/replica{i}/modified_topol{i}.gro -his -ignh",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate(input="1\n1\n0\n0\n0\n".encode())

    if p.returncode != 0:
        print(out)
        print(err)

    # subprocess.run(
    #     f"gmx genion -s system/simulations/confirmation{i}/genion_input.tpr -o system/simulations/confirmation{i}/genion_output.gro -pname NA -nname CL -neutral", shell=True)


# -maxwarn 1

# gmx_mpi_d grompp -f ../system/mdp_files/ions.mdp -c ../system/gro_files/topol0.gro -p ../system/topol.top -o ../system/simulations/replica0/genion_input.tpr
# gmx_mpi_d genion -s ../system/simulations/replica0/genion_input.tpr -o ../system/simulations/replica0/genion_output.gro -conc 0.137 -pname NA -nname CL -p ../system/topol.top
# gmx_mpi_d grompp -f ../system/mdp_files/ions.mdp -c ../system/simulations/replica0/genion_output.gro -p ../system/topol.top -o ../system/simulations/replica0/genion2_input.tpr
# gmx_mpi_d genion -s ../system/simulations/replica0/genion2_input.tpr -o ../system/simulations/replica0/genion2_output.gro -conc 0.00268 -pname K -nname CL -p ../system/topol.top
# gmx_mpi_d pdb2gmx -f gro_files/topol0.gro -o simulations/replica0/modified_topol0.gro -his -ignh

# move topol.top in all directories

# gmx grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr (mdp file is a parameters file)
# gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral
