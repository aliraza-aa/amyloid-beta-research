import subprocess

for i in range(48):
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/ions.mdp -c ../system/gro_files/topol{i}.gro -p system/topol.top -o system/simulations/replica{i}/genion_input.tpr", shell=True)

    p = subprocess.Popen(f"gmx_mpi_d genion -s system/simulations/confirmation{i}/genion_input.tpr -o system/simulations/confirmation{i}/genion_output.gro -pname NA -nname CL -neutral",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate(input="13\n".encode())
    if p.returncode != 0:
        print(out)
        print(err)

    # subprocess.run(
    #     f"gmx genion -s system/simulations/confirmation{i}/genion_input.tpr -o system/simulations/confirmation{i}/genion_output.gro -pname NA -nname CL -neutral", shell=True)


# -maxwarn 1
# gmx grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr (mdp file is a parameters file)
# gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral
