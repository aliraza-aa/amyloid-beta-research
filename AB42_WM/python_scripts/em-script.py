
import subprocess

for i in range(48):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f system/em.mdp -c system/simulations/confirmation{i}/genion_output.gro -p system/topol.top -o system/simulations/confirmation{i}/em_input.tpr", shell=True)


# gmx grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
