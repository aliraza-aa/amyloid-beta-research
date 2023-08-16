
import subprocess

for i in range(0, 1):
    print(f"working on replica{i}")
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/em.mdp -c ../system/simulations/replica{i}/genion_output2.gro -p ../system/simulaitons/replica{i}/topolm.top -o ../system/simulations/replica{i}/em_input.tpr", shell=True)


# gmx grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
