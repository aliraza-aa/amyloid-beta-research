
import subprocess

for i in range(1, 48):
    print(f"working on replica{i}")
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/em.mdp -c ../system/simulations/replica{i}/genion_output2.gro -p ../system/simulations/replica{i}/topolm.top -o ../system/simulations/replica{i}/em_input.tpr -maxwarn 1", shell=True)


# gmx grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
