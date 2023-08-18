import subprocess

for i in range(0, 1):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/production_run.mdp -c ../system/simulations/replica{i}/npt_output.gro -p ../system/simulations/replica{i}/topolm.top -o ../system/simulations/replica{i}/production_run_input.tpr -maxwarn 1", shell=True)


# gmx_mpi_d grompp -f ../system/mdp_files/production_run.mdp -c ../system/simulations/replica0/npt_output.gro -r ../system/simulations/replica0/npt_output.gro -p ../system/topol.top -o ../system/simulations/confirmation0/production_run_input.tpr
