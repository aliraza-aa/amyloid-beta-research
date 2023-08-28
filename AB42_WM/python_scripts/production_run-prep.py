import subprocess

for i in range(0):
    # subprocess.run(f"mkdir simulations_test4/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi grompp -f ../system/mdp_files/run.mdp -c ../system/simulations_test4/replica{i}/npt_output.gro -p ../system/simulations/replica{i}/topolm.top -o ../system/simulations_test9/replica{i}/production_run_input.tpr -maxwarn 1", shell=True)


# gmx_mpi_d grompp -f ../system/mdp_files/production_run.mdp -c ../system/simulations_test4/replica0/npt_output.gro -r ../system/simulations_test4/replica0/npt_output.gro -p ../system/topol.top -o ../system/simulations_test4/confirmation0/production_run_input.tpr
