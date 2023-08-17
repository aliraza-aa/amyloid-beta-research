import subprocess

for i in range(0, 1):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/nvt.mdp -c ../system/simulations/replica{i}/em_output.gro -r ../system/simulations/replica{i}/em_output.gro -p ../system/simulations/replica{i}/topolm.top -o ../system/simulations/replica{i}/nvt_input.tpr -maxwarn 1", shell=True)

# gmx_mpi_d grompp -f system/em.mdp -c system/simulations/confirmation{i}/genion_output.gro -p system/topol.top -o system/simulations/confirmation{i}/em_input.tpr
# -maxwarn 1
# gmx_mpi_d grompp -f ../system/mdp_files/nvt.mdp -c ../system/simulations/confirmation0/confout.part0001.gro -r ../system/simulations/confirmation0/confout.part0001.gro -p ../system/topol.top -o ../system/simulations/confirmation0/nvt_input.tpr
