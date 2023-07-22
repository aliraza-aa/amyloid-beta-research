import subprocess

for i in range(48):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/npt.mdp -c ../system/simulations/confirmation{i}/nvt_output.gro -r ../system/simulations/confirmation{i}/nvt_output.gro -p ../system/topol.top -o ../system/simulations/confirmation{i}/nvt_input.tpr -maxwarn 1", shell=True)

# gmx_mpi_d grompp -f system/em.mdp -c system/simulations/confirmation{i}/genion_output.gro -p system/topol.top -o system/simulations/confirmation{i}/em_input.tpr
# -maxwarn 1
# gmx_mpi_d grompp -f ../system/mdp_files/nvt.mdp -c ../system/simulations/confirmation0/confout.part0001.gro -r ../system/simulations/confirmation0/confout.part0001.gro -p ../system/topol.top -o ../system/simulations/confirmation0/nvt_input.tpr

# gmx_mpi_d grompp -f ../system/mdp_files/npt.mdp -c ../system/simulations/confirmation0/nvt_output.gro -r ../system/simulations/confirmation0/nvt_output.gro -p ../system/topol.top -o ../system/simulations/confirmation0/nvt_input.tpr
