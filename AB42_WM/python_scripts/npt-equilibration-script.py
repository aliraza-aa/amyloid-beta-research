import subprocess

for i in range(1, 48):
    print(f"\n\n working on replica{i} \n\n")
    subprocess.run(
        f"gmx_mpi_d grompp -f ../system/mdp_files/npt.mdp -c ../system/simulations/replica{i}/nvt_output.gro -r ../system/simulations/replica{i}/nvt_output.gro -p ../system/simulations/replica{i}/topolm.top -o ../system/simulations/replica{i}/npt_input.tpr -maxwarn 1", shell=True)

# gmx_mpi_d grompp -f system/em.mdp -c system/simulations/replica{i}/genion_output.gro -p system/topol.top -o system/simulations/replica{i}/em_input.tpr
# -maxwarn 1
# gmx_mpi_d grompp -f ../system/mdp_files/nvt.mdp -c ../system/simulations/confirmation0/confout.part0001.gro -r ../system/simulations/confirmation0/confout.part0001.gro -p ../system/topol.top -o ../system/simulations/confirmation0/nvt_input.tpr

# gmx_mpi_d grompp -f ../system/mdp_files/npt.mdp -c ../system/simulations/confirmation0/nvt_output.gro -r ../system/simulations/confirmation0/nvt_output.gro -p ../system/topol.top -o ../system/simulations/confirmation0/nvt_input.tpr
