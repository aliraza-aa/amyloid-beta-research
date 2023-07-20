import subprocess

for i in range(48):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"gmx grompp -f system/nvt.mdp -c system/gro_files/topol{i}.gro -r system/gro_files/topol{i}.gro -p system/topol.top -o system/simulations/confirmation{i}/nvt_input.tpr", shell=True)


# -maxwarn 1
