import subprocess

for i in range(2, 48):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"rm ../confirmation{i}/production_run.part0001.edr", shell=True)
    subprocess.run(
        f"rm ../confirmation{i}/production_run.part0001.log", shell=True)
    subprocess.run(
        f"rm ../confirmation{i}/production_run.part0001.trr", shell=True)
    subprocess.run(
        f"rm ../confirmation{i}/production_run.part0001.xtc", shell=True)
