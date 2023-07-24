import subprocess

for i in range(2, 48):
    # subprocess.run(f"mkdir simulations/confirmation_{i}", shell=True)
    subprocess.run(
        f"mv ../confirmation{i}/confout.part0001.gro ../confirmation{i}/em.gro", shell=True)
    subprocess.run(
        f"mv ../confirmation{i}/ener.part0001.edr ../confirmation{i}/em.edr", shell=True)
    subprocess.run(
        f"mv ../confirmation{i}/md.part0001.log ../confirmation{i}/em.log", shell=True)
