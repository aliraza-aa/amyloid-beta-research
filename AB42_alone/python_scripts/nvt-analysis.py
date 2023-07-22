import subprocess

for i in range(48):

    p = subprocess.Popen(f"gmx_mpi_d energy -f system/simulations/confirmation{i}/nvt.edr -o system/simulations/confirmation{i}/nvt_temperature.xvg",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate(input="16 0\n".encode())
    if p.returncode != 0:
        print(out)
        print(err)
