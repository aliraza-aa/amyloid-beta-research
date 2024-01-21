import os


# Replacing missing files
abr = ["ha", "hn", "nh", "ca", "cb", "co"]

for i in range(0, 48):
    for cs in abr:
        if not os.path.exists(f"../system/simulations/confirmation{i}/MISTATUS.cs_{cs}cs_{cs}.{i}"):
            # print(f"File {cs} does not exist in confimration{i}")
            with open(f"../system/simulations/confirmation{i}/MISTATUS.cs_{cs}cs_{cs}.{i}", 'w') as file_to_write:
                with open(f"../system/simulations/confirmation{i}/bck.last.MISTATUS.cs_{cs}cs_{cs}.{i}") as file_to_copy:
                    for row in file_to_copy:
                        file_to_write.write(row)


# Correcting File Sync errors

abr = ["ha", "hn", "nh", "ca", "cb", "co"]

for i in range(0, 48):
    for cs in abr:
        with open(f"../system/simulations/confirmation{i}/MISTATUS.cs_{cs}cs_{cs}.{i}", 'r+') as file_to_read:
            for row in file_to_read:
                row = row.strip().split(' ')
                if row[0] == '#!':
                    continue
                elif row[0] == str(3420):
                    break
                elif row[0] == str(3425):
                    with open(f"../system/simulations/confirmation{i}/bck.last.MISTATUS.cs_{cs}cs_{cs}.{i}", "r") as file_to_copy:
                        with open(f"../system/simulations/confirmation{i}/MISTATUS.cs_{cs}cs_{cs}.{i}", 'w') as file_to_write:
                            for row in file_to_copy:
                                file_to_write.write(row)
                            break
                else:
                    print(row[0])
                    print(
                        f"an error occured with confirmation{i} for filetype {cs}")
