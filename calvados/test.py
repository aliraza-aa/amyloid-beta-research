import mdtraj as md


# def test_load_dcd(top, dcd_path):
#     try:
#         traj = md.load(dcd_path, top=top)
#         print(f"Successfully loaded DCD file: {dcd_path}")
#     except OSError as e:
#         print(f"Failed to load DCD file: {dcd_path}")
#         print(f"Error: {e}")


# Test the specific DCD file
top = "ab40/top.pdb"
dcd_path = "ab40/pretraj.dcd"
traj = md.load("ab40/pretraj.dcd",
               top="ab40/top.pdb")

# test_load_dcd(top, dcd_path)
print(traj)
