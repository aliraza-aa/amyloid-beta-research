
#Example for restarting simulations:


#STEP 0: backup state files, example below assumes 48 replicas
export date1=$(date '+%d-%m-%Y')

for i in {1..47}; do mv /home/zcbtar9/Scratch/AB42_test/system/simulations_final/replica${i}/* /home/zcbtar9/projects/amyloid-beta-research/AB42_WM/system/simulations_final/replica${i}/; done
for i in {1..47}; do cp /home/zcbtar9/Scratch/AB42_test/system/simulations_final/production_run/* /home/zcbtar9/projects/amyloid-beta-research/AB42_WM/system/simulations_final/production-run-${date1}/; done

#make a state backup file for the date within each replica directory
for i in {0..47}; do mkdir replica${i}/state_backup_${date1}; done

#within each directory move newest state files into new backup folder for that date
for i in {0..47}; do mv replica${i}/state_step*.cpt replica${i}/state_backup_${date1}; done

#check largest index with python
# to find the checkpoint file with the largest number
ls -lav | tail -5

#find largest index for state file, then dump it to get timestep
for i in `seq 0 47`; do gmx_mpi dump -cp replica${i}/state_backup_${date1}/state_step123572040.cpt > replica${i}/state_backup_${date1}/dump.log; done


#use SED to check the timestep from dump.log
for i in `seq 0 47`; do sed '19q;d' replica${i}/state_backup_${date1}/dump.log; done

#I just made a note of the previous timestep
#first time you do this you won’t have this.
PREVIOUS t = 59767.680000
NOW t = 78802.080000
NOW t = 97585.680000
NOW t = 116358.080000
NOW t = 134681.680000
NOW t = 153354.480000
NOW t = 172402.480000
NOW t = 190557.280000
NOW t = 209303.280000
NOW t = 228271.28   0000
NOW t = 247144.080000
#STEP 1:reality check that HILLS files look good

#checking hills all same number of lines. 
wc -l production_run/HILLS* #wordcount -lines for all HILLS. NOTE!!! that some hills might have different header lengths

#check header lengths
head -n 20 HILLS_0 #this will show you the first 20 lines of HILLS_0 

#difference in line length for hills = timesteps +n_restarts*header_length

#check HILLS timesteps.

#make loop to do this for all hills (see loop examples above), first column should be timestep

something like: tail -48 HILLS…


#this will not be there the first run: 

for i in {0..47}; do rm replica${i}/state.cpt; done

#IMPORTANT!!! move highest index file to state.cpt

for i in {0..47}; do cp replica${i}/state_backup_${date1}/state_step123572040.cpt replica${i}/state.cpt; done

for i in {0..47}; do mv /home/zcbtar9/projects/amyloid-beta-research/AB42_WM/system/simulations_final/replica${i}/state.cpt /home/zcbtar9/Scratch/AB42_test/system/simulations_final/replica${i}/; done
for i in {0..47}; do mv /home/zcbtar9/projects/amyloid-beta-research/AB42_WM/system/simulations_final/replica${i}/production_run_input.tpr /home/zcbtar9/Scratch/AB42_test/system/simulations_final/replica${i}/; done

#double check submit script:
#-cpi state.cpt (see commented out line on run file)
#-maxh #this should be a few hours shorter than runtime

#double check that RESTART keyword is in PLUMED.DAT file

#should be ready to go!