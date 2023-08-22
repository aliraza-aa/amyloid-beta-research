
#Example for restarting simulations:


#STEP 0: backup state files, example below assumes 48 replicas

date1=$(date '+%d-%m-%Y')
#make a state backup file for the date within each replica directory
for i in {0..47}; do mkdir replica${i}/state_backup_${date1}; done

#within each directory move newest state files into new backup folder for that date
for i in {0..47}; do mv replica${i}/state_step*.cpt replica${i}/state_backup_${date1}; done

#check largest index with python
#find largest index for state file, then dump it to get timestep
for i in `seq 0 47`; do gmx_mpi_d dump -cp replica${i}/state_backup_${date1}/state_step4590040.cpt > replica${i}/state_backup_${date1}/dump.log; done 


#use SED to check the timestep from dump.log
for i in `seq 0 47`; do sed '19q;d' replica${i}/state_backup_${date1}/dump.log; done

#I just made a note of the previous timestep
PREVIOUS t =  #first time you do this you won’t have this.
NOW  t = 9180.080000

#STEP 1:reality check that HILLS files look good

#checking hills all same number of lines. 
wc -l HILLS* #wordcount -lines for all HILLS. NOTE!!! that some hills might have different header lengths

#check header lengths
head -20 HILLS_0 #this will show you the first 20 lines of HILLS_0 

#difference in line length for hills = timesteps +n_restarts*header_length

#check HILLS timesteps .

#make loop to do this for all hills (see loop examples above), first column should be timestep

something like: tail -48 HILLS…


#this will not be there the first run: for i in {0..47}; do mv r${i}/state.cpt r${i}/state_backup_07_09_20; done

#IMPORTANT!!! move highest index file to state.cpt

for i in {0..47}; do scp replica${i}/state_backup_${date1}/state_step4590040.cpt replica${i}/state.cpt; done


#double check submit script:
#-cpi state.cpt (see commented out line on run file)
#-maxh #this should be a few hours shorter than runtime

#double check that RESTART keyword is in PLUMED.DAT file

#should be ready to go!