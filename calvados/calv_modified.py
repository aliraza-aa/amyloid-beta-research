# import matplotlib_inline.backend_inline
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
from openmm import app, unit
import openmm
import mdtraj as md
from main import BlockAnalysis
from localcider.sequenceParameters import SequenceParameters
from numpy import linalg
from scipy.optimize import curve_fit
import scipy.stats as scs
import pandas as pd
import itertools
import numpy as np
import os
import shutil
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--name', type=str, required=True,
                    help='name of the simulation')
parser.add_argument('--Hc6', type=float, required=True, help='Charge on His6')
parser.add_argument('--Hc13', type=float, required=True,
                    help='Charge on His13')
parser.add_argument('--Hc14', type=float, required=True,
                    help='Charge on His14')

args = parser.parse_args()

print(f"starting simulation: {args.name}")
print(f'The charge on HIS6 is: {args.Hc6}')
print(f'The charge on HIS13 is: {args.Hc13}')
print(f'The charge on HIS14 is: {args.Hc14}')

print('Installing libraries...')

# matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')

NAME = args.name

SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV"
if " " in SEQUENCE:
    SEQUENCE = ''.join(SEQUENCE.split())
    print('Blank character(s) found in the provided sequence. Sequence has been corrected, but check for integrity:')
    print(SEQUENCE)
    print('\n')

Temperature = 278
Ionic_strength = 0.1397


charged_N_terminal_amine = True
charged_C_terminal_carboxyl = True
charged_histidine = False
Nc = 1 if charged_N_terminal_amine == True else 0
Cc = 1 if charged_C_terminal_carboxyl == True else 0

Hc6 = args.Hc6
Hc13 = args.Hc13
Hc14 = args.Hc14

pH = 4
pKa = 6

if charged_histidine == True:
    print('Define pH and pKa to set the charge of Histidines according to the Henderson-Hasselbalch equation.')
    pH = float(pH)
    pKa = float(pKa)
    Hc = 1/(1+10**(pH-pKa))
if charged_histidine == False:
    Hc = 0

np.savetxt('env_settings.txt', np.array([Temperature, Ionic_strength, Hc, Nc, Cc]).T,
           header='temperature ionic_strength, His_charge, N_term_charge, C_term_charge')

# Need to store metadata prior to condacolab restarting the kernel
f = open('seq.fasta', 'w')
f.write('>{:s}\n{:s}'.format(NAME, SEQUENCE))
f.close()


residues = pd.read_csv('residues.csv')
residues = residues.set_index('one')

# Simulation_time = "AUTO"
Simulation_time = 1

N_res = len(SEQUENCE)
N_save = 700 if N_res < 150 else int(np.ceil(3e-4*N_res**2)*1000)


def genParamsLJ(df, seq, Nc, Cc):
    fasta = seq.copy()
    r = df.copy()
    if Nc == 1:
        r.loc['X'] = r.loc[fasta[0]]
        r.loc['X', 'MW'] += 2
        fasta[0] = 'X'
    if Cc == 1:
        r.loc['Z'] = r.loc[fasta[-1]]
        r.loc['Z', 'MW'] += 16
        fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    lj_eps = 0.2*4.184
    lj_sigma = pd.DataFrame((r.sigmas.values+r.sigmas.values.reshape(-1, 1))/2,
                            index=r.sigmas.index, columns=r.sigmas.index)
    lj_lambda = pd.DataFrame((r.lambdas.values+r.lambdas.values.reshape(-1, 1))/2,
                             index=r.lambdas.index, columns=r.lambdas.index)
    return lj_eps, lj_sigma, lj_lambda, fasta, types


def genParamsDH(df, seq, temp, ionic, Nc, Cc, Hc6, Hc13, Hc14):
    kT = 8.3145*temp*1e-3
    fasta = seq.copy()
    r = df.copy()

    # Set the charge on HIS based on the pH of the protein solution
    # r.loc['H', 'q'] = Hc
    if Nc == 1:
        r.loc['X'] = r.loc[fasta[0]]
        r.loc['X', 'q'] = r.loc[seq[0], 'q'] + 1.
        fasta[0] = 'X'
    if Cc == 1:
        r.loc['Z'] = r.loc[fasta[-1]]
        r.loc['Z', 'q'] = r.loc[seq[-1], 'q'] - 1.
        fasta[-1] = 'Z'
    # Calculate the prefactor for the Yukawa potential

    def fepsw(T): return 5321/T+233.76-0.9297 * \
        T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT

    # yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    yukawa_eps = []
    for i in range(len(fasta)):
        a = fasta[i]
        if i == 5:
            print(fasta[i])
            yukawa_eps.append(Hc6 * np.sqrt(lB*kT))
        elif i == 12:
            print(fasta[i])
            yukawa_eps.append(Hc13 * np.sqrt(lB*kT))
        elif i == 13:
            print(fasta[i])
            yukawa_eps.append(Hc14 * np.sqrt(lB*kT))
        else:
            yukawa_eps.append(r.loc[a].q*np.sqrt(lB*kT))

    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    return yukawa_eps, yukawa_kappa


def genDCD(name, eqsteps=10):
    """ 
    Generates coordinate and trajectory
    in convenient formats
    """
    traj = md.load("{:s}/pretraj.dcd".format(name),
                   top="{:s}/top.pdb".format(name))
    traj = traj.image_molecules(inplace=False, anchor_molecules=[
                                set(traj.top.chain(0).atoms)], make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0, 0]/2
    traj[int(eqsteps):].save_dcd("{:s}/traj.dcd".format(name))


def simulate(residues, name, seq, temp, ionic, Nc, Cc, Hc6, Hc13, Hc14, nsteps, stride=1e3, eqsteps=1000):
    os.mkdir(name)

    lj_eps, _, _, fasta, types = genParamsLJ(residues, seq, Nc, Cc)
    yukawa_eps, yukawa_kappa = genParamsDH(
        residues, seq, temp, ionic, Nc, Cc, Hc6, Hc13, Hc14)

    N = len(fasta)
    L = (N-1)*0.38+4

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    top = md.Topology()
    pos = []
    chain = top.add_chain()
    pos.append([[0, 0, L/2+(i-N/2.)*.38] for i in range(N)])
    for resname in fasta:
        residue = top.add_residue(resname, chain)
        top.add_atom(resname, element=md.element.carbon, residue=residue)
    for i in range(chain.n_atoms-1):
        top.add_bond(chain.atom(i), chain.atom(i+1))
    md.Trajectory(np.array(pos).reshape(N, 3), top, 0, [L, L, L], [
                  90, 90, 90]).save_pdb('{:s}/top.pdb'.format(name))

    pdb = app.pdbfile.PDBFile('{:s}/top.pdb'.format(name))

    system.addParticle((residues.loc[seq[0]].MW+2)*unit.amu)
    for a in seq[1:-1]:
        system.addParticle(residues.loc[a].MW*unit.amu)
    system.addParticle((residues.loc[seq[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(
        energy_expression+'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/2)^12-(0.5*(s1+s2)/2)^6')
    yu = openmm.openmm.CustomNonbondedForce(
        'q*(exp(-kappa*r)/r - exp(-kappa*4)/4); q=q1*q2')
    yu.addGlobalParameter('kappa', yukawa_kappa/unit.nanometer)
    yu.addPerParticleParameter('q')

    ah.addGlobalParameter('eps', lj_eps*unit.kilojoules_per_mole)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    for a, e in zip(seq, yukawa_eps):
        yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
        ah.addParticle([residues.loc[a].sigmas*unit.nanometer,
                       residues.loc[a].lambdas*unit.dimensionless])

    for i in range(N-1):
        hb.addBond(i, i+1, 0.38*unit.nanometer, 8033 *
                   unit.kilojoules_per_mole/(unit.nanometer**2))
        yu.addExclusion(i, i+1)
        ah.addExclusion(i, i+1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4*unit.nanometer)
    ah.setCutoffDistance(2*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    # serialized_system = XmlSerializer.serialize(system)
    # outfile = open('system.xml','w')
    # outfile.write(serialized_system)
    # outfile.close()

    integrator = openmm.openmm.LangevinIntegrator(
        temp*unit.kelvin, 0.01/unit.picosecond, 0.010*unit.picosecond)  # 10 fs timestep

    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(
        pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed'))

    check_point = '{:s}/restart.chk'.format(name)

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(
            '{:s}/pretraj.dcd'.format(name), int(stride), append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(
            '{:s}/pretraj.dcd'.format(name), int(stride)))

    simulation.reporters.append(app.statedatareporter.StateDataReporter('{:s}/traj.log'.format(name), int(stride),
                                                                        potentialEnergy=True, temperature=True, step=True, speed=True, elapsedTime=True, separator='\t'))

    simulation.step(nsteps)

    simulation.saveCheckpoint(check_point)


# @title <b><font color='#A79AB2'>Sequence analysis Toolbox</font></b>


def calc_seq_prop(seq, residues, Nc, Cc, Hc):
    """df: DataFrame to be populated with sequence properties
    r: DataFrame of aa-specific parameters"""
    seq = list(seq).copy()
    fasta_kappa = np.array(seq.copy())
    N = len(seq)
    r = residues.copy()

    # calculate properties that do not depend on charges
    fK = sum([seq.count(a) for a in ['K']])/N
    fR = sum([seq.count(a) for a in ['R']])/N
    fE = sum([seq.count(a) for a in ['E']])/N
    fD = sum([seq.count(a) for a in ['D']])/N
    faro = sum([seq.count(a) for a in ['W', 'Y', 'F']])/N
    mean_lambda = np.mean(r.loc[seq].lambdas)

    pairs = np.array(list(itertools.combinations(seq, 2)))
    pairs_indices = np.array(list(itertools.combinations(range(N), 2)))
    # calculate sequence separations
    ij_dist = np.diff(pairs_indices, axis=1).flatten().astype(float)
    # calculate lambda sums
    ll = r.lambdas.loc[pairs[:, 0]].values+r.lambdas.loc[pairs[:, 1]].values
    # calculate SHD
    beta = -1
    shd = np.sum(ll*np.power(np.abs(ij_dist), beta))/N
    SeqOb = SequenceParameters(''.join(seq))
    omega = SeqOb.get_kappa_X(grp1=['F', 'Y', 'W'])

    # fix charges
    if Nc == 1:
        r.loc['X'] = r.loc[seq[0]]
        r.loc['X', 'q'] = r.loc[seq[0], 'q'] + 1.
        seq[0] = 'X'
        if r.loc['X', 'q'] > 0:
            fasta_kappa[0] = 'K'
        else:
            fasta_kappa[0] = 'A'
    if Cc == 1:
        r.loc['Z'] = r.loc[seq[-1]]
        r.loc['Z', 'q'] = r.loc[seq[-1], 'q'] - 1.
        seq[-1] = 'Z'
        if r.loc['Z', 'q'] < 0:
            fasta_kappa[-1] = 'D'
        else:
            fasta_kappa[-1] = 'A'
    if Hc < 0.5:
        r.loc['H', 'q'] = 0
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'A'
    elif Hc >= 0.5:
        r.loc['H', 'q'] = 1
        fasta_kappa[np.where(np.array(seq) == 'H')[0]] = 'K'

    # calculate properties that depend on charges
    pairs = np.array(list(itertools.combinations(seq, 2)))
    # calculate charge products
    qq = r.q.loc[pairs[:, 0]].values*r.q.loc[pairs[:, 1]].values
    # calculate SCD
    scd = np.sum(qq*np.sqrt(ij_dist))/N
    SeqOb = SequenceParameters(''.join(fasta_kappa))
    kappa = SeqOb.get_kappa()
    fcr = r.q.loc[seq].abs().mean()
    ncpr = r.q.loc[seq].mean()

    return np.around([fK, fR, fE, fD, faro, mean_lambda, shd, omega, scd, kappa, fcr, ncpr], 3)


# @title <b><font color='#A79AB2'>Simulation analysis Toolbox</font></b>

def autoblock(cv, multi=1, plot=False):
    block = BlockAnalysis(cv, multi=multi)
    block.SEM()

    if plot == True:
        plt.errorbar(block.stat[..., 0], block.stat[..., 1],
                     block.stat[..., 2], fmt='', color='k', ecolor='0.5')
        plt.scatter(block.bs, block.sem, zorder=10, c='tab:red')
        plt.xlabel('Block size')
        plt.ylabel('SEM')
        plt.show()

    return block.av, block.sem, block.bs


def error_ratio(v1, v2, e1, e2):
    ratio = v1/v2
    return ratio*np.sqrt((e1/v1)**2+(e2/v2)**2)


def GyrTensor(t, residues, seq):
    fasta = list(seq)
    masses = residues.loc[fasta, 'MW'].values
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis, :, np.newaxis], axis=1)/masses.sum()
    # calculate residue-cm distances
    si = t.xyz - cm[:, np.newaxis, :]
    q = np.einsum('jim,jin->jmn', si*masses.reshape(1, -1, 1), si)/masses.sum()
    trace_q = np.trace(q, axis1=1, axis2=2)
    # calculate rg
    rgarray = np.sqrt(trace_q)
    # calculate traceless matrix
    mean_trace = np.trace(q, axis1=1, axis2=2)/3
    q_hat = q - mean_trace.reshape(-1, 1, 1)*np.identity(3).reshape(-1, 3, 3)
    # calculate asphericity
    Delta_array = 3/2*np.trace(q_hat**2, axis1=1, axis2=2)/(trace_q**2)
    # calculate oblateness
    S_array = 27*linalg.det(q_hat)/(trace_q**3)
    # calculate ensemble averages
    block_tr_q_hat_2 = BlockAnalysis(
        np.trace(q_hat**2, axis1=1, axis2=2), multi=1)
    block_tr_q_hat_2.SEM()
    block_tr_q_2 = BlockAnalysis(trace_q**2, multi=1)
    block_tr_q_2.SEM()
    block_det_q_hat = BlockAnalysis(linalg.det(q_hat), multi=1)
    block_det_q_hat.SEM()
    block_tr_q_3 = BlockAnalysis(trace_q**3, multi=1)
    block_tr_q_3.SEM()
    Delta = 3/2*block_tr_q_hat_2.av/block_tr_q_2.av
    S = 27*block_det_q_hat.av/block_tr_q_3.av
    Delta_err = 3/2*error_ratio(block_tr_q_hat_2.av,
                                block_tr_q_2.av, block_tr_q_hat_2.sem, block_tr_q_2.sem)
    S_err = 27*error_ratio(block_det_q_hat.av, block_tr_q_3.av,
                           block_det_q_hat.sem, block_tr_q_3.sem)
    return rgarray, Delta_array, S_array, Delta, S, Delta_err, S_err


def Rij(traj):
    pairs = traj.top.select_pairs('all', 'all')
    d = md.compute_distances(traj, pairs)
    nres = traj.n_atoms
    ij = np.arange(2, nres, 1)
    diff = [x[1]-x[0] for x in pairs]
    dij = np.empty(0)
    for i in ij:
        dij = np.append(dij, np.sqrt((d[:, diff == i]**2).mean().mean()))

    def f(x, R0, v): return R0*np.power(x, v)
    popt, pcov = curve_fit(f, ij[ij > 5], dij[ij > 5], p0=[.4, .5])
    nu = popt[1]
    nu_err = pcov[1, 1]**0.5
    R0 = popt[0]
    R0_err = pcov[0, 0]**0.5
    return ij, dij, nu, nu_err, R0, R0_err


def HALR(r, s, l): return 4*0.8368*l*((s/r)**12-(s/r)**6)
def HASR(r, s, l): return 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)


def HA(r, s, l): return np.where(r < 2**(1/6)*s, HASR(r, s, l), HALR(r, s, l))
def HASP(r, s, l, rc): return np.where(r < rc, HA(r, s, l)-HA(rc, s, l), 0)


def calcEnergyMap(t, df, seq, rc):
    indices = t.top.select_pairs('all', 'all')
    # exclude >1, was used to exclude bonded pairs
    mask = np.abs(indices[:, 0]-indices[:, 1]) > 1
    indices = indices[mask]
    # distances between pairs for each frame
    d = md.compute_distances(t, indices)
    # d[d>rc] = np.inf
    pairs = np.array(list(itertools.combinations(list(seq), 2)))
    pairs = pairs[mask]
    sigmas = 0.5*(df.loc[pairs[:, 0]].sigmas.values +
                  df.loc[pairs[:, 1]].sigmas.values)
    lambdas = 0.5*(df.loc[pairs[:, 0]].lambdas.values +
                   df.loc[pairs[:, 1]].lambdas.values)
    emap = np.zeros(pairs.shape[0])
    switch = np.zeros(pairs.shape[0])
    emap = np.nansum(
        HASP(d, sigmas[np.newaxis, :], lambdas[np.newaxis, :], rc), axis=0)
    switch = np.nansum((.5-.5*np.tanh((d-sigmas[np.newaxis, :])/.3)), axis=0)
    return indices, emap/d.shape[0], switch/d.shape[0]


def Ree(t):
    return md.compute_distances(t, atom_pairs=np.array([[0,  len(list(t.top.atoms))-1]]))[..., 0]


def maps(traj, residues, seq):
    # energy maps
    df_emap = pd.DataFrame(index=range(traj.n_atoms),
                           columns=range(traj.n_atoms), dtype=float)
    df_cmap = pd.DataFrame(index=range(traj.n_atoms),
                           columns=range(traj.n_atoms), dtype=float)
    pairs, emap, switch = calcEnergyMap(traj, residues, seq, 2.0)
    for k, (i, j) in enumerate(pairs):
        df_emap.loc[i, j] = emap[k]
        df_emap.loc[j, i] = emap[k]
        df_cmap.loc[i, j] = switch[k]
        df_cmap.loc[j, i] = switch[k]
    return df_emap, df_cmap


def kde(a):
    min_ = np.min(a)
    max_ = np.max(a)
    x = np.linspace(min_, max_, num=100)
    d = scs.gaussian_kde(a, bw_method="silverman").evaluate(x)
    u = np.average(a)
    return x, d/np.sum(d), u


# @title <b><font color='#F26419'>2 - Sequence analysis</font></b>
df = pd.DataFrame(columns=['fK', 'fR', 'fE', 'fD', 'fARO',
                  'Mean_lambda', 'SHD', 'Omega_ARO', 'SCD', 'kappa', 'FCR', 'NCPR'])
f = open('seq.fasta', 'r').readlines()
NAME = f[0][1:].strip()
SEQUENCE = f[1].strip()
Temperature, Ionic_strength, Hc, Nc, Cc = np.loadtxt(
    'env_settings.txt', unpack=True)

df.loc[NAME] = calc_seq_prop(SEQUENCE, residues, Nc, Cc, Hc)

df


# @title <b><font color='#45B69C'>3.1 - Run MD simulation</font></b>
# @markdown Simulation time (ns):


if Simulation_time == "AUTO":
    nsteps = 1010*N_save
    print('AUTO simulation length selected. Running for {} ns'.format(nsteps*0.01/1000))
else:
    nsteps = float(Simulation_time)*1000/0.01//N_save*N_save
try:
    shutil.rmtree(NAME)
except:
    pass
simulate(residues, NAME, list(SEQUENCE), temp=Temperature, ionic=Ionic_strength,
         Nc=Nc, Cc=Cc, Hc6=Hc6, Hc13=Hc13, Hc14=Hc14, nsteps=nsteps, stride=N_save, eqsteps=10)

genDCD(NAME, eqsteps=10)
# @title <b><font color='#45B69C'>3.2 - Simulation analysis</font></b>
traj = md.load_dcd('{:s}/traj.dcd'.format(NAME),
                   top='{:s}/top.pdb'.format(NAME))
rg_array, D_array, S_array, D, S, D_err, S_err = GyrTensor(
    traj, residues, SEQUENCE)

rg, rg_err, _ = autoblock(rg_array)
rg_hist = kde(rg_array)
D_hist = kde(D_array)
S_hist = kde(S_array)

ree_array = Ree(traj)
ree, ree_err, _ = autoblock(ree_array)
ree_hist = kde(ree_array)

ij, dij, nu, nu_err, R0, R0_err = Rij(traj)

df_emap, df_cmap = maps(traj, residues, SEQUENCE)

# Plot results
mpl.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(2, 3, figsize=(
    7, 3.5), facecolor='w', dpi=300, layout='tight')
axs = axs.flatten()

axs[0].plot(rg_hist[0], rg_hist[1])
top = rg_hist[1].max()+0.1*rg_hist[1].max()
axs[0].vlines(rg, 0, top)
axs[0].set_xlabel(r'$R_g$ (nm)')
axs[0].set_ylabel(r'$p(R_g)$')
axs[0].set_ylim(0, top)
axs[0].fill_between([rg-rg_err, rg+rg_err], 0, top, alpha=0.3)

axs[1].plot(D_hist[0], D_hist[1])
top = D_hist[1].max()+0.1*D_hist[1].max()
axs[1].vlines(D, 0, top)
axs[1].set_xlabel(r'$\Delta$')
axs[1].set_ylabel(r'$p(\Delta)$')
axs[1].set_ylim(0, top)
axs[1].fill_between([D-D_err, D+D_err], 0, top, alpha=0.3)

axs[2].plot(S_hist[0], S_hist[1])
top = S_hist[1].max()+0.1*S_hist[1].max()
axs[2].vlines(S, 0, top)
axs[2].set_xlabel(r'$S$')
axs[2].set_ylabel(r'$p(S)$')
axs[2].set_ylim(0, top)
axs[2].fill_between([S-S_err, S+S_err], 0, top, alpha=0.3)

axs[3].plot(ree_hist[0], ree_hist[1])
top = ree_hist[1].max()+0.1*ree_hist[1].max()
axs[3].vlines(ree, 0, top)
axs[3].set_xlabel(r'$R_{ee}$ (nm)')
axs[3].set_ylabel(r'$p(R_{ee})$')
axs[3].set_ylim(0, top)

axs[4].plot(ij, dij)
dij_fit = R0*np.power(ij, nu)
axs[4].plot(ij, dij_fit, c='0.3', ls='dashed', label='Fit')
axs[4].set_xlabel('$|i-j|$')
axs[4].set_ylabel(r'$\sqrt{\langle R_{ij}^2 \rangle}$ (nm)')
axs[4].text(0.05, 0.9, r'$\nu$={:.2f}'.format(nu), horizontalalignment='left',
            verticalalignment='center', transform=axs[4].transAxes, fontsize=10)
axs[4].legend(loc='lower right')

im = axs[5].imshow(df_emap*1e3, extent=[1, df_emap.shape[0], 1, df_emap.shape[0]],
                   origin='lower', aspect='equal', vmin=-9, vmax=0, cmap=plt.cm.Blues_r)
cb = plt.colorbar(
    im, ax=axs[5], label=r'$U$ (J mol$^{-1}$)', fraction=0.05, pad=0.04)
cb.set_ticks([0, -3, -6, -9])
axs[5].set_xlabel('Residue #')
axs[5].set_ylabel('Residue #')

plt.savefig('conformational_properties.pdf', dpi=300, facecolor='w',
            edgecolor='w', orientation='portrait', bbox_inches='tight')

df_means = pd.DataFrame(data=np.c_[[rg, rg_err], [ree, ree_err], [nu, nu_err], [D, D_err], [
                        S, S_err]], columns=['<Rg> (nm)', '<Ree> (nm)', 'nu', '<Delta>', '<S>'], index=['Value', 'Error'])
df_means


# @title <b><font color='#E3B505'>4 - Download results</font></b>
try:
    os.remove('{}/pretraj.dcd'.format(NAME))
    os.remove('{}/restart.chk'.format(NAME))
except:
    pass
shutil.copy('env_settings.txt', '{}/env_settings.txt'.format(NAME))
try:
    os.mkdir('{}_data'.format(NAME))
except:
    pass
shutil.copytree('{:s}'.format(NAME),
                '{:s}_data/SIMULATION'.format(NAME), dirs_exist_ok=True)
try:
    shutil.move('conformational_properties.pdf',
                '{:s}_data/conformational_properties.pdf'.format(NAME))
except:
    pass

pd.DataFrame(data=np.c_[rg_array, ree_array, D_array, S_array], columns=[
             'Rg (nm)', 'Ree (nm)', 'Delta', 'S']).to_csv('{:s}_data/time_series_Rg_Ree_Delta_S.csv'.format(NAME))
pd.DataFrame(data=np.c_[ij, dij, dij_fit], columns=['ij', 'Rij (nm)', 'Rij_fit (nm)']).to_csv(
    '{:s}_data/scaling_profile.csv'.format(NAME))
df.to_csv('{:s}_data/sequence_properties.csv'.format(NAME))
df_means.to_csv('{:s}_data/mean_structural_parameters.csv'.format(NAME))
df_emap.to_csv('{:s}_data/energy_map.csv'.format(NAME))

zipper = 'zip -r {:s}_data {:s}_data.zip'.format(NAME, NAME)
# subprocess.run(zipper.split())
