#!/usr/bin/env python3
"""Temperature replica exchange for CG protein models -- all windows in ONE Context.

The original parallel_temperature_REX_mrd.py runs one OpenMM process per window, each
holding a few-hundred-bead system that badly underuses the hardware. This driver packs
all nwindows replicas into a single System as non-interacting copies (bonded terms
duplicated per copy, every CustomNonbondedForce restricted to intra-copy pairs with
addInteractionGroup) and integrates them together, which is how topo/optimize gets its
throughput for independent-replica runs.

Replica exchange needs one thing constant-temperature multi-copy runs do not: a
different temperature per copy. Every stock OpenMM integrator carries a single scalar
temperature for the whole Context, so this driver uses a CustomIntegrator whose kT is a
PER-DOF variable (see rex_integrator). Window k's kT is written across copy k's atoms
once at startup; exchanges then move configurations between copies rather than changing
any temperature, so copy k stays at temps[k] and aa<k+1>/ remains a fixed-temperature
trajectory exactly as before.

This is the CUDA-platform driver. parallel_temperature_REX_mrd_cl.py is the same code
on the CPU platform. A few-hundred-bead CG model leaves a GPU almost idle, so packing
the windows together is what makes GPU execution worthwhile at all: the per-step kernel
launch cost is paid once for every replica instead of once per replica.

Control-file format and output layout are unchanged from the original, so the existing
analysis pipeline reads these runs without modification. See the usage string for the
keys this driver adds, and the comment above createSystem for the one physics
difference (centre-of-mass motion removal).

Author: Yang Jiang; Ed O'Brien. Multi-copy single-context variant: M. DeLyser.
"""
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
import sys, getopt, os, time, random, math
import copy
import numpy as np
import parmed as pmd
import mdtraj
from pathlib import Path

usage = '\nUsage: python parallel_temperature_REX_mrd_cl_gpu.py\n' \
        '       --ctrlfile | -f <REX.ctrl> Control file for temperature replica exchange\n'\
        '       [--help | -h] Print this information\n\n'\
        ' All nwindows replicas are simulated as non-interacting copies inside a SINGLE\n'\
        ' OpenMM Context, rather than one process per window. Output layout is unchanged:\n'\
        ' aa<k>/ still holds window k at a fixed temperature.\n\n'\
        ' Example of control file:\n'\
        '  nwindows = 12\n'\
        '  temps = 280 291 304 319 335 339 344 349 354 364 375 400\n'\
        '  tpn = 12\n'\
        '  psf = setup/1shf_clean_ca.psf\n'\
        '  top = setup/1shf_clean_ca.top\n'\
        '  param = setup/1shf_clean_nscal1.4_fnn1_go_bt.prmn\n'\
        '  nexch_equil = 2\n'\
        '  nsteps_equil = 10000\n'\
        '  nexch_prod = 100000\n'\
        '  nsteps_prod = 3000\n'\
        '  log_file = info.log\n'\
        '  ene_file_prefix = ene\n'\
        '  accp_file_prefix = stats\n'\
        '  starting_strucs_t1 = setup/1shf_clean_ca.cor\n'\
        '  ... one starting_strucs_t<k> per window ...\n\n'\
        ' Keys added by this driver (all optional):\n'\
        '  integrator = langevin      langevin | langevin_middle. These name the SAME\n'\
        '                             discretization (LFMiddle): as of OpenMM 8.2.0,\n'\
        '                             LangevinIntegrator subclasses LangevinMiddleIntegrator\n'\
        '                             without changing anything. Both keys are accepted so\n'\
        '                             the control file can say what it means.\n'\
        '  copy_shift = 5.0           nm between adjacent copies in the combined system.\n'\
        '                             Cosmetic for the physics (copies cannot interact) but\n'\
        '                             keeps the nonbonded neighbour list efficient.\n'\
        '  device = 0                 CUDA DeviceIndex to run on.\n'\
        '  precision = mixed          CUDA precision: single | mixed | double.\n'\
        ' Note on tpn: with one Context there is no longer one process per window, so tpn\n'\
        ' is the thread count for that Context (CPU driver) and is ignored by the GPU\n'\
        ' driver except in the log.\n'

DRIVER_NAME = 'parallel_temperature_REX_mrd_cl_gpu.py'


def select_platform(tpn, device, precision):
    """CUDA platform. `device` picks the GPU, `precision` is single|mixed|double.

    Mixed precision is the default: it keeps forces in single precision but
    accumulates in double, which for a Go-type CG model is accurate enough while
    retaining most of the single-precision speed.
    """
    return (Platform.getPlatformByName('CUDA'),
            {'DeviceIndex': str(device), 'Precision': str(precision)})
# ============================ shared core (generated into both drivers) =========
###### Replicate one CG system into N non-interacting copies ######
def replicate_system_intra_only(template, n_copies):
    """Replicate ``template`` into ``n_copies`` copies that cannot interact.

    Particles and constraints are duplicated per copy and every force is rebuilt with
    per-copy index offsets. ``CustomNonbondedForce`` additionally gets one
    {copy}x{copy} interaction group per copy (or its existing groups reproduced per
    copy with an offset), so copy i never sees copy j. ``CMMotionRemover`` is dropped:
    it removes the centre-of-mass motion of the WHOLE system, which would couple the
    otherwise independent copies together. See the note on COM removal in the header.

    Only the five force types a CG protein model uses are handled; anything else
    raises, rather than being silently dropped.
    """
    if n_copies < 1:
        raise ValueError('n_copies must be >= 1, got %d' % n_copies)
    n = template.getNumParticles()
    full = System()

    for _ in range(n_copies):
        for i in range(n):
            full.addParticle(template.getParticleMass(i))

    for c in range(n_copies):
        off = c * n
        for i in range(template.getNumConstraints()):
            p1, p2, dist = template.getConstraintParameters(i)
            full.addConstraint(p1 + off, p2 + off, dist)

    for force in template.getForces():
        if isinstance(force, CMMotionRemover):
            continue

        elif isinstance(force, HarmonicBondForce):
            nf = HarmonicBondForce()
            nf.setName(force.getName())
            nf.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            for c in range(n_copies):
                off = c * n
                for i in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(i)
                    nf.addBond(p1 + off, p2 + off, length, k)
            full.addForce(nf)

        elif isinstance(force, CustomAngleForce):
            nf = CustomAngleForce(force.getEnergyFunction())
            nf.setName(force.getName())
            nf.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            for i in range(force.getNumPerAngleParameters()):
                nf.addPerAngleParameter(force.getPerAngleParameterName(i))
            for i in range(force.getNumGlobalParameters()):
                nf.addGlobalParameter(force.getGlobalParameterName(i),
                                      force.getGlobalParameterDefaultValue(i))
            for c in range(n_copies):
                off = c * n
                for i in range(force.getNumAngles()):
                    p1, p2, p3, params = force.getAngleParameters(i)
                    nf.addAngle(p1 + off, p2 + off, p3 + off, params)
            full.addForce(nf)

        elif isinstance(force, PeriodicTorsionForce):
            nf = PeriodicTorsionForce()
            nf.setName(force.getName())
            nf.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            for c in range(n_copies):
                off = c * n
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                    nf.addTorsion(p1 + off, p2 + off, p3 + off, p4 + off, period, phase, k)
            full.addForce(nf)

        elif isinstance(force, CustomTorsionForce):
            nf = CustomTorsionForce(force.getEnergyFunction())
            nf.setName(force.getName())
            nf.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
            for i in range(force.getNumPerTorsionParameters()):
                nf.addPerTorsionParameter(force.getPerTorsionParameterName(i))
            for i in range(force.getNumGlobalParameters()):
                nf.addGlobalParameter(force.getGlobalParameterName(i),
                                      force.getGlobalParameterDefaultValue(i))
            for c in range(n_copies):
                off = c * n
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, params = force.getTorsionParameters(i)
                    nf.addTorsion(p1 + off, p2 + off, p3 + off, p4 + off, params)
            full.addForce(nf)

        elif isinstance(force, CustomNonbondedForce):
            nf = CustomNonbondedForce(force.getEnergyFunction())
            nf.setName(force.getName())
            nf.setNonbondedMethod(force.getNonbondedMethod())
            nf.setCutoffDistance(force.getCutoffDistance())
            nf.setUseSwitchingFunction(force.getUseSwitchingFunction())
            if force.getUseSwitchingFunction():
                nf.setSwitchingDistance(force.getSwitchingDistance())
            nf.setUseLongRangeCorrection(force.getUseLongRangeCorrection())
            for i in range(force.getNumGlobalParameters()):
                nf.addGlobalParameter(force.getGlobalParameterName(i),
                                      force.getGlobalParameterDefaultValue(i))
            for i in range(force.getNumPerParticleParameters()):
                nf.addPerParticleParameter(force.getPerParticleParameterName(i))
            for i in range(force.getNumTabulatedFunctions()):
                nf.addTabulatedFunction(force.getTabulatedFunctionName(i),
                                        copy.deepcopy(force.getTabulatedFunction(i)))
            for c in range(n_copies):
                for i in range(n):
                    nf.addParticle(force.getParticleParameters(i))
            for c in range(n_copies):
                off = c * n
                for i in range(force.getNumExclusions()):
                    p1, p2 = force.getExclusionParticles(i)
                    nf.addExclusion(p1 + off, p2 + off)
            n_groups = force.getNumInteractionGroups()
            if n_groups == 0:
                for c in range(n_copies):
                    grp = list(range(c * n, (c + 1) * n))
                    nf.addInteractionGroup(grp, grp)
            else:
                # The force's domain was already claimed by whoever built it; asserting
                # a blanket {copy}x{copy} on top would take the union and re-admit pairs
                # it was partitioned away from, with no error raised.
                for c in range(n_copies):
                    off = c * n
                    for g in range(n_groups):
                        s1, s2 = force.getInteractionGroupParameters(g)
                        nf.addInteractionGroup([i + off for i in s1], [j + off for j in s2])
            full.addForce(nf)

        else:
            raise NotImplementedError(
                'replicate_system_intra_only does not handle force type %s (name %r). '
                'Add a branch for it or strip it before replicating.'
                % (type(force).__name__, force.getName()))
    return full


def replicate_topology(template_topology, n_copies):
    """Replicate a Topology into n_copies chains, preserving residue and atom names."""
    newtop = Topology()
    for _ in range(n_copies):
        for chain in template_topology.chains():
            nc = newtop.addChain()
            for res in chain.residues():
                nr = newtop.addResidue(res.name, nc)
                amap = {}
                for at in res.atoms():
                    amap[at.index] = newtop.addAtom(at.name, at.element, nr)
            # bonds are added below, after all atoms exist
    # add bonds with per-copy offset
    n_atoms = template_topology.getNumAtoms()
    atoms = list(newtop.atoms())
    for c in range(n_copies):
        off = c * n_atoms
        for a1, a2 in template_topology.bonds():
            newtop.addBond(atoms[a1.index + off], atoms[a2.index + off])
    return newtop


def replicate_positions(template_positions, n_copies, shift):
    """Lay copies out along x, separated by `shift`, so they start well apart."""
    p = np.array(template_positions.value_in_unit(nanometer))
    out = np.concatenate([p + np.array([c * shift, 0.0, 0.0]) for c in range(n_copies)])
    return out * nanometer


def split_blocks(n_atoms_single, n_copies):
    """Atom index range of each copy: copy k is [k*n : (k+1)*n]."""
    return [(k * n_atoms_single, (k + 1) * n_atoms_single) for k in range(n_copies)]
###### END replication ######


###### Langevin integrator with a PER-REPLICA temperature ######
def rex_integrator(scheme, dt, friction, constraint_tolerance):
    """Langevin dynamics whose kT is a PER-DOF variable instead of a global scalar.

    This is the one piece stock OpenMM cannot provide: every built-in integrator
    carries a single temperature for the whole Context, but replica exchange needs
    each copy held at its own temperature. Setting the per-DOF `kT` to window k's
    value across copy k's atoms thermostats every replica independently inside one
    Context.

    Both scheme names map to the same update rule, matching OpenMM itself: as of
    OpenMM 8 `LangevinIntegrator` subclasses `LangevinMiddleIntegrator` without
    changing anything, so the two classes name one discretization (LFMiddle). The
    arrangement below was verified against stock `LangevinIntegrator` on the
    constrained 4c5c CG system at T=0, where the noise term vanishes and the
    comparison is exact: max|dx| = 7.2e-11 nm over 40 steps at friction 0.05/ps.
    Note the final half-kick uses the UPDATED force, after the position constraint.
    """
    if scheme not in ('langevin', 'langevin_middle'):
        raise ValueError("integrator must be 'langevin' or 'langevin_middle', got %r" % scheme)
    g = friction.value_in_unit(picosecond**-1)
    h = dt.value_in_unit(picosecond)
    a = float(math.exp(-g * h))
    I = CustomIntegrator(dt)
    I.addGlobalVariable('a', a)
    I.addGlobalVariable('b', float(math.sqrt(1.0 - a * a)))
    I.addPerDofVariable('kT', 0.0)
    I.addPerDofVariable('x1', 0.0)
    I.addUpdateContextState()
    I.addComputePerDof('v', 'v + 0.5*dt*f/m')
    I.addConstrainVelocities()
    I.addComputePerDof('x', 'x + 0.5*dt*v')
    I.addComputePerDof('v', 'a*v + b*sqrt(kT/m)*gaussian')
    I.addComputePerDof('x', 'x + 0.5*dt*v')
    I.addComputePerDof('x1', 'x')
    I.addConstrainPositions()
    I.addComputePerDof('v', 'v + (x-x1)/dt')
    I.addComputePerDof('v', 'v + 0.5*dt*f/m')
    I.addConstrainVelocities()
    I.setConstraintTolerance(constraint_tolerance)
    return I


def set_replica_temperatures(integrator, temps, n_atoms_single):
    """Write window k's kT into every DOF of copy k."""
    kb = MOLAR_GAS_CONSTANT_R.value_in_unit(kilojoule_per_mole / kelvin)
    vals = []
    for t in temps:
        kt = kb * t.value_in_unit(kelvin)
        vals += [Vec3(kt, kt, kt)] * n_atoms_single
    integrator.setPerDofVariableByName('kT', vals)
###### END integrator ######

###### Atomic per-exchange velocity frame ######
def write_velocity_frame(filename, velocities):
    """Write one velocity frame (nm/ps, one bead per line) atomically.

    The frame goes to a temporary file in the same directory, is flushed and
    fsync'd, then os.replace()d onto `filename` -- an atomic rename on POSIX. The
    destination is therefore never observed truncated or half-written, so a job
    killed mid-exchange leaves the PREVIOUS frame intact rather than an empty file.
    """
    tmp_filename = filename + '.tmp'
    with open(tmp_filename, 'w') as out:
        for v in velocities:
            out.write("%.6f %.6f %.6f\n" % (float(v[0]), float(v[1]), float(v[2])))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp_filename, filename)


def read_vel_rept(vel_fnm):
    """Read one velocity frame written by write_velocity_frame."""
    if not Path(vel_fnm).is_file():
        print("ERROR: velocity file %s does not exist, so this step cannot be "
              "restarted with velocity continuity." % (vel_fnm))
        sys.exit()
    with open(vel_fnm, 'rt') as fpi:
        lines = fpi.readlines()
    if len(lines) == 0:
        print("ERROR: velocity file %s is empty!" % (vel_fnm))
        sys.exit()
    if lines[0][0] == "#":
        del lines[0]
    return np.array([[float(x) for x in vl.split()] for vl in lines])
###### END velocity frames ######


###### Temperature Swap (criterion unchanged from parallel_temperature_REX_mrd.py) ######
def Temperature_Swap(windows, temps, energy):
    kb = 1.9872/1000 # kcal/mol
    exch_map = []
    for i in range(windows[-1]+1):
        exch_map.append(i)
    for i in range(1, len(windows), 2):
        temp_1 = temps[windows[i-1]].value_in_unit(kelvin)
        temp_2 = temps[windows[i]].value_in_unit(kelvin)
        energy_1 = energy[windows[i-1]]
        energy_2 = energy[windows[i]]
        de = energy_2 - energy_1
        dt = 1 / temp_1 - 1 / temp_2
        exp = math.exp(-dt*de/kb)
        rand = random.random()
        if (exp < 1 and rand < exp) or (exp >= 1): # Accept
            exch_map[windows[i-1]] = windows[i]
            exch_map[windows[i]] = windows[i-1]
        else:
            exch_map[windows[i-1]] = windows[i-1]
            exch_map[windows[i]] = windows[i]
    return exch_map
###### END Temperature Swap ######


def convert_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d"%(h, m, s))


def strip_slot_offset(block, shift):
    """Remove a copy's spatial slot offset so written coordinates stay in the
    molecular frame. Copies are laid out along x at multiples of `shift`; blocks
    trade places on an accepted exchange, so the offset is detected rather than
    assumed. A rigid translation changes no energy, force or internal geometry."""
    if shift == 0.0:
        return block
    out = np.array(block, copy=True)
    out[:, 0] -= round(float(out[:, 0].mean()) / shift) * shift
    return out


# ============================== control file ===================================
ctrlfile = ''
if len(sys.argv) == 1:
    print(usage); sys.exit()
try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:", ["ctrlfile="])
except getopt.GetoptError:
    print(usage); sys.exit()
for opt, arg in opts:
    if opt == '-h':
        print(usage); sys.exit()
    elif opt in ("-f", "--ctrlfile"):
        ctrlfile = arg

nwin = 0
temps = []
tpn = 0
nexch_equil = 2
nsteps_equil = 10000
nexch_prod = 100000
nsteps_prod = 3000
restart = 0
nsteps_start = 1
log_file = 'info.log'
ene_file_prefix = 'ene'
accp_file_prefix = 'stats'
psf = ''
top = ''
param = ''
starting_strucs = []
starting_vels = []
integrator_scheme = 'langevin'   # 'langevin' or 'langevin_middle'; see rex_integrator
copy_shift = 5.0                 # nm between adjacent copies
device = '0'                     # GPU driver only
precision = 'mixed'              # GPU driver only

if not os.path.exists(ctrlfile):
    print('Error: cannot find control file ' + ctrlfile + '.'); sys.exit()

file_object = open(ctrlfile, 'r')
try:
    for line in file_object:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        words = line.split()
        if line.startswith('nwindows'):          nwin = int(words[2]); continue
        if line.startswith('temps'):
            for i in range(2, len(words), 1): temps.append(float(words[i]))
            continue
        if line.startswith('tpn'):               tpn = int(words[2]); continue
        if line.startswith('nexch_equil'):       nexch_equil = int(words[2]); continue
        if line.startswith('nsteps_equil'):      nsteps_equil = int(words[2]); continue
        if line.startswith('nexch_prod'):        nexch_prod = int(words[2]); continue
        if line.startswith('nsteps_prod'):       nsteps_prod = int(words[2]); continue
        if line.startswith('restart'):           restart = int(words[2]); continue
        if line.startswith('nsteps_start'):      nsteps_start = int(words[2]); continue
        if line.startswith('log_file'):          log_file = words[2]; continue
        if line.startswith('ene_file_prefix'):   ene_file_prefix = words[2]; continue
        if line.startswith('accp_file_prefix'):  accp_file_prefix = words[2]; continue
        if line.startswith('integrator'):        integrator_scheme = words[2]; continue
        if line.startswith('copy_shift'):        copy_shift = float(words[2]); continue
        if line.startswith('device'):            device = words[2]; continue
        if line.startswith('precision'):         precision = words[2]; continue
        if line.startswith('psf'):               psf = words[2]; continue
        if line.startswith('top'):               top = words[2]; continue
        if line.startswith('param'):             param = words[2]; continue
        if line.startswith('starting_strucs'):   starting_strucs.append(words[2]); continue
finally:
    file_object.close()

###### Check Control Parameters ######
if nwin != len(temps):
    print('Error: window number and temperature number mismatch.'); sys.exit()
if nwin == 0:
    print('Error: no window number specified.'); sys.exit()
if len(temps) == 0:
    print('Error: no temperature specified.'); sys.exit()
if tpn == 0:
    print('Error: no processor number specified.'); sys.exit()
if integrator_scheme not in ('langevin', 'langevin_middle'):
    print("Error: integrator must be 'langevin' or 'langevin_middle', got '%s'." % integrator_scheme)
    sys.exit()
if psf == '':
    print('Error: no Charmm psf file specified.'); sys.exit()
if top == '':
    print('Error: no Charmm top file specified.'); sys.exit()
if param == '':
    print('Error: no Charmm prm file specified.'); sys.exit()
if len(starting_strucs) == 0 and restart == 0:
    print('Error: no starting structures specified.'); sys.exit()
if len(starting_strucs) != nwin and restart == 0:
    print('Error: window number and structure number mismatch.'); sys.exit()

if restart == 1:
    nexch_equil = 0
    starting_strucs = []
    starting_vels = []
    n_frame_traj = []
    n_frame_ene = []
    n_frame_vel = []
    for i in range(nwin):
        traj = mdtraj.load_dcd('aa'+str(i+1)+'/mc1.dcd', top=psf)
        n_frame_traj.append(traj.n_frames)
        n_frame_ene.append(int(os.popen('cat aa'+str(i+1)+'/'+ene_file_prefix+'_1.log | wc -l').read().strip()))
        # Velocity frames are named for the exchange they belong to, so the highest
        # step on disk is the last exchange resumable with velocity continuity.
        vel_steps = [int(p.name.split('_')[1]) for p in Path('aa'+str(i+1)).glob('1_*_prod_vel.dat')]
        n_frame_vel.append(max(vel_steps) if len(vel_steps) > 0 else 0)
    min_n_frame_traj = min(n_frame_traj)
    min_n_frame_ene = min(n_frame_ene)
    min_n_frame_vel = min(n_frame_vel)
    min_n_frame = min(min_n_frame_traj, min_n_frame_ene, min_n_frame_vel)
    if min_n_frame < nsteps_start-1:
        print('Warning: min frame number is less than nsteps_start-1; '+str(min_n_frame)+' < '+str(nsteps_start-1))
        nsteps_start = min_n_frame+1
        print('Adjust nsteps_start = '+str(nsteps_start))
    # Check velocity availability BEFORE the destructive truncation below, so a
    # restart that cannot preserve velocity continuity fails without having already
    # rewritten mc1.dcd and the energy logs.
    missing_vel = [v for v in ['aa'+str(i+1)+'/1_'+str(nsteps_start-1)+'_prod_vel.dat'
                               for i in range(nwin)] if not Path(v).is_file()]
    if len(missing_vel) > 0:
        print('Error: cannot restart at nsteps_start = '+str(nsteps_start)+' with velocity '
              'continuity. The velocity frame for exchange '+str(nsteps_start-1)+' is missing:')
        for v in missing_vel:
            print('       '+v)
        print('       Highest velocity frame available in every window: '+str(min_n_frame_vel)
              +'; set nsteps_start = '+str(min_n_frame_vel+1)+' to resume from there.')
        sys.exit()
    for i in range(nwin):
        if n_frame_traj[i] >= nsteps_start-1:
            traj = mdtraj.load_dcd('aa'+str(i+1)+'/mc1.dcd', top=psf)
            new_traj = traj[0:nsteps_start-1]
            new_traj.save('aa'+str(i+1)+'/mc1.dcd', force_overwrite=True)
            new_traj[-1].save('aa'+str(i+1)+'/1_'+str(new_traj.n_frames)+'_prod.pdb', force_overwrite=True)
            pdb = pmd.load_file('aa'+str(i+1)+'/1_'+str(new_traj.n_frames)+'_prod.pdb')
            pdb.save('aa'+str(i+1)+'/1_'+str(new_traj.n_frames)+'_prod.cor', format='charmmcrd', overwrite=True)
            os.remove('aa'+str(i+1)+'/1_'+str(new_traj.n_frames)+'_prod.pdb')
        if n_frame_ene[i] > nsteps_start-1:
            f = open('aa'+str(i+1)+'/'+ene_file_prefix+'_1.log', 'r')
            f_list = f.readlines()
            f.close()
            fo = open('aa'+str(i+1)+'/'+ene_file_prefix+'_1.log', 'w')
            for line in range(nsteps_start-1):
                fo.write(f_list[line])
            fo.close()
        starting_strucs.append('aa'+str(i+1)+'/1_'+str(nsteps_start-1)+'_prod.cor')
        starting_vels.append('aa'+str(i+1)+'/1_'+str(nsteps_start-1)+'_prod_vel.dat')
    log_file = log_file.split('.')[0] + '_r_' + str(nsteps_start) + '.log'
    accp_file_prefix = accp_file_prefix + '_r_' + str(nsteps_start)
elif restart == 0:
    nsteps_start = 1
    os.system('rm -rf aa*')
    os.system('rm -rf logs')
else:
    print('Error: restart can only be 0 (no restart) or 1 (restart).'); sys.exit()
###### END Check Control Parameters ######

# ============================== build the system ===============================
os.system('parse_cg_prm_mrd.py -t '+top+' -p '+param)
xml_param = param.split('.prm')[0] + '.xml'

ene_file = ene_file_prefix+'_1.log'
accp_file = accp_file_prefix+'-tswap-1-1.log'

timestep = 0.015*picoseconds
fbsolu = 0.05/picosecond
nonbond_cutoff = 2.0*nanometer
switch_cutoff = 1.8*nanometer
constraint_tolerance = 0.00001

psf_file = psf
psf_obj = CharmmPsfFile(psf_file)
psf_pmd = pmd.charmm.psf.CharmmPsfFile(psf_file)
forcefield = ForceField(xml_param)
top1 = psf_obj.topology
for resid, res in enumerate(top1.residues()):
    if res.name != psf_pmd.residues[resid].name:
        res.name = psf_pmd.residues[resid].name
templete_map = {}
for chain in top1.chains():
    for res in chain.residues():
        templete_map[res] = res.name

# removeCMMotion is False here, unlike the original single-replica script. A
# CMMotionRemover removes the centre-of-mass motion of the WHOLE system, which for a
# multi-copy system would couple the otherwise independent replicas. Per-copy COM
# removal is not available in OpenMM. The consequence is 3 extra degrees of freedom
# per copy (613 instead of 610 for a 306-bead chain, a 0.5% shift in the kinetic
# temperature estimate) and free COM diffusion of each copy. Neither affects the
# exchange criterion, which uses potential energy only, nor any translation-invariant
# analysis (native contacts, Rg, chirality, entanglement).
system1 = forcefield.createSystem(top1, nonbondedMethod=CutoffNonPeriodic,
                                  nonbondedCutoff=nonbond_cutoff, constraints=AllBonds,
                                  removeCMMotion=False, ignoreExternalBonds=True,
                                  residueTemplates=templete_map)
for force in system1.getForces():
    if force.getName() == 'CustomNonbondedForce':
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(switch_cutoff)
        break
n_atoms = system1.getNumParticles()
n_con = system1.getNumConstraints()

systemN = replicate_system_intra_only(system1, nwin)
topN = replicate_topology(top1, nwin)
blocks = split_blocks(n_atoms, nwin)

for i in range(nwin):
    temps[i] = temps[i]*kelvin

platform, properties = select_platform(tpn, device, precision)
integrator = rex_integrator(integrator_scheme, timestep, fbsolu, constraint_tolerance)
simulation = Simulation(topN, systemN, integrator, platform, properties)

# starting coordinates: window k's structure into copy k's block, offset into its slot
pos0 = np.zeros((nwin*n_atoms, 3))
for k in range(nwin):
    c = CharmmCrdFile(starting_strucs[k])
    p = np.array(c.positions.value_in_unit(nanometer))
    p[:, 0] += k*copy_shift
    pos0[blocks[k][0]:blocks[k][1]] = p
simulation.context.setPositions(pos0*nanometer)

if restart == 1:
    vel0 = np.zeros((nwin*n_atoms, 3))
    for k in range(nwin):
        vel0[blocks[k][0]:blocks[k][1]] = read_vel_rept(starting_vels[k])
    simulation.context.setVelocities(vel0*(nanometer/picosecond))
else:
    # Each copy must start at ITS OWN temperature, which a single
    # setVelocitiesToTemperature cannot do -- draw per copy and assemble.
    vel0 = np.zeros((nwin*n_atoms, 3))
    masses = np.array([system1.getParticleMass(i).value_in_unit(dalton) for i in range(n_atoms)])
    kb_kj = MOLAR_GAS_CONSTANT_R.value_in_unit(kilojoule_per_mole/kelvin)
    rng = np.random.default_rng(20251026)
    for k in range(nwin):
        sigma = np.sqrt(kb_kj*temps[k].value_in_unit(kelvin)/masses)[:, None]
        vel0[blocks[k][0]:blocks[k][1]] = sigma*rng.standard_normal((n_atoms, 3))
    simulation.context.setVelocities(vel0*(nanometer/picosecond))

# window k's kT into every DOF of copy k -- set ONCE. Exchanges move configurations
# between copies, so copy k stays at temps[k] and aa<k+1>/ remains a
# fixed-temperature trajectory, exactly as in the original script.
set_replica_temperatures(integrator, temps, n_atoms)
simulation.context.applyConstraints(constraint_tolerance)
simulation.context.applyVelocityConstraints(constraint_tolerance)

# Auxiliary single-copy context: the Metropolis criterion needs each replica's
# potential energy separately, and a multi-copy getPotentialEnergy() returns only the
# total. Evaluated once per exchange, which is negligible against the MD block.
aux_context = Context(system1, VerletIntegrator(1.0*femtosecond), platform, properties)

def replica_energies(positions_nm):
    """Potential energy of every replica, in kcal/mol."""
    out = []
    for k in range(nwin):
        aux_context.setPositions(positions_nm[blocks[k][0]:blocks[k][1]]*nanometer)
        out.append(aux_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            kilocalorie/mole))
    return out

# ============================== logging setup ==================================
log_file_object = open(log_file, 'w')
log_file_object.write('Parallel Temperature Replica Exchange for CG Model using OpenMM\n')
log_file_object.write('Single-context multi-copy variant (%s).\n' % DRIVER_NAME)
log_file_object.write('Author: Yang Jiang; Ed O\'Brien. Multi-copy variant: M. DeLyser.\n')
log_file_object.write('Start at '+time.asctime(time.localtime(time.time()))+'\n')
log_file_object.write('Number of windows: '+str(nwin)+'\n')
log_file_object.write('Temperatures: ')
for i in range(len(temps)):
    log_file_object.write(str(temps[i].value_in_unit(kelvin))+' ')
log_file_object.write('\n')
log_file_object.write('All %d windows run as non-interacting copies in ONE OpenMM Context\n' % nwin)
log_file_object.write('Platform: '+platform.getName()+'  properties: '+str(properties)+'\n')
log_file_object.write('Integrator: '+integrator_scheme+' (per-DOF kT CustomIntegrator)\n')
log_file_object.write('Copy separation: '+str(copy_shift)+' nm\n')
log_file_object.write('Atoms per copy: '+str(n_atoms)+'  total: '+str(nwin*n_atoms)+'\n')
log_file_object.write('Constraints per copy: '+str(n_con)+'\n')
log_file_object.write('Number of exchanges in equilibrium: '+str(nexch_equil)+'\n')
log_file_object.write('Number of steps in equilibrium for each exchange: '+str(nsteps_equil)+'\n')
log_file_object.write('Number of exchanges in production: '+str(nexch_prod)+'\n')
log_file_object.write('Number of steps in production for each exchange: '+str(nsteps_prod)+'\n')
log_file_object.write('Log file name: '+str(log_file)+'\n')
log_file_object.write('Eenergy file name: '+ene_file+'\n')
log_file_object.write('Acceptance file name: '+accp_file+'\n')
log_file_object.write('Charmm psf file: '+str(psf_file)+'\n')
log_file_object.write('Charmm top file: '+str(top)+'\n')
log_file_object.write('Charmm prm file: '+str(param)+'\n')
if restart == 0:
    log_file_object.write('No restart requested\n')
else:
    log_file_object.write('Restart requested from '+str(nsteps_start)+'\n')
for i in range(len(starting_strucs)):
    log_file_object.write('Starting structure for T'+str(i+1)+': '+starting_strucs[i]+'\n')
for i in range(len(starting_vels)):
    log_file_object.write('Starting velocities for T'+str(i+1)+': '+starting_vels[i]+'\n')

# Consistency check: with correct replication the per-copy energies must sum to the
# multi-copy total. A replication bug (a force not restricted to intra-copy pairs)
# would show up here rather than as quietly wrong thermodynamics.
_st = simulation.context.getState(getPositions=True, getEnergy=True)
_pos = _st.getPositions(asNumpy=True).value_in_unit(nanometer)
_total = _st.getPotentialEnergy().value_in_unit(kilocalorie/mole)
_sum = sum(replica_energies(_pos))
log_file_object.write('Replication check: sum of per-copy Ep = %.6f kcal/mol, '
                      'multi-copy total = %.6f kcal/mol, difference = %.3e\n'
                      % (_sum, _total, abs(_sum-_total)))
log_file_object.close()
if abs(_sum - _total) > 1e-3*max(1.0, abs(_total)):
    print('Error: per-copy energies do not sum to the multi-copy total '
          '(%.6f vs %.6f). The copies are interacting -- replication is wrong.'
          % (_sum, _total))
    sys.exit()

if not os.path.exists('logs'):
    os.mkdir('logs')
accp_file_object = open('logs/'+accp_file, 'w')
accp_file_object.close()
for i in range(nwin):
    if not os.path.exists('aa'+str(i+1)):
        os.mkdir('aa'+str(i+1))

# one DCD per copy, handles kept open and flushed after every frame
dcd_handles = []
dcd_writers = []
for k in range(nwin):
    _path = 'aa'+str(k+1)+'/mc1.dcd'
    if restart == 1 and os.path.exists(_path) and os.path.getsize(_path) > 0:
        _fh = open(_path, 'r+b')
        dcd_writers.append(DCDFile(_fh, top1, nsteps_prod*timestep, append=True))
    else:
        _fh = open(_path, 'wb')
        dcd_writers.append(DCDFile(_fh, top1, nsteps_prod*timestep))
    dcd_handles.append(_fh)


def pull_state():
    st = simulation.context.getState(getPositions=True, getVelocities=True)
    return (st.getPositions(asNumpy=True).value_in_unit(nanometer),
            st.getVelocities(asNumpy=True).value_in_unit(nanometer/picosecond))


def push_state(pos, vel):
    simulation.context.setPositions(pos*nanometer)
    simulation.context.setVelocities(vel*(nanometer/picosecond))


def apply_exchange(exch_map, pos, vel):
    """Move configurations between copies on an accepted swap.

    Mirrors the original script: swap the coordinates, rescale the velocities by
    sqrt(T_other/T_self), then swap those too, so the configuration arriving in slot j
    carries velocities scaled to temps[j]. The per-DOF kT is NOT touched -- copy j
    stays at temps[j], which is what keeps aa<j+1>/ a fixed-temperature trajectory.
    A block permutation preserves both the distance and the velocity constraints, so
    no re-projection is needed.
    """
    for j in range(nwin-1):
        if exch_map[j] == j+1 and exch_map[j+1] == j:
            a0, a1 = blocks[j]
            b0, b1 = blocks[j+1]
            tj = temps[j].value_in_unit(kelvin)
            tk = temps[j+1].value_in_unit(kelvin)
            ratu = math.sqrt(tk/tj)
            ratd = math.sqrt(tj/tk)
            pj = np.array(pos[a0:a1], copy=True)
            pk = np.array(pos[b0:b1], copy=True)
            pos[a0:a1] = pk
            pos[b0:b1] = pj
            vj = np.array(vel[a0:a1], copy=True)*ratu
            vk = np.array(vel[b0:b1], copy=True)*ratd
            vel[a0:a1] = vk
            vel[b0:b1] = vj
    return pos, vel


def write_cor(path, block):
    psf_pmd.positions = strip_slot_offset(block, copy_shift)*nanometer
    psf_pmd.save(path, format='charmmcrd', overwrite=True)


# ============================== equilibration ==================================
window_track = list(range(nwin))
exch_map = list(range(nwin))
accp = [0]*nwin
nexch = [0]*nwin

for i in range(nexch_equil):
    start_time = time.time()
    simulation.step(nsteps_equil)
    pos, vel = pull_state()
    energy = replica_energies(pos)
    for k in range(nwin):
        write_cor('aa'+str(k+1)+'/1_'+str(i+1)+'_equil.cor', pos[blocks[k][0]:blocks[k][1]])
    exch_map = Temperature_Swap(range(i % 2, nwin), temps, energy)
    pos, vel = apply_exchange(exch_map, pos, vel)
    push_state(pos, vel)
    end_time = time.time()
    cost_time = end_time-start_time
    log_file_object = open(log_file, 'a')
    log_file_object.write('EQUIL %d: %s %s %.2f %s\n' % (
        i+1, time.strftime('%H:%M:%S', time.localtime(start_time)),
        time.strftime('%H:%M:%S', time.localtime(end_time)), cost_time,
        convert_time((nexch_equil-i-1)*cost_time)))
    log_file_object.close()
    new_window_track = [window_track[exch_map[w]] for w in range(nwin)]
    window_track = new_window_track

# ============================== production =====================================
window_track = list(range(nwin))
for i in range(nsteps_start-1, nexch_prod):
    start_time = time.time()
    log_file_object = open(log_file, 'a')
    log_file_object.write('PROD '+str(i+1)+': ')
    for window in range(nwin):
        if window == nwin-1:
            log_file_object.write(str(window_track[window]+1)+'| ')
        else:
            log_file_object.write(str(window_track[window]+1)+'||| ')

    simulation.step(nsteps_prod)
    pos, vel = pull_state()
    energy = replica_energies(pos)

    for k in range(nwin):
        blk = strip_slot_offset(pos[blocks[k][0]:blocks[k][1]], copy_shift)
        dcd_writers[k].writeModel(blk*nanometer)
        dcd_handles[k].flush()
        write_cor('aa'+str(k+1)+'/1_'+str(i+1)+'_prod.cor', pos[blocks[k][0]:blocks[k][1]])
        write_velocity_frame('aa'+str(k+1)+'/1_'+str(i+1)+'_prod_vel.dat',
                             vel[blocks[k][0]:blocks[k][1]])
        f = open('aa'+str(k+1)+'/'+ene_file, 'a')
        f.write('%.6f\n' % (energy[k]))
        f.close()

    exch_map = Temperature_Swap(range(i % 2, nwin), temps, energy)
    pos, vel = apply_exchange(exch_map, pos, vel)
    push_state(pos, vel)

    end_time = time.time()
    cost_time = end_time-start_time
    log_file_object.write('%s %s %.2f %s\n' % (
        time.strftime('%H:%M:%S', time.localtime(start_time)),
        time.strftime('%H:%M:%S', time.localtime(end_time)), cost_time,
        convert_time((nexch_prod-i-1)*cost_time)))
    log_file_object.close()

    new_window_track = [0]*nwin
    for window in range(nwin):
        new_window_track[window] = window_track[exch_map[window]]
        if i > 0:
            prev_cor = 'aa'+str(window+1)+'/1_'+str(i)+'_prod.cor'
            if os.path.exists(prev_cor):
                os.remove(prev_cor)
            prev_vel = 'aa'+str(window+1)+'/1_'+str(i)+'_prod_vel.dat'
            if os.path.exists(prev_vel):
                os.remove(prev_vel)
    window_track = new_window_track

    # acceptance ratio bookkeeping (unchanged from the original script)
    accp_file_object = open('logs/'+accp_file, 'a')
    accp_file_object.write('T-EXCHNG %d: ' % (i+1))
    if i % 2 == 0:
        if nwin % 2 == 0:
            for window in range(nwin): nexch[window] += 1
        else:
            for window in range(nwin-1): nexch[window] += 1
    else:
        if (nwin-1) % 2 == 0:
            for window in range(1, nwin): nexch[window] += 1
        else:
            for window in range(1, nwin-1): nexch[window] += 1
    tag = -1
    for window in range(nwin):
        if window != exch_map[window]:
            accp[window] += 1
            if window != tag:
                accp_file_object.write('%.1f <=> %.1f ' % (
                    temps[window].value_in_unit(kelvin),
                    temps[exch_map[window]].value_in_unit(kelvin)))
            tag = exch_map[window]
    for window in range(nwin):
        if nexch[window] == 0:
            accp_file_object.write('%.2f ' % (0.0))
        else:
            accp_file_object.write('%.2f ' % (accp[window]/nexch[window]))
    accp_file_object.write('\n')
    accp_file_object.close()

for _fh in dcd_handles:
    _fh.close()
