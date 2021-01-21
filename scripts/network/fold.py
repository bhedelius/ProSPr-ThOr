import pyrosetta as pr
from pyrosetta.toolbox import pose_from_rcsb
import numpy as np
import tempfile
import random
import os
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

def set_random_dihedral(pose):
    nres = pose.total_residue()
    for i in range(1, nres):
        phi,psi=random_dihedral()
        pose.set_phi(i,phi)
        pose.set_psi(i,psi)
        pose.set_omega(i,180)

    return(pose)

def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return(phi, psi)

def remove_clash(scorefxn, mover, pose):
    for _ in range(0, 5):
        if float(scorefxn(pose)) < 10:
            break
        mover.apply(pose)


pr.init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')

#cleanATOM("3kewA01.pdb")

pose = pr.pose_from_pdb("3kewA01.clean.pdb")
seq = pose.sequence()

N = pose.total_residue()
coords = np.empty((N,3))

for i in range(N):
    res = pose.residue(i+1)
    #if (res.name()[:3]=="GLY"):
    coords[i] = res.xyz("CA")
    #else:
        #print(type(res.xyz("CB")))
        #coords[i] = res.xyz("CB")

vec = coords[None, :, :] - coords[:, None, :]
vec = vec / np.linalg.norm(vec, axis=2, keepdims=True)
angles = np.arccos(np.sum(vec[:, :, None, :] * vec[:, None, :, :], axis=3))

#tmpdir = tempfile.TemoraryDirectory('/')




#######################################
# Scoring functions and movers
#######################################
sf = pr.ScoreFunction()
sf.add_weights_from_file('data/scorefxn.wts')

sf1 = pr.ScoreFunction()
sf1.add_weights_from_file('data/scorefxn1.wts')

sf_vdw = pr.ScoreFunction()
sf_vdw.add_weights_from_file('data/scorefxn_vdw.wts')

sf_cart = pr.ScoreFunction()
sf_cart.add_weights_from_file('data/scorefxn_cart.wts')

mmap = pr.MoveMap()
mmap.set_bb(True)
mmap.set_chi(False)
mmap.set_jump(True)

min_mover = MinMover(mmap, sf, 'lbfgs_armijo_nonmonotone', 0.0001, True)
min_mover.max_iter(10000)

min_mover1 = MinMover(mmap, sf1, 'lbfgs_armijo_nonmonotone', 0.0001, True)
min_mover1.max_iter(10000)

min_mover_vdw = MinMover(mmap, sf_vdw, 'lbfgs_armijo_nonmonotone', 0.0001, True)
min_mover_vdw.max_iter(5000)

min_mover_cart = MinMover(mmap, sf_cart, 'lbfgs_armijo_nonmonotone', 0.0001, True)
min_mover_cart.max_iter(10000)
min_mover_cart.cartesian(True)

repeat_mover = pr.RepeatMover(min_mover, 3)

########################################################
# initialize pose
########################################################

pose = pr.pose_from_sequence(seq, 'centroid')

# mutate GLY to ALA
for i, a in enumerate(seq):
    if a == 'G':
        mutator = pr.rosetta.protocols.simple_moves.MutateResidue(i + 1, 'ALA')
        mutator.apply(pose)
        print('mutation: G%dA' % (i + 1))

set_random_dihedral(pose)
remove_clash(sf_vdw, min_mover_vdw, pose)


rst = {'angle' : []}

for i in range(N):
    for j in range(i):
        for k in range(j):
            if random.random()<0.1:
                rst_line = 'Angle %s %d %s %d %s %d HARMONIC %f %f'%('CA',j+1,'CA',i+1,'CA',k+1, angles[i,j,k], 0.1)
                rst['angle'].append([i,j,k,rst_line])

tmp_name = "constraints.txt"

with open(tmp_name, 'w') as f:
    for line in rst['angle']:
        f.write(line[3] + '\n')

constraints = pr.rosetta.protocols.constraint_movers.ConstraintSetMover()
constraints.constraint_file(tmp_name)
constraints.add_constraints(True)
constraints.apply(pose)
os.remove(tmp_name)

repeat_mover.apply(pose)
min_mover_cart.apply(pose)

for i,a in enumerate(seq):
    if a=='G':
        mutator = pr.rosetta.protocols.simple_moves.MutateResidue(i+1, 'GLY')
        mutator.apply(pose)

sf_fa = pr.create_score_function('ref2015')
sf_fa.set_weight(pr.rosetta.core.scoring.angle_constraint, 1)

mmap = pr.MoveMap()
mmap.set_bb(True)
mmap.set_chi(True)
mmap.set_jump(True)

relax = pr.rosetta.protocols.relax.FastRelax()
relax.set_scorefxn(sf_fa)
relax.max_iter(200)
relax.dualspace(True)
relax.set_movemap(mmap)

pose.remove_constraints()
switch = pr.SwitchResidueTypeSetMover("fa_standard")
switch.apply(pose)

relax.apply(pose)

pose.dump_pdb("out.pdb")