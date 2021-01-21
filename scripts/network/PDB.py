from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue

import numpy as np

from Bio.Data import SCOPData

np.seterr(invalid='raise')

class Domain(Polypeptide):
    def get_sequence(self):
        s = "".join(
            SCOPData.protein_letters_3to1.get(res.get_resname(), '-') for res in self
        )
        return s

    def extend(self, pp):
        last = self[-1].id[1]
        first = pp[0].id[1]
        gap = first - last - 1
        if abs(gap-8)>8:
            return False
        super(Polypeptide, self).extend(gap*[Residue(0,'-',0)])
        super(Polypeptide, self).extend(pp)
        return True

    def coords(self, id='CB'):
        coords = np.empty((len(self),3)) * np.nan
        for i, res in enumerate(self):
            if res.has_id(id):
                coords[i] = res[id].get_coord()
            elif id=='CB' and res.resname=='GLY' and res.has_id('CA'):
                coords[i] = res['CA'].get_coord()
        return np.ma.masked_invalid(coords)

class PDB:
    def __init__(self, filename):
        self.domains = self.parse(filename)

    @staticmethod
    def parse(filename):
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        structure = parser.get_structure(filename, filename)
        ppb = PPBuilder()
        pp = ppb.build_peptides(structure)
        domains = []
        domain = Domain(pp[0])
        for p in pp[1:]:
            if not domain.extend(p):
                if len(domain)>16:
                    domains.append(domain)
                domain = Domain(p)
        if len(domain)>16:
            domains.append(domain)
        return domains

    def __getitem__(self, item):
        return self.domains[item]

class Calc:
    @staticmethod
    def dist(xyz_i, xyz_j):
        return np.linalg.norm(xyz_i[:,None] - xyz_j[None,:], axis=2)

    @staticmethod
    def angle(xyz_i, xyz_j, xyz_k, ij=False, jk=False):
        if ij:
            vec1 = (xyz_i - xyz_j)[None,:]
        else:
            vec1 = xyz_i[:,None] - xyz_j[None,:]
        vec1 /= np.linalg.norm(vec1, axis=-1, keepdims=True)
        vec1 = vec1[:,:,None]
        if jk:
            vec2 = (xyz_k - xyz_j)[:,None]
        else:
            vec2 = xyz_k[None,:] - xyz_j[:,None]
        vec2 /= np.linalg.norm(vec2, axis=-1, keepdims=True)
        vec2 = vec2[None,:,:]
        dot = np.sum(vec1*vec2, axis=-1)
        dot[dot>1.0] = 1.0
        angle = np.arccos(dot).squeeze()
        return angle

    # https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    @staticmethod
    def dihederal(xyz_i, xyz_j, xyz_k, xyz_l, ij=False, jk=False, kl=False):
        if jk:
            return Calc.dihederal_jk(xyz_i, xyz_j, xyz_k, xyz_l, ij=ij, kl=kl)
        if ij:
            b0 = -1.0 * (xyz_j - xyz_i)[None,:]
        else:
            b0 = -1.0 * (xyz_j[None,:] - xyz_i[:,None])
        b0 = b0[:,:,None,None]

        b1 = xyz_k[None,:] - xyz_j[:,None]
        b1 /= np.linalg.norm(b1,axis=-1, keepdims=True)
        b1 = b1[None,:,:,None]

        if kl:
            b2 = (xyz_l - xyz_k)[:,None]
        else:
            b2 = xyz_l[None,:] - xyz_k[:,None]
        b2 = b2[None,None,:,:]

        v = b0 - np.sum(b0*b1, axis=-1, keepdims=True) * b1
        w = b2 - np.sum(b2*b1, axis=-1, keepdims=True) * b1

        x = np.sum(v*w, axis=-1, keepdims=True)
        y = np.sum(np.cross(b1, v)*w, axis=-1, keepdims=True)
        return np.arctan2(y, x).squeeze()

    @staticmethod
    def dihederal_jk(xyz_i, xyz_j, xyz_k, xyz_l, ij=False, kl=False):
        if ij:
            b0 = -1.0 * (xyz_j - xyz_i)[None, :]
        else:
            b0 = -1.0 * (xyz_j[None, :] - xyz_i[:, None])
        b0 = b0[:, :, None]

        b1 = xyz_k - xyz_j
        b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
        b1 = b1[None, :, None]

        if kl:
            b2 = (xyz_l - xyz_k)[:, None]
        else:
            b2 = xyz_l[None, :] - xyz_k[:, None]
        b2 = b2[None, :, :]

        v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
        w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

        x = np.sum(v * w, axis=-1, keepdims=True)
        y = np.sum(np.cross(b1, v) * w, axis=-1, keepdims=True)
        return np.arctan2(y, x).squeeze()

class Bin:
    @staticmethod
    def dist(dists):
        bins = np.empty_like(dists, dtype=np.long)
        bins[:] = 2 * dists - 3
        bins[dists < 2] = 0
        bins[dists > 21] = 39
        bins[np.isnan(dists)] = 0
        if np.ma.is_masked(dists):
            bins[dist.mask] = 0
        return bins

    @staticmethod
    def dihederal(dihederals):
        bins = np.empty_like(dihederals, dtype=np.long)
        bins[:] = (dihederals / np.pi + 1.0) * 17.99 + 1 # 36 + 1 bins
        bins[np.isnan(dihederals)] = 0
        if np.ma.is_masked(dihederals):
            bins[dihederals.mask] = 0
        return bins

    @staticmethod
    def angle(angles):
        bins = np.empty_like(angles, dtype=np.long)
        bins[:] = (angles / np.pi) * 17.99 + 1 # 18 + 1 bins
        bins[np.isnan(angles)] = 0
        if np.ma.is_masked(angles):
            bins[angles.mask] = 0
        return bins

class Label:
    def __init__(self, domain, il,ih,jl,jh,kl,kh):
        c = domain.coords(id='C')
        ca= domain.coords(id='CA')
        cb= domain.coords(id='CB')
        n = domain.coords(id='N')

        self.dist  = Bin.dist(     Calc.dist(cb[il:ih],cb[kl:kh]))
        self.alpha = Bin.angle(    Calc.angle(ca[il:ih],cb[il:ih],cb[kl:kh],ij=True))                        # trRosetta's phi
        self.beta  = Bin.dihederal(Calc.dihederal(n[il:ih],ca[il:ih],cb[il:ih],cb[kl:kh],ij=True,jk=True))   # trRosetta's theta
        self.gamma = Bin.dihederal(Calc.dihederal(ca[il:ih],cb[il:ih],cb[kl:kh],ca[kl:kh],ij=True,kl=True))  # trRosetta's omega
        self.theta = Bin.angle(    Calc.angle(cb[il:ih],cb[jl:jh],cb[kl:kh]))
        self.phi   = Bin.dihederal(Calc.dihederal(c[il:ih],n[jl:jh],ca[jl:jh],c[kl:kh],jk=True))             # Generalized phi
        self.psi   = Bin.dihederal(Calc.dihederal(n[il:ih],ca[jl:jh],c[jl:jh],n[kl:kh],jk=True))             # Generalized psi

#p = PDB('data/2j66A01')[0]
#l = Label(p,0,30,0,30,0,30)
