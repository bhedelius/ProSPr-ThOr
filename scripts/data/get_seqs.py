from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue

import os
import string

from Bio.Data import SCOPData

class Domain(Polypeptide):
    def get_sequence(self):
        s = "".join(
            SCOPData.protein_letters_3to1.get(res.get_resname(), "-") for res in self
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


folder = "../../db/cath/dompdb/"
files = os.listdir(folder)

parser = PDBParser(PERMISSIVE=1, QUIET=True)
builder = PPBuilder()

for i,f in enumerate(files):
    structure = parser.get_structure(f,folder+f)
    pp = builder.build_peptides(structure)
    domains = []
    domain = Domain(pp[0])
    for p in pp[1:]:
        if not domain.extend(p):
            if len(domain)>16:
                domains.append(domain)
            domain = Domain(p)
    if len(domain)>16:
        domains.append(domain)
    subdir = '../../data/seq/'+f[:2]+'/'
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    for j,domain in enumerate(domains):
        name = f+string.ascii_lowercase[j]
        with open(subdir+name,'w') as g:
            g.write(">{}\n{}".format(name,domain.get_sequence()))
    if i%100==0:
        print(i)
