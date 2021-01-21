from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.Residue import Residue

import os

class Domain(Polypeptide):
    def extend(self, pp):
        last = self[-1].id[1]
        first = pp[0].id[1] 
        gap = first - last - 1
        if abs(gap-4)>4:
            return False
        self = self + gap*[Residue(0,0,0)] + pp
        return True

def log(s):
    with open("protein_count.txt","a") as l:
        l.write(s)

folder = "dompdb/"
files = os.listdir("dompdb")

parser = PDBParser(PERMISSIVE=1, QUIET=True)
builder = PPBuilder()

total = 0

for f in files:
    structure = parser.get_structure(f,folder+f)
    pp = builder.build_peptides(structure)
    domains = []
    domain = Domain(pp[0])
    for p in pp[1:]:
        if not domain.extend(p):
            domains.append(domain)
            domain = Domain(p)
    domains.append(domain)
    print(f,len(pp),len(domains))
    log("{} {}\n".format(f,len(domains)))
    total = total + len(domains)

print("The total is: {}".format(total))
log("The total is: {}".format(total))

