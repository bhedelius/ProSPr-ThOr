from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

import os

def log(s):
    with open("pp_count.txt","a") as l:
        l.write(s)

folder = "dompdb/"
files = os.listdir("dompdb")

parser = PDBParser(PERMISSIVE=1, QUIET=True)
builder = PPBuilder()

total = 0

for f in files:
    structure = parser.get_structure(f,folder+f)
    pp = builder.build_peptides(structure)
    print(f,len(pp))
    log("{} {}\n".format(f,len(pp)))
    total = total + len(pp)
print("The total is: ",total)
log("The total is: {}".format(total))
