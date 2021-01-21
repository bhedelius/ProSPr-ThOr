from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

import os

folder = "dompdb/"
files = os.listdir("dompdb")

parser = PDBParser(PERMISSIVE=1, QUIET=True)
builder = PPBuilder()

total = 0

gaps = dict()

for f in files:
    structure = parser.get_structure(f,folder+f)
    pp = builder.build_peptides(structure)
    print(f,len(pp))
    for i in range(1,len(pp)):
        gap = pp[i][0].id[1] - pp[i-1][-1].id[1] - 1
        if not gap in gaps.keys():
            gaps[gap] = 0
        gaps[gap] = gaps[gap] + 1

with open("gaps_count.txt",'w') as f:
    for key, value in gaps.items():
        f.write("{}	{}\n".format(key,value))
