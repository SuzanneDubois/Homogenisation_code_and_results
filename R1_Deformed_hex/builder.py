import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_utils import write_msh2, plot_mesh

if __name__ == "__main__":
    #coordinates of the nodes
    nodes = [
        [0.0, 0.0, 0.0],
        [1, 0.0, 0.0],
        [1, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1],
        [1, 0.0, 1],
        [1, 1.0, 1],
        [0.0, 1.0, 1]
    ]

    for i in range(len(nodes)):
        if nodes[i][2] == 1 :
            nodes[i][0] += 0.5

    #create all possible bonds between nodes
    bonds = []
    for i in range(1, 9):
        for j in range(i + 1, 9):
            bonds.append((i, j, 0))

    #polt the mesh
    plot_mesh(nodes, bonds)

    #write the mesh to a file
    write_msh2("shear.msh", nodes, bonds, name_to_physical_id={})