import numpy as np


#todo===========================================
class MESH:
    def __init__(self):
        pass

class ELEMENT:
    def __init__(self, nodes):
        self.node_A = nodes[0]
        self.node_B = nodes[1]
        self.node_C = nodes[2]
#todo===========================================

class NODE:
    def  __init__(self, coords, node_tag, idx):
        self.coords   = coords
        self.node_tag = node_tag        #*GMSH node tag
        self.idx      = idx             #*It is the index in the node list

        #!.update() is used to add more node at once in the set(), .add() if you add one by one
        #!You can use "if node_b in node_a.neighbors", "for n in node_a.neighbors", "len(node_a.neighbors)"
        #!But no indexation: "node_a.neighbors[0]" will be wrong i think, bcs it is a set
        self.neighbors = set()          #*Set of object NODE (set() avoids double values, so it avoids to have 2 times the same neighbor)


        self.adjacent_triangles = []    #TODO List of triangles: 
                                        #??Should we add that ??

        #*Attributs for FMM (init here)
        self.dist = float('inf')
        self.state = 'FAR'      #*'FAR', 'TRIAL', ou 'ALIVE'

    def distance_to_other_node(self, other_node):
        return np.linalg.norm(self.coords - other_node.coords)