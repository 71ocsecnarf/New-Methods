import numpy as np
import heapq

#todo===========================================
class ELEMENT:
    def __init__(self, nodes):
        #*               A         B         C
        self.nodes  = [nodes[0], nodes[1], nodes[2]] 
        self.edges = []

        self.IsObtuse = False

class EDGE:
    def __init__(self, node_a, node_b):
        self.nodes = [node_a, node_b]
        self.kappa = 0.0
        self.adjacent_triangles = []
        self.Is_innit_in_target_elem = False
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
                                        #??Should we add that ?? - I have added it, in case we will remove it - Fra

        #*Attributs for FMM (init here)
        self.dist = float('inf')
        self.state = 'FAR'      #*'FAR', 'TRIAL', ou 'ALIVE'
        self.grad = np.zeros(2)

    def distance_to_other_node(self, other_node):
        return np.linalg.norm(self.coords - other_node.coords)
    
    def __lt__(self, other):
        return self.dist < other.dist

class TRIAL_BAND:
    def __init__(self):
        self.heap = []

    def add_node(self, node):
        node.state = 'TRIAL'
        heapq.heappush(self.heap, node)

    def pop_closest(self):
        while self.heap:
            node = heapq.heappop(self.heap)
            if node.state == 'TRIAL':
                return node
        return None
    
    #* Checks if the heap is empty
    def is_empty(self):
        return len(self.heap) == 0
    
