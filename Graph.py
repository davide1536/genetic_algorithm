from Arch import Arch
from Node import Node
class Graph:
    def __init__(self) -> None:
        self.n_nodes = 0
        self.n_arches = 0
        self.n_obstacles = 0
        
        self.adj_list = {} #dictionary where key: node(int), value: list of arches attached to the node
        
        
    
   
    def get_node(self, node_id):
        return self.id2Node[node_id]
    

