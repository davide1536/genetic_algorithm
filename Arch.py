class Arch:
    def __init__(self, node1, node2, weight, obstacle) -> None:
        self.node1 = node1
        self.node2 = node2
        self.weight = int(weight)
        self.obstacle = obstacle

    def get_arch(self):
        return [self.node1, self.node2, self.weight, self.obstacle]
    
  