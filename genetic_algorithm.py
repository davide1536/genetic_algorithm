from typing import Dict
from numpy.random import randint
from collections import defaultdict
from Graph import Graph
from Node import Node
from Arch import Arch
from numpy.random import rand
import math
import os
import numpy as np
def code(integer1, integer2, dim):
    bin1 = '{0:0b}'.format(integer1).zfill(int(dim))
    
    bin2 = '{0:0b}'.format(integer2).zfill(int(dim))
    return [bin1, bin2]
def get_path(vector, g):
    current_node = 0
    current_node = vector[0]
    print(vector)
    vector = list(dict.fromkeys(vector))
    return vector     

def decode(bitstring):
    decodedNumber = list()
    #print("bitstring da decodificare: ",bitstring)
    for i in range(0, len(bitstring), 2):
        # print("bitstring: ", bitstring)
        elem_1 = bitstring[i]
        elem_2 = bitstring[i+1]
        # print("elem1: ", elem_1)
        # print("elem2: ", elem_2)
        number = elem_1 * 2 + elem_2 
        # print("number: ", number)
        decodedNumber.append(number)
    return decodedNumber
    
#define the objective function as the inverse of the path cost. The higher is the cost the lower is the objective function
def objective(path, g):
    totalCost = 0
    obstacle_flag = 0
    fit = 0
    #print("il path in cui verrà calcolata l'obj è: ", path)
    for i in range(0, len(path)-1):
        node1 = path[i]
        node2 = path[i+1]
        #print("node 1: ", node1)
        #print("node2: ", node2)
        for arch in g.adj_list[node1]:
            attributes = arch.get_arch()
            if attributes[1] == node2:
                if arch.obstacle == 1: 
                    #print("weight arch ", arch.weight)
                    obstacle_flag = 1
                    totalCost += arch.weight
                else:
                   
                    #print("weight arch ", arch.weight)
                    totalCost += arch.weight

    #print("total cost:", totalCost)

    if obstacle_flag != 1:
        fit = 1.0/totalCost
    
    else:
        fit = -1
    
    #print("fit", fit)
    
    return fit

#select randomly parents according to theire fitness
def selection(pop, scores, k=3):
    # first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    
    if rand() < r_cross:
        #first crossover segment
        pt1 = randint(2, len(p1) - 3)
        #second crossover segment
        pt2 = randint(2, len(p1) - 3)
        #switch the "middle part", defined by the 2 segments, of both parents and create the 2 childs c1 and c2
        if pt2 >= pt1:
            middle_p1 = p1[pt1:pt2+1]
            middle_p2 = p2[pt1:pt2+1]
            c1 = p1[0:pt1] + middle_p2 + p1[pt2+1:len(p1)]
            c2 = p2[0:pt1] + middle_p1 + p2[pt2+1:len(p1)]
           
        elif pt2 < pt1:
            middle_p1 = p1[pt2:pt1+1]
            middle_p2 = p2[pt2:pt1+1]
            c1 = p1[0:pt2] + middle_p2 + p1[pt1+1:len(p1)]
            c2 = p2[0:pt2] + middle_p1 + p2[pt1+1:len(p1)]
           

    
        if (len(c1) != 8) or (len(c2) != 8):
            print("wrong dimension!")
            print(len(c1))
            print(len(c2))
            exit()
    return [c1,c2]


def mutation(bitstring, r_mut):
    for i in range(2,len(bitstring)-2):
        if rand() < r_mut:
            #flip bit
            bitstring[i] = 1 - bitstring[i]

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, graph, starting_point, ending_point):
    #initalize population
    pop = [randint(0,2,n_bits).tolist() for _ in range(n_pop)]
   

    for x in pop:
        x[0:2] = starting_point
        x[len(x)-2:len(x)] = ending_point
        
    best, best_eval = 0, objective(decode(pop[0]), graph)
    #main loop
    for gen in range(n_iter):
        decoded = [decode(p) for p in pop]
        scores = [objective(c, graph) for c in decoded]
        #check for best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        
        #select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        #create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            #crossover
            for c in crossover(p1, p2, r_cross):
                #mutation
                mutation(c, r_mut)
                children.append(c)
        
        #print ("i figli sono: ", children)
        pop = children
        #print("nuova popolazione: ", pop)
    
    cost = 1/best_eval
    
    return [best, best_eval, cost]
    


def parsing(directory):
    for file in os.listdir(directory):
        create_graph(file)

#inizializza grafo
def create_graph():
   

    # weights = [[]]
    # nodes = []
    # list_values = []
    # total_node = []
    # f = open("dataset/" + file, "r")
    # first_line = f.readline().split(" ")
    # n_nodes = first_line[0]
    # my_dict = {emp_list: [] for emp_list in range(int(n_nodes))}
    # lines = f.read().splitlines()
    
    # for line in lines:
    #     total_node.append(line.split()[0])
    #     total_node.append(line.split()[1])
    #     my_dict[int(line.split()[0])-1].insert(int(line.split()[1])-1, int(line.split()[2]))
    #     my_dict[int(line.split()[1])-1].insert(int(line.split()[0])-1, int(line.split()[2]))
    
    # print(my_dict)
    # nodes = list(dict.fromkeys(total_node))
    # nodes = [int(x) for x in nodes]
    # nodes.sort()
    # print (nodes)
    # exit()

    adj_list = {}

    
    obstacle = 0
    g = Graph()
    #define a fully connected graph with 4 nodes and 2 obstacle
    nodes = [0, 1,  2, 3]
    weights = [[0,3,1,5],[3,0,1,4],[1,1,0,2],[5,4,2,0]] #symmetric weight matrix
    for node in nodes:
        #ending point of the graph
        if node == 1:
            obj_node = Node(node, 1, 0)
        #starting point of the graph
        elif node == 0:
            obj_node = Node(node, 0, 1)
        else:
            obj_node = Node(node, 0, 0)

    
        adj_list.setdefault(node, [])
    #convert in binary the starting and the ending point of the graph
    bit_len = math.log2(len(nodes))

    starting_point, ending_point = code(0,1, bit_len)


    for node1 in nodes:
        for node2 in nodes:
            
            weight = weights[node1][node2]
        
        
            # if weight == None:
            #     Weight = weights[node2-1][node1-1]
            if (node1 == 0 and node2 == 2) or (node1 == 2 and node2 == 0):
                obstacle = 1
            elif (node1 == 2 and node2 == 1) or (node1 == 1 and node2 == 2):
                obstacle = 1
            else:
                obstacle = 0
            
            arch_1 = Arch(node1, node2, weight, obstacle)
            adj_list[node1].append(arch_1)
            

    g.adj_list = adj_list
    return [g, starting_point, ending_point]
    
n_obstacles = 2
#parsing('dataset/')
g, starting_point, ending_point = create_graph()
best, score, cost = genetic_algorithm(objective, (n_obstacles+2)*2, 50, 100, 0.9, 1.0/(4*2), g, [int(x) for x in starting_point], [int(x) for x in ending_point])
decoded = decode(best)
decoded_path = get_path(decoded, g)
print("solution: ",decoded_path, "score: ", score, "cost:", cost)



