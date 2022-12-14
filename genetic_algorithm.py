from cProfile import label
from typing import Dict
from unittest import result
from numpy.random import randint
from collections import defaultdict
from Graph import Graph
from Node import Node
from Arch import Arch
from numpy.random import rand
import math
import os
import numpy as np
import timeit
import matplotlib.pyplot as plt
def checkNodePresence(node1, node2,g):
    if g.adj_list.get(node1) and g.adj_list.get(node2):
        return 0
    else:
        return 1
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

def decode(bitstring, cod_len):

    decodedNumber = list()
    number = 0
    #print("bitstring da decodificare: ",bitstring)
    for i in range(0, len(bitstring), cod_len):
        number = 0
        for j in range(i,cod_len+i):
            #print("sto facendo: ", bitstring[j])
            # print("j ",j)
            # print("i ",i)
            # print("totale ",cod_len-1-(j-i) )
            number += bitstring[j]*2**(cod_len-1-(j-i))
        # # print("bitstring: ", bitstring)
        # elem_1 = bitstring[i]
        # elem_2 = bitstring[i+1]
        # # print("elem1: ", elem_1)
        # # print("elem2: ", elem_2)
        # number = elem_1 * 2 + elem_2 
        #print("number: ", number)
        decodedNumber.append(number)
    return decodedNumber
    
#define the objective function as the inverse of the path cost. The higher is the cost the lower is the objective function
def objective(path, g):
    totalCost = 0
    obstacle_flag = 0
    absence_flag = 0
    outOfBound_flag = 0
    fit = 0
    #print("il path in cui verrà calcolata l'obj è: ", path)
    for i in range(0, len(path)-1):
        node1 = path[i]
        node2 = path[i+1]

       
        outOfBound_flag = checkNodePresence(node1, node2, g)

        
        if outOfBound_flag != 1:
            for arch in g.adj_list[node1]:
                attributes = arch.get_arch()
                if attributes[1] == node2:
                    if arch.weight != -1.:
                        if arch.obstacle == 1: 
                            #print("weight arch ", arch.weight)
                            obstacle_flag = 1
                            totalCost += arch.weight
                        else:
                        
                            #print("weight arch ", arch.weight)
                            totalCost += arch.weight
                    else:
                        absence_flag = 1
                        totalCost += arch.weight
        else:
            return -1

    # print("total cost:", totalCost)
    # print("flag trovate (obstacle, absence): ",obstacle_flag, " ", absence_flag)

    if obstacle_flag != 1 and absence_flag != 1 and outOfBound_flag != 1:
        fit = 1.0/totalCost
    
    else:
        fit = -1
    
    #print("fit", fit)
    # if fit > 0:
    #     print ("fit: ",fit)
    
    return fit

#select randomly parents according to theire fitness
def selection(pop, scores, k=6):
    # first random selection
	selection_ix = randint(len(pop))
    
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2, r_cross, cod_len):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        #first crossover segment
        pt1 = randint(cod_len, len(p1) - (cod_len+1))
        #second crossover segment
        pt2 = randint(cod_len, len(p1) - (cod_len+1))
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
           

        #print("figlio 1: ", c1)
        #print("figlio 2: ", c2)
        if c1 == 0 or c2 == 0:
            print("c1 = 0 o c2")
            exit()
        if (len(c1) != len(p1)) or (len(c2) != len(p1)):
            print("wrong dimension!")
            print(len(c1))
            print(len(c2))
            exit()
    return [c1,c2]


def mutation(bitstring, r_mut, bit_len):
    for i in range(bit_len,len(bitstring)-bit_len):
        if rand() < r_mut:
            #flip bit
            bitstring[i] = 1 - bitstring[i]

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, graph, starting_point, ending_point, bit_len):
    #initalize population
    timePerIteration = 0
    pop = [randint(0,2,n_bits).tolist() for _ in range(n_pop)]
    
    cod_len = len(starting_point)
    #print("lunghezza codifica: ", cod_len)

    for x in pop:
        x[0:cod_len] = starting_point
        x[len(x)-cod_len:len(x)] = ending_point
    

    
    best, best_eval = 0, objective(decode(pop[0], cod_len), graph)
    #main loop
    bestGen = 0
    for gen in range(n_iter):
        start = timeit.default_timer()
        #print("inizia la decodifica")
        decoded = [decode(p, cod_len) for p in pop]
        #print("finisce la decodifica")
        scores = [objective(c, graph) for c in decoded]
        #check for best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                bestGen = gen
                best, best_eval = pop[i], scores[i]
                #print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        
        #select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        #create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            #crossover
            for c in crossover(p1, p2, r_cross, cod_len):
                #mutation
                mutation(c, r_mut, bit_len)
                children.append(c)
        
        #print ("i figli sono: ", children)
        pop = children
        end = timeit.default_timer()

        timePerIteration += (end-start)
        #print("nuova popolazione: ", pop)
    
    cost = 1/best_eval
    timePerIteration = timePerIteration / n_iter
    print("best gen: ", bestGen)
    return [best, best_eval, cost, bestGen, timePerIteration]
    


def parsing(directory, length):
    for file in os.listdir(directory):
        g, starting_point, ending_point, bit_len=create_graph(file,length)
        return [g, starting_point, ending_point, bit_len]


#inizializza grafo
def create_graph(file,bit_len):
   

    weights = []
    nodes = []
    list_values = []
    total_node = []
    f = open("dataset/" + file, "r")
    first_line = f.readline().split(" ")
    n_nodes = first_line[0]
    my_dict = {i: np.zeros(int(n_nodes)) for i in range(int(n_nodes))}
    lines = f.read().splitlines()


    for line in lines:
        total_node.append(line.split()[0])
        total_node.append(line.split()[1])
        my_dict[int(line.split()[0])][int(line.split()[1])] = int(line.split()[2])
        my_dict[int(line.split()[1])][int(line.split()[0])] = int(line.split()[2])
    
   
    nodes = list(dict.fromkeys(total_node))
    nodes = [int(x) for x in nodes]
    nodes.sort()
    
    for i in range(len(nodes)):
        my_dict[i] = np.where(my_dict[i] == 0, -1,my_dict[i])
        my_dict[i][i] = 0.0
        weights.append(my_dict[i].tolist())
    

    adj_list = {}

    # print("nodi: ", nodes)
    # print("pesi: ", weights)
    obstacle = 0
    g = Graph()
    #define a fully connected graph with 4 nodes and 2 obstacle
    #nodes = [0, 1,  2, 3]
    #weights = [[0,3,1,5],[3,0,1,4],[1,1,0,2],[5,4,2,0]] #symmetric weight matrix
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

    starting_point, ending_point = code(0,1, bit_len)
    # print(starting_point)
    # print(ending_point)


    for node1 in nodes:
        for node2 in nodes:
            
            weight = weights[node1][node2]
        
        
            if weight == None:
                Weight = weights[node2-1][node1-1]
            
            
            if (node1 == 1 and node2 == 2) or (node1 == 2 and node2 == 1):
                obstacle = 1
            elif (node1 == 0 and node2 == 1) or (node1 == 1 and node2 == 0):
                obstacle = 1
           
            else:
                obstacle = 0
            
            arch_1 = Arch(node1, node2, weight, obstacle)
            adj_list[node1].append(arch_1)
            

    g.adj_list = adj_list
    return [g, starting_point, ending_point,bit_len]

def main(pop, bit_len):
    outcome = 0
    n_obstacles = 2
    g, starting_point, ending_point,bit_len= parsing('dataset/', bit_len)
    iterations = 1000
    # g, starting_point, ending_point = create_graph()
    
    best, score, cost, gen, timePerIteration = genetic_algorithm(objective, int((n_obstacles+2)*bit_len), iterations, pop, 0.9, 1.0/(n_obstacles+2)*bit_len, g, [int(x) for x in starting_point], [int(x) for x in ending_point], bit_len)
    
    if best == 0:
        print("percorso valido non trovato")
        gen = iterations
    else:
        decoded = decode(best, bit_len)
        decoded_path = get_path(decoded, g)
        outcome = 1
        print("solution: ",decoded_path, "score: ", score, "cost:", cost)

    return gen, timePerIteration, outcome

def drawGraphs(totalIterations, totalTimesPerIteration, totalTimes, totalOutcomes, pop, length):
    plt.plot(pop,  totalTimesPerIteration[0], label = "time per iteration")
    plt.legend()
    plt.title("times per iteration")
    plt.xlabel("population size")
    plt.ylabel("time per iteration")
    plt.show()
    for i,l in enumerate(length):
        
        plt.plot(pop, totalIterations[i], label ="iterations")
        plt.legend()
        plt.title("Number of iterations to get the solution bit len = %i" % l )
        plt.ylabel("number iterations")
        plt.xlabel("population size")
        plt.show()

        plt.plot(pop, totalTimes[i], label="total time")
        plt.legend()
        plt.title("Total time spent by the algorithm in order to get the solution len = %i" % l)
        plt.xlabel("population size")
        plt.ylabel("time")

        plt.show()


            


def experiments():
    
    n_pop = [2, 10, 30, 50, 100]
    n_experiments = 20
    bit_len = [2,4,6]
    matrixOutcomes = []
    matrixIterations = []
    matrixTotalTimes = []
    matrixTimePerIteration = []
    for i,length in enumerate(bit_len):
        iterations = []
        totalTimes = []
        timesPerIteration = []
        outcomes = []
        for j,pop in enumerate(n_pop):
            outcome = 0
            iteration = 0
            totalTime = 0
            timePerIteration = 0
            for _ in range(n_experiments):
                results = main(pop, length)
                iteration += results[0] + 1
                timePerIteration += results[1]
                totalTime += iteration * timePerIteration
                outcome += results[2]

        
            percentageOutcome = outcome/n_experiments
            iteration = round(iteration/n_experiments)
            totalTime = totalTime/n_experiments
            timePerIteration = timePerIteration/n_experiments  
            outcomes.append(percentageOutcome)  
            iterations.append(iteration)
            totalTimes.append(totalTime)
            timesPerIteration.append(timePerIteration)
        
        matrixIterations.append(iterations)
        matrixOutcomes.append(outcomes)
        matrixTimePerIteration.append(timesPerIteration)
        matrixTotalTimes.append(totalTimes)

    
    print(len(matrixIterations))
    print(len(matrixIterations[0]))
    
    # for i in range(len(n_pop)):
    #     print("pop: ",n_pop[i])
    #     for j in range(len(bit_len)):
    #         print("bit len: ", bit_len[j])
    #         print("iterations done: ",matrixIterations[i][j])
    #         print("total times elapsed: ", matrixTotalTimes[i][j])
    #         print("times per iteration: ", matrixTimePerIteration[i][j])
    #         print("percentage of outcomes: ", matrixOutcomes[i][j])


    drawGraphs(matrixIterations, matrixTimePerIteration, matrixTotalTimes, matrixOutcomes, n_pop, bit_len)

experiments()



