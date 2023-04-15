import random
import decimal
import copy
import matplotlib.pyplot as plt
from math import cos
from math import pi
from math import exp
from math import sqrt

#Epoches 
iteration = 500
#Number of genes
P = 50
#Number of chromosomes in gene
N = 10

bestIndividual = []
avarageInd = []

#rosenbrock_function -100 <= x <= 100
#ackley_function -32 <= x <= 32
upperbound = 100
lowerbound = -100

#Class for each individual 
class individual:
    def __init__(self):
        self.gene = [0]*N
        self.fitness = 0

#Works out the avarage individual from a generation
def work_out_avarage(offspring):
    result = 0
    for x in range(P):
        result += offspring[x].fitness
    avarageInd.append(result / len(offspring))

#Gets the best individual from population and the worst from offspring and swaps them
def get_best_ind(offspring, population, previous_best):
    bestInd = offspring[0]
    for i in range(P):  
        if(offspring[i].fitness < bestInd.fitness):
            #gets best individual from generation
            bestInd = offspring[i]

    #get best individual from population
    pop_index = 0
    best_pop = population[0]
    for i in range(P):
        if(best_pop.fitness > population[i].fitness):
            best_pop = population[i]
            pop_index = i
            
    #get worst individual from offspring
    offspring_index = 0
    worst_offspring = offspring[0]
    for i in range(P):
        if(worst_offspring.fitness < offspring[i].fitness):
            worst_offspring = offspring[i]
            offspring_index = i

    #replace best from pop with worst in offspring
    offspring[offspring_index] = population[pop_index]

    #compare the previous individual with the current one
    if(previous_best.fitness > bestInd.fitness):
        previous_best = bestInd

        bestIndividual.append(bestInd)
    else:
        bestIndividual.append(previous_best)

    return previous_best, offspring

#Gets the total population fitness
def get_total_fitness(population):
    total = 0
    for x in range(len(population)):
        if(population[x].fitness == 0):
            pass
        else:
            total += population[x].fitness
    return total

######################
# Rastrigin function #
######################
def rastrigin_function(ind):
    fitness = 0
    # n = length
    size = len(ind.gene)
    #Adds the fitness of a genes fitness
    for x in range(size):
        value = ind.gene[x]
        fit = (value ** 2 - 10 * cos(2 * pi * value))
        fitness = fit + fitness
 
    result = (10 * size) + fitness
    return result

#####################
#Rosenbrock function#
#####################
def rosenbrock_function(ind):
    fitness = 0
    #n = length of individual gene
    
    for i in range(N - 1):
        x = ind.gene[i]
        x2 = ind.gene[i + 1]

        p1 = 100 * (x2 - (x ** 2)) ** 2
        p2 = (1 - x) ** 2

        fitness += (p1 + p2)

    return fitness

#################
#Ackley function#
#################
def ackley_function(ind):
    fitness = 0
    p1 = 0
    p2 = 0
    n = len(ind.gene)
    #the sum of both sigma formulas
    for i in range(n):
        x = ind.gene[i]
        p1 += x * x
        p2 += cos(2 * pi * x)

    fitness = -20 * exp(-0.2 * sqrt(p1 / n)) - exp(p2 / n)

    return fitness

def get_fitness(population):
    for x in range(0, P):
        population[x].fitness = rosenbrock_function(population[x])
        # population[x].fitness = ackley_function(population[x])
    return population
    

#will generate a random population of individuals
def init_pop():
    population = []
    # ind = individual()
    #looping over the population and append random values
    for x in range(0, P):
        tempgene = []
        #giving each individual a random number
        for y in range(0, N):
            tempgene.append(random.uniform(lowerbound,upperbound))

        newind = individual()

        #copy random genes to a new individual
        newind.gene = tempgene.copy()
        population.append(newind)

    
    population = get_fitness(population)    
    #Returns the random population of floats
    return population
   
#Tournament selection 
def tournament_selection(population):
    #Define offspring for better solution
    offspring = []
    #Parent Selection
    for x in range(0, P):
        #gets random parent from the population
        p1 = random.randint(0, P-1)
        #sets it to a offspring
        off1 = population[p1]

        #Gets random parent for poulation
        p2 = random.randint(0, P-1)
        #sets it to a offspring
        off2 = population[p2]

        #compares the parents fitness to the other 
        if(off1.fitness > off2.fitness):
            offspring.append(off2)
        else:
            offspring.append(off1)
        
    return offspring

#Roulette wheel selection for minimisation function
def roulette_wheel_selection(population):
    offspring = []
    
    max_total_fitness = get_total_fitness(population)
    
    for x in range(P):
        selection_point = random.uniform(max_total_fitness,0)
    
        running_total = 0
        j = 0
        while(running_total >= selection_point):
            if(population[j].fitness == 0):
                pass
            else:
                running_total += population[j].fitness
                j += 1

        offspring.append(population[j - 1])

    return offspring

#Multi point crossover
#Will select a start and end point and then swap everything inbetween
def multi_point_crossover(offspring):
    #set gap size
    gap = 3

    off1 = individual()
    off2 = individual()
    temp = individual()

    for i in range(0,P - 1):
        off1 = copy.deepcopy(offspring[i])
        off2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])

        start_point = random.randint(1, N)
        end_point = start_point + gap

        if(end_point > N):
            end_point -= gap

        for x in range(start_point, end_point ):
            #grap value @ x from off1 and swap with value @ x from off2
            off1.gene[x] = off2.gene[x]
            off2.gene[x] = temp.gene[x]
        
        offspring[i] = copy.deepcopy(off1)
        offspring[i + 1] = copy.deepcopy(off2)
            
    return offspring

def whole_arithmetic_crossover(offspring):
    off1 = individual()
    off2 = individual()

    for i in range(0, P-1):
        off1 = copy.deepcopy(offspring[i])
        off2 = copy.deepcopy(offspring[i + 1])

        for x in range(0, N - 1):
            result = (off1.gene[x] + off2.gene[x]) / 2
            off1.gene[x] = result
            off2.gene[x] = result 

        offspring[i] = copy.deepcopy(off1)
        offspring[i + 1] = copy.deepcopy(off2)
        
    return offspring

#single point crossover 
def crossover(offspring):
    ''' 
    Single point cross over will pick a random point in the gene pool and swap everyuthing from one
    side to the other
    '''
    off1 = individual()
    off2 = individual()
    temp = individual()

    for i in range(0,P,2):
        #Get 2 different offspring to crossover with
        off1 = copy.deepcopy(offspring[i])
        off2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])

        #random selection point in the genes
        crosspoint = random.randint(1,N)
        for j in range(crosspoint, N):
            off1.gene[j] = off2.gene[j]
            off2.gene[j] = temp.gene[j]
        
        offspring[i] = copy.deepcopy(off1)
        offspring[i + 1] = copy.deepcopy(off2)

    return offspring

def gaussian_mutation(offspring, mutation_rate ,diviation):

    for i in range(P):
        for j in range(N):
            rand_num = random.random()
            if(rand_num < mutation_rate):
                alter = random.gauss(0,diviation)
                offspring[i].gene[j] = offspring[i].gene[j] * alter
                if(offspring[i].gene[j] > upperbound):
                    offspring[i].gene[j] = upperbound
                elif(offspring[i].gene[j] < lowerbound):
                    offspring[i].gene[j] = upperbound
    return offspring

#Mutation for reels 
def mutation(offspring, mutation_rate, MUTSTEP):
    #Loops over each individual in the population
    for i in range(P):
        #loops over each gene in an individual
        for j in range(N):
            #gets random float value
            rand_num = random.random()
            #defining the chance that a gene will mutate
            if(rand_num < mutation_rate):    
                #gets a gene to alter between 0 and the mutation step
                alter = random.uniform(0,MUTSTEP)
                flip = random.randint(0, 1)
                if (flip == 0):
                    #mutate the bit by adding the alter to the gene
                    offspring[i].gene[j]=offspring[i].gene[j] + alter
                    if(offspring[i].gene[j] > upperbound):
                        offspring[i].gene[j] = upperbound
                else:
                    offspring[i].gene[j] = offspring[i].gene[j] - alter

                    if(offspring[i].gene[j] < lowerbound):
                        offspring[i].gene[j] = lowerbound

    return offspring

#loop through 2d array for how many elements are in each array
def work_out_graph_avarage(avarage):
    temp = []
    for count in range(iteration):
        result = 0
        for x in avarage:
            for y_index, y in enumerate(x):
                if(y_index == count):
                    result += y
        temp.append(result / len(avarage))

    return temp

def put_bestind():
    return [bestIndividual[x].fitness for x in range(len(bestIndividual))]

def get_avarage_best(avarage5Best):
    result = 0
    for x in avarage5Best:
        result += x[-1]
    return result / len(avarage5Best)
    


def graph(avarage5Best, mut_rate_array):
    
    avarageBest = get_avarage_best(avarage5Best)
    avarage = work_out_graph_avarage(avarage5Best)


    plt.title("Each iteration with Mutation Rate")
    plt.text(13,8,'Best Individual'+str(avarageBest))
    for index,x in enumerate(avarage5Best):
        
        plt.plot(x)

    plt.legend()
    

    figure, axis = plt.subplots(1,2)
    
    axis[0].text(13,8,'Best Individual'+str(avarageBest))
    axis[0].set_title("Each individual iteration")
    axis[0].set_xlabel("Itteration")
    axis[0].set_ylabel("Performance")

    for x in avarage5Best:
        axis[0].plot(x)                          

    axis[1].text(13,8,'Best Individual'+str(avarageBest))
    axis[1].set_title("10 Runs avaraged out")
    axis[1].set_xlabel("Itteration")
    axis[1].set_ylabel("Performance")
    axis[1].plot(avarage)

    for index,x in enumerate(avarage5Best):
        print("{} : {}".format(mut_rate_array[index], x[-1]))

    print("Best individual = ", avarageBest)


    plt.show()


def main():
    avarage5Best = []
    
    mutation_rate_array = [0.2, 0.4, 0.6, 0.8, 1]

    #gets avarage of each mutation rate
    for i in range(5):
        mutation_rate = mutation_rate_array[i]
        # mutation_rate = 1
        MUTSTEP = 5
        diviation = 0.2

        population = init_pop()
        previous_best = individual()
        previous_best = population[0] 

        for epoch in range(iteration):
            # offspring = tournament_selection(population)
            offspring = roulette_wheel_selection(population)
            
            # offspring = crossover(offspring) 
            offspring = whole_arithmetic_crossover(offspring)
            # offspring = multi_point_crossover(offspring)

            offspring = mutation(offspring, mutation_rate, MUTSTEP)
            # offspring = gaussian_mutation(offspring, mutation_rate ,diviation)
            
            offspring = get_fitness(offspring)

            previous_best,offspring = get_best_ind(offspring, population, previous_best)
            work_out_avarage(offspring)

            #Copy offspring over to the new population 
            population = copy.deepcopy(offspring) 

        tempBest = put_bestind()
        avarage5Best.append(tempBest)
        avarage5Best = copy.deepcopy(avarage5Best)
        bestIndividual.clear()
        avarageInd.clear()

    graph(avarage5Best, mutation_rate_array)

main()




