import os
import random
import math
from time import perf_counter
import sys
import bisect
import csv

test_directory = 'ALL_atsp'

def get_minimum(individual,matrix):
    result = 0
    for i in range(len(matrix)-1):
        cost = matrix[individual[i]-1][individual[i+1]-1]
        result += cost        
    
    result += matrix[individual[-1]-1][individual[0]-1]
    
    return result


def max_value(matrix):
    n = len(matrix)
    maximum = 0 
    for i in range(n):
        for j in range(n):
           if i != j and matrix[i][j]>maximum:
               maximum = matrix[i][j]
    return maximum * n


def fitness(individual,maximum,matrix):
    fitness = (maximum/get_minimum(individual,matrix))**2
    return fitness


def get_new_individual(dimensions):
    individual = [i for i in range(dimensions)]
    random.shuffle(individual)
    
    return individual


def get_population(size,dimensions):
    population = [get_new_individual(dimensions) for _ in range(size)]
    return population


def evaluatePop(population,maximum,matrix):
    eval = [fitness(individual,maximum,matrix) for individual in population]
    return eval


def criteriaSort(listToBeSorted,criteria):
    population = listToBeSorted[:]
    zipped_list = zip(criteria,population)
    zipped_list = sorted(zipped_list)
    
    criteria, population = zip(*zipped_list)
    population = list(population)
    criteria = list(criteria)
    
    return population[::-1]


def inverse_individual(individual):
    n = len(individual)
    left = random.randrange(0,n-1)
    right = random.randrange(left+1,n)
    
    v_copy = individual[:left]
    v_copy.extend(individual[left:right][::-1])
    v_copy.extend(individual[right:])
    
    return v_copy 


def insert_individual(individual):
    n = len(individual)
    left = random.randrange(0,n-1)
    right = random.randrange(left+1,n)
    
    v_copy = individual[:]
    item = v_copy.pop(right)
    v_copy.insert(left,item)
    
    return v_copy


def swap_individual(individual):
    n = len(individual)
    left = random.randrange(0,n-1)
    right = random.randrange(left+1,n)
    
    v_copy = individual[:]
    v_copy[left],v_copy[right] = v_copy[right],v_copy[left]
    
    return v_copy


def tournamentSelect(population,fitnessValues,POP_SIZE,max_value,matrix):
    sampleSize = 15
    selectedSize = 5
    newPopulation = []
    
    for i in range((POP_SIZE)//selectedSize):
        sample = random.sample(population,sampleSize)
        sampleFitness = evaluatePop(sample,max_value,matrix)
        sample = criteriaSort(sample,sampleFitness)
        
        for element in sample[:selectedSize]:
            newPopulation.append(element)
        # newPopulation.append(random.choice(sample[selectedSize:]))
    
    return newPopulation


def populationSelect(population,fitnessValues,POP_SIZE,max_value):
    totalFitness = sum(fitnessValues,max_value)
    individualP = [value/totalFitness for value in fitnessValues]   
    
    accumulatedP = [0]*(len(population)+1)
    for i in range(len(population)):
        accumulatedP[i+1] = accumulatedP[i] + individualP[i]

    newPopulation = []
    for i in range(POP_SIZE):
        r = 1 - random.random()
        index = bisect.bisect_right(accumulatedP,r) - 1
        newPopulation.append(population[index])
    
    return newPopulation


def constant_mutation(population,mutationP):
    mutation_nr = len(population)//2
    newPopulation = []
    
    for i in range(len(population)):
        newChild = population[i][:]
        for j in range(mutation_nr):
            r = 1-random.random()
            first = random.randrange(0,len(newChild))
            second = random.randrange(0,len(newChild))
            if r < mutationP:
                newChild[first],newChild[second] = newChild[second],newChild[first]
        
        newPopulation.append(newChild)
        newPopulation.append(population[i])
    
    return newPopulation


def deterministic_mutation(population,mutationP):
    newPopulation = []
    
    for i in range(len(population)):
        newChild = population[i][:]
        for j in range(len(population[i])):
            r = 1-random.random()
            first = random.randrange(0,len(newChild))
            second = random.randrange(0,len(newChild))
            if r < mutationP:
                newChild[first],newChild[second] = newChild[second],newChild[first]
        
        newPopulation.append(newChild)
        newPopulation.append(population[i])
    
    return newPopulation


def crossover(population,fitnessValues,POP_SIZE,max_value):
    crossoverP = 0.20
    
    newP = [0]*len(population)
    newPopulation = population[:]
    for i in range(len(population)):
        newP[i] = random.random()
    
    zipped_list = zip(newP,newPopulation)
    zipped_list = sorted(zipped_list)
    
    newP, newPopulation = zip(*zipped_list)
    newPopulation = list(newPopulation)
    newP = list(newP)
    
    right = bisect.bisect_left(newP,crossoverP)
    if right % 2 == 0:
        chance = random.getrandbits(1)
        if chance == 1:
            right += 1
        else:
            right -= 1
    
    parents = [newPopulation[i] for i in range(right+1)]
    # totalFitness = sum(fitnessValues,max_value)
    # individualP = [value/totalFitness for value in fitnessValues]   
    
    # accumulatedP = [0]*(len(population)+1)
    # for i in range(len(population)):
    #     accumulatedP[i+1] = accumulatedP[i] + individualP[i]
    # parents = []
    # for i in range(POP_SIZE//2):
    #     r = 1 - random.random()
    #     index = bisect.bisect_right(accumulatedP,r) - 2
    #     parents.append(population[index])
        
    random.shuffle(parents)
    
    index = 0
    while index < len(parents):
        leftLocus = random.randrange(1,len(parents[0])-1)
        rightLocus = random.randrange(2,len(parents[0]))
        while rightLocus <= leftLocus:
            rightLocus = random.randrange(2,len(parents[0]))
        
        child1 = child2 = [None] * len(parents[0])
        for i in range(leftLocus,rightLocus+1):
            child1[i] = parents[index+1][i]
            child2[i] = parents[index][i]
        
        order1 = parents[index][rightLocus+1:] + parents[index][:rightLocus+1]
        order2 = parents[index+1][rightLocus+1:] + parents[index+1][:rightLocus+1]

        for i in range(leftLocus,rightLocus+1):
            order1.remove(child2[i])
            order2.remove(child1[i])

        for i in range(rightLocus+1,len(parents[0])):
            child1[i] = order2[i-rightLocus-1]
            child2[i] = order1[i-rightLocus-1]
        
        for i in range(leftLocus):
            child1[i] = order2[i+len(parents[0])-rightLocus-1]
            child2[i] = order1[i+len(parents[0])-rightLocus-1] 
        
        population.append(child1)
        population.append(child2)
        
        index+=2

    return population    


def genetic(matrix):
    genT = 0
    POP_SIZE = 100
    maximum_value = max_value(matrix)
    
    population = get_population(POP_SIZE,len(matrix))
    fitnessValues = evaluatePop(population,maximum_value,matrix)
    population = criteriaSort(population,fitnessValues)
    
    mutationP = 0.03
    
    while genT < 1000:
        genT += 1
        # deterministM = max((10 + ((len(matrix)*5)/(1000-1))*genT)**(-1),mutationP)
        
        #Elitism : Save the best 5 individuals
        saved = population[:5]
        
        #Selection
        population = tournamentSelect(population[len(saved):],fitnessValues[len(saved):],
                                      POP_SIZE-len(saved),maximum_value,matrix)
        
        # population = populationSelect(population,fitnessValues,POP_SIZE,maximum_value)
        
        #Mutation
        population = constant_mutation(population,mutationP)
        # population = deterministic_mutation(population,deterministM)
        
        #Crossover
        population[:0] = saved
        fitnessValues = evaluatePop(population,maximum_value,matrix)
        population = crossover(population,fitnessValues,POP_SIZE,maximum_value)
        
        #Evaluate
        fitnessValues = evaluatePop(population,maximum_value,matrix)
        
        #Sort population by fitness. (Optional)
        population = criteriaSort(population,fitnessValues)
        

    minim = maximum_value
    
    for individual in population:
        value = get_minimum(individual,matrix)  
        if value < minim:
            minim = value  
            
    return minim

def simulated_annealing(matrix):
    T = 105
    rate = 0.985
    iterations = 10000
    
    v_current = get_new_individual(len(matrix))
    current_best = get_minimum(v_current,matrix)
    
    while T> 10**(-8) and iterations>0:
        local = False
        
        improve = False
        tries = 110
        
        while not improve:
            # Use 3 ways to determine a better neighbour : Inverse, Insert and Swap
            v_inverse = inverse_individual(v_current)
            v_insert = insert_individual(v_current)        
            v_swap = swap_individual(v_current)
            
            inverse_value = get_minimum(v_inverse,matrix)
            insert_value = get_minimum(v_insert,matrix)
            swap_value = get_minimum(v_swap,matrix)
            
            new_inds = [v_inverse,v_insert,v_swap]
            new_values = [inverse_value,insert_value,swap_value]        
            
            bestOf = min(new_values)
            
            if bestOf < current_best:
                v_current = new_inds[new_values.index(bestOf)][:]
                current_best = bestOf
            else:
                P = math.exp((-abs(bestOf-current_best))/T)
                if random.random() < P:
                    v_current = new_inds[new_values.index(bestOf)][:]
                    current_best = bestOf
                else:
                    tries -= 1
            
            if tries == 0:
                improve = True
    
        T *= rate
        iterations -= 1
        
    return current_best


def parse_file(filename):
    # Open file and parse its contents.
    matrix = []
    with open(f"{test_directory}\\{filename}") as test_file:
        header = [test_file.readline() for _ in range(7)]
        
        name = header[0][5:].strip()
        dimensions = int(header[3][10:].strip())
        
        numbers = test_file.read().split()
        numbers.pop()
        numbers = [int(number) for number in numbers]
        
        for i in range(dimensions):
            matrix.append(numbers[i*dimensions:(i+1)*dimensions])
    
    return matrix
       

def main():
    # Iterate over all test files.
     with open(f"results\\GA-ST[15,5]-E-S-m3.csv",newline='',mode='w') as csvFile:
        fieldNames = ['File','Best','Worst','Mean','SD','Time']
        writer = csv.DictWriter(csvFile,fieldnames=fieldNames)
        writer.writeheader()
        for filename in os.listdir(test_directory):
            graph = parse_file(filename)
            
            print(f"The processed file is : {filename}")
            best = worst = None
            avg = 0
            time = 0
            sol = []
            
            for test in range(30):
                startTime = perf_counter()
                result = genetic(graph)
                endTime = perf_counter()
                
                print(f"{test}. {round(endTime-startTime,5)} seconds : {round(result,5)}")
                
                if best is None or result < best:
                    best = result
                if worst is None or result > worst:
                    worst = result
                
                time += (endTime - startTime)
                avg += result
                sol.append(result)
            
            print(f"Best: {best} --- Worst: {worst}")
            avg/=30
            stdev = 0
            
            for index in sol:
                stdev += (index-avg)**2
                
            stdev /= 30
            print(f"Mean: {avg} --- StDev: {math.sqrt(stdev)}")
            time/=30
            print(f"Mean Time: {time}")
            
            writer.writerow({'File':filename,
                                'Best':round(best,5),'Worst':round(worst,5),
                                'Mean':round(avg,5),'SD':round(math.sqrt(stdev),5),'Time':round(time,5)})
            
            
if __name__  == "__main__":
    main()