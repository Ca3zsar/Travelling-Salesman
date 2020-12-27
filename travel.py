import os
import random
import math
from time import perf_counter
import sys
import bisect
from operator import attrgetter
import csv

test_directory = 'ALL_atsp'

def get_minimum(individual, matrix):
    result = 0
    for i in range(len(matrix)-1):
        cost = matrix[individual[i]][individual[i+1]]
        result += cost

    result += matrix[individual[-1]][individual[0]]

    return result

def max_value(matrix):
    n = len(matrix)
    maximum = 0
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > maximum:
                maximum = matrix[i][j]
    return maximum * n

def check(population):
    n = len(population[0])
    for ind in population:
        if (n*(n-1)/2) != sum(ind):
            return 0
    return 1

def fitness(individual, matrix):
    fitness = (1/get_minimum(individual, matrix))**2
    return fitness

def get_new_individual(dimensions):
    individual = [i for i in range(dimensions)]
    random.shuffle(individual)

    return individual

# def get_new_individual(dimensions):
    # individual = []
    # for i in range(dimensions):
    #     individual.append(random.randrange(0,dimensions-i))
    
    # return individual

def get_population(size, dimensions):
    population = [get_new_individual(dimensions) for _ in range(size)]
    return population

def encode(individual,dimensions):
    vec = [i for i in range(dimensions)]
    new_ind = individual[:]
    
    for i in range(dimensions):
        pos = 0
        for j in range(len(vec)):
            if individual[i] == vec[j]:
                pos = j
                break
        new_ind[i] = pos
        vec.pop(pos)
    
    return new_ind
            
def decode(individual,dimensions):
    vec = []
    new_ind = individual[:]
    
    for i in range(len(individual)):
        vec.append(i)
        
    for i in range(len(individual)):
        pos = new_ind[i]
        new_ind[i] = vec[pos]
        vec.pop(pos)
    
    return new_ind
        
def population_control(population, matrix, dimensions,POP_SIZE):
    yes = 1

    while yes:
        yes = 0
        for ind in population:
            value = population.count(ind)
            if value > POP_SIZE*0.01:
                to_keep = ind[:]

                population = [el for el in population if el != to_keep]
                population.append(to_keep)

                yes = 1
                break

    population.extend(get_population(POP_SIZE-len(population), dimensions))

    return population

def reset_population(population, POP_SIZE, dimensions):
    to_save = POP_SIZE//10

    new_population = population[:to_save]
    new_population.extend(get_population(POP_SIZE-to_save, dimensions))

    return new_population

def evaluatePop(population, matrix):
    eval = [fitness(individual, matrix) for individual in population]
    return eval

def criteriaSort(listToBeSorted, criteria):
    population = listToBeSorted[:]
    zipped_list = zip(criteria, population)
    zipped_list = sorted(zipped_list)

    criteria, population = zip(*zipped_list)
    population = list(population)
    criteria = list(criteria)

    return population[::-1]

def get_best(number, population, matrix):
    fitnessValues = evaluatePop(population, matrix)

    newPop = population[:]
    newPop = criteriaSort(newPop, fitnessValues)

    return newPop[:number]

def inverse_individual(individual):
    n = len(individual)
    left = random.randrange(0, n-1)
    right = random.randrange(left+1, n)

    v_copy = individual[:left]
    v_copy.extend(individual[left:right][::-1])
    v_copy.extend(individual[right:])

    return v_copy

def insert_individual(individual):
    n = len(individual)
    left = random.randrange(0, n-1)
    right = random.randrange(left+1, n)

    v_copy = individual[:]
    item = v_copy.pop(right)
    v_copy.insert(left, item)

    return v_copy

def swap_individual(individual):
    n = len(individual)
    left = random.randrange(0, n-1)
    right = random.randrange(left+1, n)

    v_copy = individual[:]
    v_copy[left], v_copy[right] = v_copy[right], v_copy[left]

    return v_copy

def scramble_individual(individual):
    n = len(individual)
    left = random.randrange(0,n-1)
    right = random.randrange(left+1,n)
    
    v_copy = individual[:left]
    v_copy2 = individual[left:right]
    random.shuffle(v_copy2)
    v_copy.extend(v_copy2)
    v_copy.extend(individual[right:])
    
    return v_copy

def tournamentSelect(population, fitnessValues, POP_SIZE, matrix):
    sampleSize = 15
    selectedSize = 5
    done = 0
    newPopulation = []

    matingP = 0.2

    for i in range((POP_SIZE)//selectedSize):
        sample = random.sample(population, sampleSize)
        sampleFitness = evaluatePop(sample, matrix)
        sample = criteriaSort(sample, sampleFitness)

        for element in sample[:selectedSize]:
            newPopulation.append(element)

        r = random.random()
        if r < matingP and len(population) > 2*sampleSize:
            for ind in sample:
                population.remove(ind)

        if len(population) < 2*sampleSize:
            break
        
        # newPopulation.append(random.choice(sample[selectedSize:]))

    return newPopulation

def constant_mutation(population, mutationP, matrix):
    newPopulation = []

    for i in range(len(population)):
        newChild = population[i][:]

        r = random.random()

        if r < mutationP:
            v_inverse = inverse_individual(newChild)
            v_insert = insert_individual(newChild)
            v_swap = swap_individual(newChild)
            v_scramble = scramble_individual(newChild)

            inverse_value = get_minimum(v_inverse, matrix)
            insert_value = get_minimum(v_insert, matrix)
            swap_value = get_minimum(v_swap, matrix)
            scramble_values = get_minimum(v_scramble,matrix)

            new_inds = [v_inverse, v_insert, v_swap,v_scramble]
            new_values = [inverse_value, insert_value, swap_value,scramble_values]

            bestOf = min(new_values)

            newChild = new_inds[new_values.index(bestOf)][:]
            current_best = bestOf

            newPopulation.append(newChild)
        else:
            newPopulation.append(population[i])

    return newPopulation

def coded_mutation(population,mutationP):
    new_population = []
    length = len(population[0])
     
    for i in range(len(population)):
        newChild = population[i][:]
        
        for j in range(len(newChild)):
            r = 1-random.random()
            if r < mutationP:
                newChild[j] = random.randrange(0,length-j)
            
        new_population.append(newChild)
        new_population.append(population[i])
    
    return new_population

def pmx(a, b, start, stop):
    child = [None]*len(a)

    # Copy a slice from first parent:
    child[start:stop] = a[start:stop]

    # Map the same slice in parent b to child using indices from parent a:
    for ind, x in enumerate(b[start:stop]):
        ind += start
        if x not in child:
            while child[ind] != None:
                ind = b.index(a[ind])
            child[ind] = x

    # Copy over the rest from parent b
    for ind, x in enumerate(child):
        if x == None:
            child[ind] = b[ind]
    return child

def pmx_pair(a, b):
    half = len(a) // 2
    start = random.randint(0, len(a)-half)
    stop = start + half
    return pmx(a, b, start, stop), pmx(b, a, start, stop)

def cycle_xover(parent1, parent2):
    a = parent1
    b = parent2
    child = [None]*len(a)
    while None in child:
        ind = child.index(None)
        indices = []
        values = []
        while ind not in indices:
            val = a[ind]
            indices.append(ind)
            values.append(val)
            ind = a.index(b[ind])
        for ind, val in zip(indices, values):
            child[ind] = val
        a, b = b, a
    return child

def cycle_xover_pair(a, b):
    return cycle_xover(a, b), cycle_xover(b, a)

def order_xover(a,b):
    n = len(a)
    
    leftLocus = random.randrange(1,n-1)
    rightLocus = random.randrange(2,n)
    
    while rightLocus <= leftLocus:
        rightLocus = random.randrange(2,n)

    child1 = child2 = [None] * n
    for i in range(leftLocus,rightLocus+1):
        child1[i] = b[i]
        child2[i] = a[i]

    order1 = a[rightLocus+1:] + a[:rightLocus+1]
    order2 = b[rightLocus+1:] + b[:rightLocus+1]

    for i in range(leftLocus,rightLocus+1):
        order1.remove(child2[i])
        order2.remove(child1[i])

    for i in range(leftLocus):
        child1[i] = order2[i]
        child2[i] = order1[i]

    for i in range(rightLocus+1,n):
        child1[i] = order2[i-rightLocus-1+leftLocus]
        child2[i] = order1[i-rightLocus-1+leftLocus]
    
    return a,b


def crossover(population,fitnessValues):
    parents = criteriaSort(population,fitnessValues)

    population = []

    index = 1
    while index < len(parents)//2:
        
        child1,child2 = order_xover(parents[0],parents[index])
        
        population.append(child1)
        population.append(child2)

        child1,child2 = pmx_pair(parents[0],parents[index])
        
        population.append(child1)
        population.append(child2)
        
        child1, child2 = cycle_xover_pair(parents[0], parents[index])

        population.append(child1)
        population.append(child2)

        index += 1

    return population

def coded_crossover(population):
    crossoverP = 0.25

    newP = [0]*len(population)
    newPopulation = population[:]
    for i in range(len(population)):
        newP[i] = random.random()

    zipped_list = zip(newP, newPopulation)
    zipped_list = sorted(zipped_list)

    newP, newPopulation = zip(*zipped_list)
    newPopulation = list(newPopulation)

    right = bisect.bisect_left(newP, crossoverP)
    if right % 2 == 0:
        chance = random.getrandbits(1)
        if chance == 1:
            right += 1
        else:
            right -= 1

    parents = [newPopulation[i] for i in range(right+1)]
    random.shuffle(parents)
    
    index = 0
    while index < len(parents):
        locus1 = random.randrange(1,len(parents[index])-1)
        locus2 = random.randrange(locus1,len(parents[index]))
        
        child1 = parents[index][:locus1]
        child1.extend(parents[index+1][locus1:locus2])
        child1.extend(parents[index][locus2:])
        
        child2 = parents[index+1][:locus1]
        child2.extend(parents[index][locus1:locus2])
        child2.extend(parents[index+1][locus2:])
        
        index += 2
        
        population.append(child1)
        population.append(child2)
    
    # print(f"{len(parents)} - {len(population)}")
    return population
    

def genetic(matrix):
    genT = 0
    POP_SIZE = 100
    maximum_value = max_value(matrix)

    population = get_population(POP_SIZE, len(matrix))
    fitnessValues = evaluatePop(population, matrix)
    # population = criteriaSort(population,fitnessValues)
    mutationP = 0.01
    minMutation = 0.02
    maxMutation = 0.25
    same = 0

    to_save = 5

    previous = minim_curent = minim = None

    while genT < 10000:
        genT += 1

        population = population_control(population, matrix, len(matrix),POP_SIZE)

        saved = get_best(to_save, population, matrix)
        
        # Selection
        population = tournamentSelect(population[to_save:], fitnessValues[to_save:],
                                      POP_SIZE-to_save, matrix)
        
        # Crossover
        population[:0] = saved
        fitnessValues = evaluatePop(population, matrix)
        population = crossover(population,fitnessValues)
        
        # Mutation
        # population[:0] = saved
        # population = coded_mutation(population, mutationP)
        population = constant_mutation(population,mutationP,matrix)

        # Evaluate
        population[:0] = saved
        fitnessValues = evaluatePop(population, matrix)

        # Sort population by fitness. (Optional)
        # population = criteriaSort(population,fitnessValues)

        minim = maximum_value
        for i in population:
            value = get_minimum(i, matrix)
            if value < minim:
                minim = value

        if minim_curent and previous:
            if previous == minim:
                same += 1
            else:
                if minim > minim_curent:
                    same = 1
                    to_save = min(to_save+1, 5)
                else:
                    minim_curent = minim
                    mutationP *= 0.98
                    same = 1
                previous = minim
        else:
            same = 1
            previous = minim
            minim_curent = minim

        if same % 5 == 0:
            if to_save > 0:
                to_save -= 1

            mutationP = min(mutationP + 0.001, maxMutation)
            if mutationP == maxMutation:
                mutationP = minMutation
        
        if same % 50 == 0:
            reset_population(population,POP_SIZE,len(matrix))
            to_save = 5
        
        print(f"{genT}. {minim} {mutationP}")

    minim = maximum_value

    for individual in population:
        value = get_minimum(individual, matrix)
        if value < minim:
            minim = value

    return minim

def simulated_annealing(matrix):
    T = 105
    rate = 0.985
    iterations = 10000

    v_current = get_new_individual(len(matrix))
    current_best = get_minimum(v_current, matrix)

    while T > 10**(-8) and iterations > 0:
        local = False

        improve = False
        tries = 110

        while not improve:
            # Use 3 ways to determine a better neighbour : Inverse, Insert and Swap
            v_inverse = inverse_individual(v_current)
            v_insert = insert_individual(v_current)
            v_swap = swap_individual(v_current)

            inverse_value = get_minimum(v_inverse, matrix)
            insert_value = get_minimum(v_insert, matrix)
            swap_value = get_minimum(v_swap, matrix)

            new_inds = [v_inverse, v_insert, v_swap]
            new_values = [inverse_value, insert_value, swap_value]

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
    with open(f"results\\GA-newST[20,5]-E-S-cycle.csv", newline='', mode='w') as csvFile:
        fieldNames = ['File', 'Best', 'Worst', 'Mean', 'SD', 'Time']
        writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
        writer.writeheader()
        for filename in os.listdir(test_directory)[1:]:
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

                print(
                    f"{test}. {round(endTime-startTime,5)} seconds : {round(result,5)}")

                if best is None or result < best:
                    best = result
                if worst is None or result > worst:
                    worst = result

                time += (endTime - startTime)
                avg += result
                sol.append(result)

            print(f"Best: {best} --- Worst: {worst}")
            avg /= 30
            stdev = 0

            for index in sol:
                stdev += (index-avg)**2

            stdev /= 30
            print(f"Mean: {avg} --- StDev: {math.sqrt(stdev)}")
            time /= 30
            print(f"Mean Time: {time}")

            writer.writerow({'File': filename,
                             'Best': round(best, 5), 'Worst': round(worst, 5),
                             'Mean': round(avg, 5), 'SD': round(math.sqrt(stdev), 5), 'Time': round(time, 5)})


if __name__ == "__main__":
    main()
