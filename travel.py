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
    return maximum


def get_new_individual(dimensions):
    individual = [i for i in range(dimensions)]
    random.shuffle(individual)
    
    return individual


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
     with open(f"results\\SA.csv",newline='',mode='w') as csvFile:
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
                result = simulated_annealing(graph)
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