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
        cost = matrix[individual[i]][individual[i+1]]
        result += cost        
    
    result += matrix[individual[-1]][individual[0]]
    
    return cost


def max_value(matrix):
    n = len(matrix)
    maximum = 0 
    for i in range(n):
        for j in range(n):
           if i != j and matrix[i][j]>maximum:
               maximum = matrix[i][j]
    return maximum


def get_new_individual(dimensions):
    individual = [range(dimensions)]
    random.shuffle(individual)
    
    return individual


def simulated_annealing(matrix):
    T = 110
    rate = 0.955
    
    v_current = get_new_individual(len(matrix))
    current_best = get_minimum(v_current,matrix)
    


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
    for filename in os.listdir(test_directory):
        graph = parse_file(filename)
        
        print(f"The processed file is : {filename}")
        with open(f"results\\{filename}-SA.csv",newline='',mode='w') as csvFile:
            fieldNames = ['Approach','Best','Worst','Mean','SD','Time']
            writer = csv.DictWriter(csvFile,fieldnames=fieldNames)
            writer.writeheader()
            
            best = worst = None
            avg = 0
            time = 0
            sol = []
            
            for test in range(30):
                startTime = perf_counter()
                result = simulated_annealing(matrix)
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
            
            writer.writerow({'Approach':'SA',
                                'Best':round(best,5),'Worst':round(worst,5),
                                'Mean':round(avg,5),'SD':round(math.sqrt(stdev),5),'Time':round(time,5)})
            
            
if __name__  == "__main__":
    main()