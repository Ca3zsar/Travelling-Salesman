import os

test_directory = 'ALL_atsp'

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
        
        
if __name__  == "__main__":
    main()