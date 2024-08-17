import os
import random

# Function to generate input files with random float values
def generate_input_files(folder_path, folder_index):
    input_values = []
    for i in range(2):
        with open(os.path.join(folder_path, f'input{i}.raw'), 'w') as file:
            N = folder_index ** 7 + 1
            input_values.append([])
            input_values[i].append(N)
            file.write(f"{N}\n")
            for _ in range(N):
                value = round(random.uniform(0, 100), 2)
                input_values[i].append(value)
                file.write(f"{value:.2f} ")
    return input_values

# Function to generate output file with sum of float values from input0 and input1
def generate_output_file(folder_path, input_values):
    with open(os.path.join(folder_path, 'output.raw'), 'w') as file:
        N = input_values[0][0]
        file.write(f"# ({N}, 1)\n")
        for j in range(1, N+1):
            sum_value = input_values[0][j] + input_values[1][j]
            file.write(f"{sum_value:.2f} ")

# Create folders and generate files
for i in range(0, 10):
    print(f"Generating dataset {i}")
    folder_path = f"{i}"
    os.makedirs(folder_path, exist_ok=True)
    input_values = generate_input_files(folder_path, i)
    generate_output_file(folder_path, input_values)
