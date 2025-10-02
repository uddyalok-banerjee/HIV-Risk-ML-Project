# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:37:48 2023

@author: ubanerje
"""
import random

def generate_random_numbers(num_of_numbers):
    return [random.randint(1, 129684) for _ in range(num_of_numbers)]

# Generate 100 random numbers:
random_numbers = generate_random_numbers(100)

df = pd.DataFrame({"Random Numbers": random_numbers})

# Save the DataFrame to an Excel file
file_path = "C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/random_numbers.xlsx"
df.to_excel(file_path, index=False)

#above cutoff

def generate_random_numbers(num_of_numbers):
    return [random.randint(1, 26944) for _ in range(num_of_numbers)]

# Generate 100 random numbers:
random_numbers_above = generate_random_numbers(100)

df = pd.DataFrame({"Random Numbers": random_numbers_above})

# Save the DataFrame to an Excel file
file_path = "C:/Users/ubanerje/OneDrive - rush.edu/Desktop/HIV ML/random_numbers_above.xlsx"
df.to_excel(file_path, index=False)



import random

def generate_random_sets(num_sets, total_numbers):
    numbers_list = list(range(1, total_numbers + 1))
    random_sets = []

    for _ in range(num_sets - 1):
        random_set = random.sample(numbers_list, total_numbers // num_sets)
        random_sets.append(random_set)
        numbers_list = list(set(numbers_list) - set(random_set))

    # Add the remaining numbers to the last set
    random_sets.append(numbers_list)

    return random_sets

# Generate 3 random sets of numbers from 1 to 200:
num_sets = 3
total_numbers = 200
random_sets = generate_random_sets(num_sets, total_numbers)

# Print the sets
for i, random_set in enumerate(random_sets, 1):
    print(f"Set {i}: {random_set}")

    
