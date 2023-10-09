import numpy as np
import random
import math
import itertools
import time


def calculate_total_distance(route, distances):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distances[route[i]][route[i + 1]]
    total_distance += distances[route[-1]][route[0]]  # Return to the starting point
    return total_distance


def simulated_annealing(distances, initial_route, temperature, cooling_rate):
    current_route = initial_route.copy()
    current_distance = calculate_total_distance(current_route, distances)
    best_route = current_route.copy()
    best_distance = current_distance

    while temperature > 1:
        # Generate a random neighboring route by swapping two random locations
        new_route = current_route.copy()
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_distance = calculate_total_distance(new_route, distances)

        # Calculate the acceptance probability
        delta_distance = new_distance - current_distance
        acceptance_prob = math.exp(-delta_distance / temperature)

        # Accept the new route with a certain probability
        if delta_distance < 0 or random.random() < acceptance_prob:
            current_route = new_route
            current_distance = new_distance

            # Update the best route if needed
            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        # Cool down the temperature
        temperature *= cooling_rate

    return best_route, best_distance


def brute_force_tsp(distances):
    n = len(distances)
    min_distance = float("inf")
    max_distance = 0  # Track the longest distance
    best_route = None
    all_routes = itertools.permutations(range(n))

    for route in all_routes:
        total_distance = calculate_total_distance(route, distances)
        if total_distance < min_distance:
            min_distance = total_distance
            best_route = route
        if total_distance > max_distance:
            max_distance = total_distance

    return best_route, min_distance, max_distance


def generate_random_test_case(n):
    # Generate random distances between readers
    distances = np.random.randint(1, 10, size=(n, n))

    # Ensure distances from a reader to itself are zero
    np.fill_diagonal(distances, 0)

    return distances


# how many readers and books
n = 11
distances = generate_random_test_case(n)

# Parameters for simulated annealing
initial_temperature = 1000.0
cooling_rate = 0.95

# Solve the problem using SA
start_time = time.time()
best_route_sa, best_distance_sa = simulated_annealing(distances, list(range(n)), initial_temperature, cooling_rate)
sa_execution_time = time.time() - start_time

# Solve the problem using brute force
start_time = time.time()
best_route_bf, best_distance_bf, max_distance_bf = brute_force_tsp(distances)
bf_execution_time = time.time() - start_time

print("Random Distances:")
print(distances)
print("\nSimulated Annealing Result:")
print("Best Route (SA):", best_route_sa)
print("Best Distance (SA):", best_distance_sa)
print("Execution Time (SA):", sa_execution_time, "seconds")

print("\nBrute Force Result:")
print("Best Route (Brute Force):", best_route_bf)
print("Best Distance (Brute Force):", best_distance_bf)
print("Longest Distance (Brute Force):", max_distance_bf)
print("Execution Time (Brute Force):", bf_execution_time, "seconds")
