import itertools
import math
import multiprocessing as mp
import random
import time

log_results = False

##################################################
print("LOG RESULTS", log_results)
input("BEGIN estimate_pi ...\n")
##################################################

if log_results: log = open("log1.csv", "w", encoding = "utf-8")
if log_results: log.write("Processes,Speedup\n")

def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

for i in range(4):
    seeds = list(range(2 ** i))

    t0 = time.perf_counter()
    seq_results = [estimate_pi(s) for s in seeds]
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        par_results = pool.map(estimate_pi, seeds)
    t_par = time.perf_counter() - t0

    print(f"Sequential wall-clock: {t_seq:.3f} s")
    print(f"Parallel wall-clock:   {t_par:.3f} s (Speedup: {t_seq/t_par:.2f}x)")
    print(f"pi estimate (mean of {len(seeds)} runs): {sum(par_results) / len(par_results):.5f}")
    if log_results: log.write(f"{len(seeds)},{t_seq/t_par}\n")
    if i < 3:
        print("")

if log_results: log.close()

##################################################
print("\nCOOL")
input("BEGIN knapsack ...\n")
##################################################

if log_results: log = open("log2.csv", "w", encoding = "utf-8")

def greedy_knapsack(values: list[int], weights: list[int], W: int, greediest: bool):
    ratios = [a / b for a, b in zip(values, weights)]
    ratios_items_values_weights = sorted(zip(ratios, items, values, weights), reverse = greediest)
    items_stored = []
    value_stored = 0
    weight_stored = 0
    for thing in ratios_items_values_weights:
        if weight_stored + thing[3] > W:
            break
        else:
            items_stored.append(thing[1])
            value_stored += thing[2]
            weight_stored += thing[3]
    return items_stored, value_stored, weight_stored

items, values, weights = ["A", "B", "C"], [60, 100, 120], [10, 20, 30]
print("Items:", items)
print("Values:", values)
print("Weights:", weights)
print(f"Greedy approximation: {greedy_knapsack(values, weights, 50, True)}")
print(f"Modified greedy approximation: {greedy_knapsack(values, weights, 50, False)}")
print("")
items, values, weights = ["A", "B", "C"], [60, 100, 120], [30, 20, 10]
print("Items:", items)
print("Values:", values)
print("Weights:", weights)
print(f"Greedy approximation: {greedy_knapsack(values, weights, 50, True)}")
print(f"Modified greedy approximation: {greedy_knapsack(values, weights, 50, False)}")

if log_results: log.close()

##################################################
print("\nCOOL")
input("BEGIN tsp ...\n")
##################################################

if log_results: log = open("log3.csv", "w", encoding = "utf-8")
if log_results: log.write("Swaps (random),Distance from optimum (random),Swaps (nearest neighbor),Distance from optimum (nearest neighbor)\n")

cities = {"A": (0, 0), "B": (0, 3), "C": (2, 5), "D": (4, 3), "E": (4, 0), "F": (0, 5), "G": (1, 5), "H": (5, 5)}

def path_length(start: str, end: str) -> float:
    return math.dist(cities[start], cities[end])

def tour_length(route: list[str]) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def tour_random():
    tour = list(cities)
    random.shuffle(tour)
    return tour

def tour_nearest_neighbor():
    start_node = random.choice(list(cities))

    unvisited = set(list(cities))
    current_node = start_node
    unvisited.remove(current_node)
    tour = [current_node]

    while unvisited:
        next_node = sorted([(path_length(current_node, node), node) for node in list(unvisited)])[0][1]
        current_node = next_node
        unvisited.remove(current_node)
        tour.append(current_node)

    return tour

def tsp_brute_force():
    nodes = list(cities)
    n = len(nodes)

    if n <= 1:
        return nodes, 0.0

    start_node = nodes[0]
    remaining_nodes = nodes[1:]

    best_tour = None
    min_distance = float('inf')

    for perm in itertools.permutations(remaining_nodes):
        current_tour = [start_node] + list(perm)
        current_distance = tour_length(current_tour)

        if current_distance < min_distance:
            min_distance = current_distance
            best_tour = current_tour

    return best_tour

def tsp_2_opt(tour):
    best_tour = list(tour)
    improved = True
    count = 0
    n = len(tour)

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                if j - i == 1 and i == 0 and j == n:
                    continue

                new_tour = two_opt_move(best_tour, i, j)

                if tour_length(new_tour) < tour_length(best_tour):
                    best_tour = new_tour
                    count += 1
                    improved = True
                    break
            if improved:
                break

    return best_tour, count

route = tour_random()
optimal_route = tsp_brute_force()
optimal_length = tour_length(optimal_route)
for i in range(100):
    print(f"Optimal (brute): {optimal_route}  L = {optimal_length:.2f}")
    print("")
    route = tour_random()
    new_route, count = tsp_2_opt(route)
    print("Random tour:")
    print(f"Before: {route}  L = {tour_length(route):.2f}")
    print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
    print(f"Swap count:  {count}")
    if log_results: log.write(f"{count},{tour_length(new_route) - optimal_length},")
    print("")
    route = tour_nearest_neighbor()
    new_route, count = tsp_2_opt(route)
    print("Nearest neighbor tour:")
    print(f"Before: {route}  L = {tour_length(route):.2f}")
    print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
    print(f"Swap count:  {count}")
    if log_results: log.write(f"{count},{tour_length(new_route) - optimal_length}\n")
    if i < 99:
        print("")

if log_results: log.close()

##################################################
print("\nCOOL")
input("BEGIN simulated_annealing ...\n")
##################################################

if log_results: log1 = open("log4-1.csv", "w", encoding = "utf-8")
if log_results: log2 = open("log4-2.csv", "w", encoding = "utf-8")
if log_results: log3 = open("log4-3.csv", "w", encoding = "utf-8")
if log_results: log4 = open("log4-4.csv", "w", encoding = "utf-8")
if log_results: log5 = open("log4-5.csv", "w", encoding = "utf-8")
if log_results: log6 = open("log4-6.csv", "w", encoding = "utf-8")
if log_results: log7 = open("log4-7.csv", "w", encoding = "utf-8")
if log_results: log8 = open("log4-8.csv", "w", encoding = "utf-8")
if log_results: log1.write("Cost,Iterations\n")
if log_results: log2.write("Cost,Iterations\n")
if log_results: log3.write("Cost,Iterations\n")
if log_results: log4.write("Cost,Iterations\n")
if log_results: log5.write("Time,Quality\n")
if log_results: log6.write("Time,Quality\n")
if log_results: log7.write("Time,Quality\n")
if log_results: log8.write("Time,Quality\n")

cities = {"A": (0, 0), "B": (0, 3), "C": (2, 5), "D": (4, 3), "E": (4, 0), "F": (0, 5), "G": (1, 5), "H": (5, 5)}

def tour_length(route: list[str]) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def two_opt_move_random(route: list[str]) -> list[str]:
    i, k = random.sample(range(len(route)), 2)
    if i > k:
        i, k = k, i
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

def tsp_brute_force():
    nodes = list(cities)
    n = len(nodes)

    if n <= 1:
        return nodes, 0.0

    start_node = nodes[0]
    remaining_nodes = nodes[1:]

    best_tour = None
    min_distance = float('inf')

    for perm in itertools.permutations(remaining_nodes):
        current_tour = [start_node] + list(perm)
        current_distance = tour_length(current_tour)

        if current_distance < min_distance:
            min_distance = current_distance
            best_tour = current_tour

    return best_tour

def linear_schedule(alpha: float):
    def linear(x: int | float):
        return max(0, x - alpha)
    return linear

def exponential_schedule(alpha: float):
    def exponential(x: int | float):
        return max(0, x * alpha)
    return exponential

def roll(n: float) -> bool:
    if n == 0:
        return False
    return random.random() < 1 / n

def simulated_annealing(route: list[str], schedule):
    current = list(route)
    t = 0
    T = 500
    graph = []
    graph.append((tour_length(current), t))
    while True:
        t += 1
        T = schedule(T)
        if t > 1000 or T <= 0:
            return current, graph
        new = two_opt_move_random(current)
        delta_E = tour_length(new) - tour_length(current)
        if delta_E <= 0:
            current = list(new)
        else:
            if roll(math.exp(-1 * delta_E / T)):
                current = list(new)
        graph.append((tour_length(current), t))

def random_route(seed: int):
    route = list(cities)
    random.shuffle(route)
    return route

route = random_route(0)

route = tsp_brute_force()
optimal_length = tour_length(route)
print(f"Optimal (brute): {route}  L = {optimal_length:.2f}")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, linear_schedule(0.5))
t1 = time.perf_counter() - t0
print("Linear schedule 1:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log1.write(f"{pair[0]},{pair[1]}\n")
if log_results: log5.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, linear_schedule(0.9))
t1 = time.perf_counter() - t0
print("Linear schedule 2:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log2.write(f"{pair[0]},{pair[1]}\n")
if log_results: log6.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, exponential_schedule(0.5))
t1 = time.perf_counter() - t0
print("Exponential schedule 1:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log3.write(f"{pair[0]},{pair[1]}\n")
if log_results: log7.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")
t0 = time.perf_counter()
new_route, graph = simulated_annealing(route, exponential_schedule(0.9))
t1 = time.perf_counter() - t0
print("Exponential schedule 2:")
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")
print(f"Time:   {t1:.3f} s")
for pair in graph:
    if log_results: log4.write(f"{pair[0]},{pair[1]}\n")
if log_results: log8.write(f"{t1},{abs((optimal_length - tour_length(new_route)) / optimal_length) * 100}\n")
print("")

if log_results: log1.close()
if log_results: log2.close()
if log_results: log3.close()
if log_results: log4.close()
if log_results: log5.close()
if log_results: log6.close()
if log_results: log7.close()
if log_results: log8.close()

##################################################
print("\nCOOL")
input("BEGIN mst ...\n")
##################################################

if log_results: log1 = open("log5-1.csv", "w", encoding = "utf-8")
if log_results: log2 = open("log5-2.csv", "w", encoding = "utf-8")
if log_results: log1.write("Ratio\n")
if log_results: log2.write("MST,MST + 2-opt,Local Search,Simulated Annealing\n")

def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children

def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    return [root] + [c for child in children[root] for c in preorder(children, child)]

def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

def tsp_2_opt(points, tour):
    best_tour = list(tour)
    improved = True
    n = len(tour)

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                if j - i == 1 and i == 0 and j == n:
                    continue

                new_tour = two_opt_move(best_tour, i, j)

                if tour_length(points, new_tour) < tour_length(points, best_tour):
                    best_tour = new_tour
                    improved = True
                    break
            if improved:
                break

    return best_tour

def simulated_annealing(points, tour):
    current = list(tour)
    t = 0
    T = 500
    while True:
        t += 1
        T = T * 0.5
        if t > 1000 or T <= 0:
            return current
        new = two_opt_move_random(current)
        delta_E = tour_length(points, new) - tour_length(points, current)
        if delta_E <= 0:
            current = list(new)
        else:
            if roll(math.exp(-1 * delta_E / T)):
                current = list(new)

for i in range(100):
    random.seed(i)
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(9)]

    approx1 = preorder(mst_prim(points))
    approx2 = tsp_2_opt(points, approx1)
    approx3 = list(approx1)
    random.shuffle(approx3)
    approx3 = tsp_2_opt(points, approx3)
    approx4 = list(approx1)
    random.shuffle(approx4)
    approx4 = simulated_annealing(points, approx4)
    optimal = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
                  key=lambda t: tour_length(points, t))

    print(f"MST approximation: {approx1}  L = {tour_length(points, approx1):.2f}")
    print(f"+ 2-opt:           {approx2}  L = {tour_length(points, approx2):.2f}")
    print(f"Optimal (brute):   {list(optimal)}  L = {tour_length(points, optimal):.2f}")
    print(f"Ratio = {tour_length(points, approx1) / tour_length(points, optimal):.3f}  (guaranteed <= 2)")
    if log_results: log1.write(f"{tour_length(points, approx1) / tour_length(points, optimal)}\n")
    if log_results: log2.write(f"{tour_length(points, approx1)},{tour_length(points, approx2)},{tour_length(points, approx3)},{tour_length(points, approx4)}\n")
    if i < 99:
        print("")

if log_results: log1.close()
if log_results: log2.close()

##################################################
print("\nCOOL")
input("END OF PROGRAM ...\n")
##################################################
