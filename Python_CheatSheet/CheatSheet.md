

## Data Structures:

1. [Lists:](#lists) Python's built-in list data structure is versatile and widely used for storing collections of items.
2. [Strings:](#Strings) Strings are immutable sequences of characters and are extensively used for handling textual data.
3. [Dictionaries:](#Dictionaries) Dictionaries are collections of key-value pairs, allowing efficient lookup and manipulation.
4. [Sets:](#Sets) Sets are unordered collections of unique elements, useful for tasks like eliminating duplicates.
5. [Trees:](#Trees): Often implemented using classes and pointers, trees are crucial for many algorithm problems like binary search trees (BSTs), heaps, etc.
6. [Graphs:](#Graphs): Graphs can be represented using adjacency lists or matrices and are essential for problems involving relationships between elements.

Functions:
1. Sorting: sorted() function or sort() method for sorting lists.
2. Searching: binary_search() or bisect module for efficient searching.
3. Regular Expressions: re module for pattern matching in strings.
4. Math Operations: math module for mathematical operations.
5. Input/Output: Functions like input() and print() for handling input/output.
6. Recursion: Understanding recursion and using it to solve problems efficiently.
7. Dynamic Programming: Implementing dynamic programming solutions for optimization problems.

Algorithms:
1. Sorting Algorithms: Understanding and implementing algorithms like quicksort, mergesort, heapsort, etc.
2. Search Algorithms: Binary search, depth-first search (DFS), breadth-first search (BFS), etc.
3. Dynamic Programming: Techniques for solving optimization problems by breaking them down into simpler subproblems.
4. Graph Algorithms: Algorithms like Dijkstra's algorithm, Kruskal's algorithm, etc., for solving graph-related problems.
5. Greedy Algorithms: Solving problems by making the locally optimal choice at each stage with the hope of finding a global optimum.
6. Backtracking: A technique for systematically searching through a space of possible solutions.

Additional Libraries:
1. NumPy: For numerical computing and handling arrays.
2. Pandas: For data manipulation and analysis.
3. Matplotlib/Seaborn: For data visualization.
4. Scikit-learn: For machine learning tasks.
5. NetworkX: For graph analysis and manipulation.

## Lists:
### Example 1: Basic list operations
#### Creating a list
my_list = [1, 2, 3, 4, 5]

#### Accessing elements
print(my_list[0])  # Output: 1

#### Adding elements
my_list.append(6)
print(my_list)  # Output: [1, 2, 3, 4, 5, 6]

#### Removing elements
my_list.remove(3)
print(my_list)  # Output: [1, 2, 4, 5, 6]

## Strings:
Example 2: String manipulation

#### Concatenation
greeting = "Hello"
name = "John"
message = greeting + ", " + name + "!"
print(message)  # Output: Hello, John!

#### Splitting
sentence = "This is a sample sentence"
words = sentence.split()
print(words)  # Output: ['This', 'is', 'a', 'sample', 'sentence']

#### Substring
substring = sentence[5:7]
print(substring)  # Output: is

## Dictionaries:
Example 3: Dictionary operations


#### Creating a dictionary
my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}

#### Accessing values
print(my_dict['name'])  # Output: John

#### Adding a new key-value pair
my_dict['occupation'] = 'Engineer'
print(my_dict)  # Output: {'name': 'John', 'age': 30, 'city': 'New York', 'occupation': 'Engineer'}

# Removing a key-value pair
del my_dict['age']
print(my_dict)  # Output: {'name': 'John', 'city': 'New York', 'occupation': 'Engineer'}

## Sets:
Example 4: Set operations

# Creating a set
my_set = {1, 2, 3, 4, 5}

# Adding elements
my_set.add(6)
print(my_set)  # Output: {1, 2, 3, 4, 5, 6}

# Removing elements
my_set.remove(3)
print(my_set)  # Output: {1, 2, 4, 5, 6}

## Trees (Binary Search Tree):
Example 5: BST implementation
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Insertion
def insert(root, value):
    if root is None:
        return TreeNode(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)
    return root

# Inorder traversal
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.value)
        inorder_traversal(root.right)

# Example usage
root = None
root = insert(root, 5)
root = insert(root, 3)
root = insert(root, 7)
root = insert(root, 1)
root = insert(root, 4)

inorder_traversal(root)  # Output: 1, 3, 4, 5, 7

## Graphs (Adjacency List Representation):
Example 6: Graph representation and traversal

class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append(v)

    def dfs(self, node, visited=set()):
        if node not in visited:
            print(node)
            visited.add(node)
            if node in self.adj_list:
                for neighbor in self.adj_list[node]:
                    self.dfs(neighbor, visited)

# Example usage
graph = Graph()
graph.add_edge(0, 1)
graph.add_edge(0, 2)
graph.add_edge(1, 2)
graph.add_edge(2, 0)
graph.add_edge(2, 3)
graph.add_edge(3, 3)

graph.dfs(2)  # Output: 2, 0, 1, 3


## Sorting:
### Example 1: Using the sorted() function
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5]
sorted_list = sorted(my_list)
print(sorted_list)  # Output: [1, 1, 2, 3, 4, 5, 5, 6, 9]

### Example 2: Using the sort() method
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5]
my_list.sort()
print(my_list)  # Output: [1, 1, 2, 3, 4, 5, 5, 6, 9]

Searching:
### Example 3: Using the bisect module for binary search

import bisect
my_list = [1, 3, 5, 7, 9]
index = bisect.bisect_left(my_list, 5)
print(index)  # Output: 2

### Regular Expressions:
Example 4: Using the re module for pattern matching


import re

text = "The quick brown fox jumps over the lazy dog"
pattern = r'\b\w{3}\b'  # Match 3-letter words
matches = re.findall(pattern, text)
print(matches)  # Output: ['The', 'fox', 'the', 'dog']
Math Operations:

Example 5: Using the math module for mathematical operations


import math

num = 16
square_root = math.sqrt(num)
print(square_root)  # Output: 4.0
Input/Output:

Example 6: Using input() and print() functions


name = input("Enter your name: ")
print("Hello,", name)
Recursion:

Example 7: Recursively calculate factorial


def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(result)  # Output: 120
Dynamic Programming:

Example 8: Fibonacci sequence using dynamic programming


def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

result = fibonacci(6)
print(result)  # Output: 8

Sorting Algorithms:

Example 1: Quicksort algorithm



def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5]
sorted_list = quicksort(my_list)
print(sorted_list)  # Output: [1, 1, 2, 3, 4, 5, 5, 6, 9]
Example 2: Mergesort algorithm



def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5]
sorted_list = mergesort(my_list)
print(sorted_list)  # Output: [1, 1, 2, 3, 4, 5, 5, 6, 9]
Search Algorithms:

Example 3: Binary search algorithm



def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

my_list = [1, 3, 5, 7, 9]
index = binary_search(my_list, 5)
print(index)  # Output: 2
Example 4: Depth-first search (DFS) algorithm



def dfs(graph, node, visited=set()):
    if node not in visited:
        print(node)
        visited.add(node)
        if node in graph:
            for neighbor in graph[node]:
                dfs(graph, neighbor, visited)

# Example usage
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
dfs(graph, 2)  # Output: 2, 0, 1, 3
Dynamic Programming:

Example 5: Fibonacci sequence using dynamic programming
(Already provided in a previous response)
Graph Algorithms:

Example 6: Dijkstra's algorithm for finding shortest paths


import heapq

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
shortest_paths = dijkstra(graph, 'A')
print(shortest_paths)  # Output: {'A': 0, 'B': 1, 'C': 3, 'D': 4}
Greedy Algorithms:

Example 7: Fractional Knapsack problem


def fractional_knapsack(items, capacity):
    items.sort(key=lambda x: x[1] / x[0], reverse=True)
    total_value = 0
    for weight, value in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
        else:
            total_value += (capacity / weight) * value
            break
    return total_value

# Example usage
items = [(10, 60), (20, 100), (30, 120)]
capacity = 50
max_value = fractional_knapsack(items, capacity)
print(max_value)  # Output: 240.0
Backtracking:

Example 8: N-Queens problem


def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col or \
                board[i] - i == col - row or \
                board[i] + i == col + row:
            return False
    return True

def solve_n_queens(n):
    def backtrack(row):
        if row == n:
            solutions.append(list(board))
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(row + 1)

    solutions = []
    board = [-1] * n
    backtrack(0)
    return solutions

# Example usage
n = 4
solutions = solve_n_queens(n)
print(solutions)  # Output: [[1, 3, 0, 2], [2, 0, 3, 1]]

NumPy:

Example 1: Creating and manipulating arrays



import numpy as np

# Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])

# Performing mathematical operations on arrays
squared_arr = arr ** 2
print(squared_arr)  # Output: [ 1  4  9 16 25]

# Reshaping arrays
reshaped_arr = arr.reshape(5, 1)
print(reshaped_arr)
Example 2: Computing statistical measures



# Computing mean and standard deviation
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)
Pandas:

Example 3: Loading and exploring a dataset



import pandas as pd

# Loading a CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Displaying first few rows of the DataFrame
print(df.head())

# Computing summary statistics
print(df.describe())
Example 4: Data manipulation and analysis



# Filtering data
filtered_data = df[df['column'] > 10]

# Grouping and aggregating data
grouped_data = df.groupby('category')['value'].mean()

# Merging/joining DataFrames
merged_df = pd.merge(df1, df2, on='key')
Matplotlib/Seaborn:

Example 5: Creating basic plots with Matplotlib



import matplotlib.pyplot as plt

# Plotting a simple line graph
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function')
plt.show()
Example 6: Creating advanced plots with Seaborn



import seaborn as sns

# Creating a heatmap from a DataFrame
data = np.random.rand(10, 12)
df = pd.DataFrame(data)
sns.heatmap(df, cmap='coolwarm')
plt.show()
Scikit-learn:

Example 7: Training a machine learning model



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
Example 8: Evaluating a machine learning model



from sklearn.metrics import mean_squared_error

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
NetworkX:

Example 9: Creating and analyzing a graph



import networkx as nx

# Creating a graph
G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)

# Analyzing the graph
print("Nodes:", G.nodes())
print("Edges:", G.edges())
print("Degree of node 2:", G.degree(2))
Example 10: Visualizing a graph



# Drawing the graph
nx.draw(G, with_labels=True)
plt.show()
