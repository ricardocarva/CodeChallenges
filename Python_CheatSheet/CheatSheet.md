## Basics

1. [Variables](#variables)
2. [If-statements](#if-statements)
3. [Loops](#loops)
4. [Math](#math)
5. [Arrays](#arrays)
6. [Strings](#strings)
7. [Queues](#queues)
8. [HashSets](#hashSsets)
9. [HashMaps](#hashmaps)
10. [Tuples](#tuples)
11. [Heaps](#heaps)
12. [Functions](#functions)
13. [Classes](#classes)
14. [Data Structures](#data-structures)

### Variables

```
# Variables are dynamicly typed
n = 0
print('n =', n)
>>> n = 0

n = "abc"
print('n =', n)
>>> n = abc

# Multiple assignments
n, m = 0, "abc"
n, m, z = 0.125, "abc", False

# Increment
n = n + 1 # good
n += 1    # good
n++       # bad

# None is null (absence of value)
n = 4
n = None
print("n =", n)
>>> n = None

```

### If-statements
```# If statements don't need parentheses 
# or curly braces.
n = 1
if n > 2:
    n -= 1
elif n == 2:
    n *= 2
else:
    n += 2

# Parentheses needed for multi-line conditions.
# and = &&
# or  = ||
n, m = 1, 2
if ((n > 2 and 
    n != m) or n == m):
    n += 1

```

### Loops

```
n = 5
while n < 5:
    print(n)
    n += 1

# Looping from i = 0 to i = 4
for i in range(5):
    print(i)

# Looping from i = 2 to i = 5
for i in range(2, 6):
    print(i)

# Looping from i = 5 to i = 2
for i in range(5, 1, -1):
    print(i)
```

### Math
```
# Division is decimal by default
print(5 / 2)

# Double slash rounds down
print(5 // 2)

# CAREFUL: most languages round towards 0 by default
# So negative numbers will round down
print(-3 // 2)

# A workaround for rounding towards zero
# is to use decimal division and then convert to int.
print(int(-3 / 2))


# Modding is similar to most languages
print(10 % 3)

# Except for negative values
print(-10 % 3)

# To be consistent with other languages modulo
import math
from multiprocessing import heap
print(math.fmod(-10, 3))

# More math helpers
print(math.floor(3 / 2))
print(math.ceil(3 / 2))
print(math.sqrt(2))
print(math.pow(2, 3))

# Max / Min Int
float("inf")
float("-inf")

# Python numbers are infinite so they never overflow
print(math.pow(2, 200))

# But still less than infinity
print(math.pow(2, 200) < float("inf"))
```

### Arrays
```
# Arrays (called lists in python)
arr = [1, 2, 3]
print(arr)

# Can be used as a stack
arr.append(4)
arr.append(5)
print(arr)

arr.pop()
print(arr)

arr.insert(1, 7)
print(arr)

arr[0] = 0
arr[3] = 0
print(arr)

# Initialize arr of size n with default value of 1
n = 5
arr = [1] * n
print(arr)
print(len(arr))

# Careful: -1 is not out of bounds, it's the last value
arr = [1, 2, 3]
print(arr[-1])

# Indexing -2 is the second to last value, etc.
print(arr[-2])

# Sublists (aka slicing)
arr = [1, 2, 3, 4]
print(arr[1:3])

# Similar to for-loop ranges, last index is non-inclusive
print(arr[0:4])

# But no out of bounds error
print(arr[0:10])

# Unpacking
a, b, c = [1, 2, 3]
print(a, b, c)

# Be careful though
# a, b = [1, 2, 3]

# Loop through arrays
nums = [1, 2, 3]

# Using index
for i in range(len(nums)):
    print(nums[i])

# Without index
for n in nums:
    print(n)

# With index and value
for i, n in enumerate(nums):
    print(i, n)

# Loop through multiple arrays simultaneously with unpacking
nums1 = [1, 3, 5]
nums2 = [2, 4, 6]
for n1, n2 in zip(nums1, nums2):
    print(n1, n2)

# Reverse
nums = [1, 2, 3]
nums.reverse()
print(nums)

# Sorting
arr = [5, 4, 7, 3, 8]
arr.sort()
print(arr)

arr.sort(reverse=True)
print(arr)

arr = ["bob", "alice", "jane", "doe"]
arr.sort()
print(arr)

# Custom sort (by length of string)
arr.sort(key=lambda x: len(x))
print(arr)


# List comprehension
arr = [i for i in range(5)]
print(arr)

# 2-D lists
arr = [[0] * 4 for i in range(4)]
print(arr)
print(arr[0][0], arr[3][3])

# This won't work
# arr = [[0] * 4] * 4
```
### Strings
```
# Strings are similar to arrays
s = "abc"
print(s[0:2])

# But they are immutable
# s[0] = "A"

# So this creates a new string
s += "def"
print(s)

# Valid numeric strings can be converted
print(int("123") + int("123"))

# And numbers can be converted to strings
print(str(123) + str(123))

# In rare cases you may need the ASCII value of a char
print(ord("a"))
print(ord("b"))

# Combine a list of strings (with an empty string delimitor)
strings = ["ab", "cd", "ef"]
print("".join(strings))
```
### Queues
```
# Queues (double ended queue)
from collections import deque

queue = deque()
queue.append(1)
queue.append(2)
print(queue)

queue.popleft()
print(queue)

queue.appendleft(1)
print(queue)

queue.pop()
print(queue)
```
### HashSets
```
# HashSet
mySet = set()

mySet.add(1)
mySet.add(2)
print(mySet)
print(len(mySet))

print(1 in mySet)
print(2 in mySet)
print(3 in mySet)

mySet.remove(2)
print(2 in mySet)

# list to set
print(set([1, 2, 3]))

# Set comprehension
mySet = { i for i in range(5) }
print(mySet)
```
### HashMaps
```
# HashMap (aka dict)
myMap = {}
myMap["alice"] = 88
myMap["bob"] = 77
print(myMap)
print(len(myMap))

myMap["alice"] = 80
print(myMap["alice"])

print("alice" in myMap)
myMap.pop("alice")
print("alice" in myMap)

myMap = { "alice": 90, "bob": 70 }
print(myMap)

# Dict comprehension
myMap = { i: 2*i for i in range(3) }
print(myMap)

# Looping through maps
myMap = { "alice": 90, "bob": 70 }
for key in myMap:
    print(key, myMap[key])

for val in myMap.values():
    print(val)

for key, val in myMap.items():
    print(key, val)
```
### Tuples
```
# Tuples are like arrays but immutable
tup = (1, 2, 3)
print(tup)
print(tup[0])
print(tup[-1])

# Can't modify
# tup[0] = 0

# Can be used as key for hash map/set
myMap = { (1,2): 3 }
print(myMap[(1,2)])

mySet = set()
mySet.add((1, 2))
print((1, 2) in mySet)

# Lists can't be keys
# myMap[[3, 4]] = 5
```
### Heap
```
import heapq

# under the hood are arrays
minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)

# Min is always at index 0
print(minHeap[0])

while len(minHeap):
    print(heapq.heappop(minHeap))

# No max heaps by default, work around is
# to use min heap and multiply by -1 when push & pop.
maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)

# Max is always at index 0
print(-1 * maxHeap[0])

while len(maxHeap):
    print(-1 * heapq.heappop(maxHeap))

# Build heap from initial values
arr = [2, 1, 8, 4, 5]
heapq.heapify(arr)
while arr:
    print(heapq.heappop(arr))
```
### Functions
```
def myFunc(n, m):
    return n * m

print(myFunc(3, 4))

# Nested functions have access to outer variables
def outer(a, b):
    c = "c"

    def inner():
        return a + b + c
    return inner()

print(outer("a", "b"))

# Can modify objects but not reassign
# unless using nonlocal keyword
def double(arr, val):
    def helper():
        # Modifying array works
        for i, n in enumerate(arr):
            arr[i] *= 2
        
        # will only modify val in the helper scope
        # val *= 2

        # this will modify val outside helper scope
        nonlocal val
        val *= 2
    helper()
    print(arr, val)

nums = [1, 2]
val = 3
double(nums, val)
```
### Classes
```
class MyClass:
    # Constructor
    def __init__(self, nums):
        # Create member variables
        self.nums = nums
        self.size = len(nums)
    
    # self key word required as param
    def getLength(self):
        return self.size

    def getDoubleLength(self):
        return 2 * self.getLength()

myObj = MyClass([1, 2, 3])
print(myObj.getLength())
print(myObj.getDoubleLength())
```
## Data Structures

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
