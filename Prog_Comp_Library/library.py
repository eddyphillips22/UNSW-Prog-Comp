import sys
import re
import string
import math
import random
import heapq
from collections import defaultdict
from math import gcd


#// STRING MANIPULATION FUNCTIONS //#-------------------------------------------------------------------

def read_lines(path):
    """
    Read a text file and return a list of lines (newline stripped).
    Usage:
        lines = read_lines('input.txt')
    """
    with open(path, 'r') as f:
        return [line.rstrip('\n') for line in f]

def input(): 
    return sys.stdin.readline().rstrip('\n')

def normalize(s):
    """
    Strip out non-alphanumeric characters and lowercase the rest.
    Use before palindrome checks to ignore punctuation.
    """
    return ''.join(ch.lower() for ch in s if ch.isalnum())

def is_palindrome(s):
    """
    Return True if `s` is a palindrome (ignoring case/punctuation).
    Usage:
        is_palindrome("A man, a plan, a canal: Panama")  # → True
    """
    ns = normalize(s)
    return ns == ns[::-1]


def is_pangram(s):
    """
    Return True if `s` contains every letter a–z at least once.
    Usage:
        is_pangram("The quick brown fox jumps over the lazy dog")  # → True
    """
    return set(string.ascii_lowercase) <= set(s.lower())


def insert_string(x, y, pos):
    """
    Insert string `x` into `y` at index `pos`.
    Usage:
        insert_string("XX", "hello", 2)  # → "heXXllo"
    """
    return y[:pos] + x + y[pos:]    

def read_ints():
    """
    Read a line and map to ints: a, b, c = read_ints()
    """
    return map(int, sys.stdin.readline().split())

def read_list_of_ints():
    """Read a line and return a list of ints"""
    return list(read_ints())

def read_matrix(rows, dtype=int):
    """
    Read `rows` lines of space-separated values of type `dtype`
    """
    return [list(map(dtype, sys.stdin.readline().split())) for _ in range(rows)]

#// MATH FUNCTIONS //#-------------------------------------------------------------------------------------

def fibonacci_pairs():
    """
    Generator of consecutive Fibonacci pairs (a, b).
    Usage:
        gen = fibonacci_pairs()
        next(gen)  # → (1, 1)
        next(gen)  # → (1, 2)
    """
    a, b = 1, 1
    while True:
        yield a, b
        a, b = b, a + b
        
def find_golden_ratio(threshold=1e-14):
    """
    Find an approximation to the golden ratio using fib ratios.
    Returns (phi, error, (a, b)) for the first pair within error < threshold.
    Usage:
        phi, err, (a, b) = find_golden_ratio()
    """
    for a, b in fibonacci_pairs():
        phi = b / a
        err = abs(phi*phi - phi - 1)
        if err < threshold:
            return phi, err, (a, b)

def count_correct_times_table(grid):
    """
    Given an n×n grid (0-indexed list of lists) of ints, count how many cells
    equal (row+1)*(col+1).
    Usage:
        grid = [[1,2],[2,4]]
        count_correct_times_table(grid)  # → 4
    """
    n = len(grid)
    return sum(
        1
        for i in range(n)
        for j in range(n)
        if grid[i][j] == (i+1)*(j+1)
    )

def lcm(a, b):
    """
    Compute least common multiple via gcd.
    Usage:
        lcm(12, 15)  # → 60
    """
    return a // gcd(a, b) * b

def modexp(a, e, M):
    """
    Fast modular exponentiation: returns (a^e) % M in O(log e).
    Usage:
        modexp(2, 10, 1000)  # → 24
    """
    r = 1
    a %= M
    while e:
        if e & 1:
            r = (r * a) % M
        a = (a * a) % M 
        e >>= 1
    return r

#// SIMULATIONS AND PROBABILITIES \\--------------------------------------------------------------------

def estimate_probability(event_func, trials=100_000):
    """
    Run `event_func()` `trials` times and return the fraction of True results.
    Usage:
        # approximate probability that a fair coin lands heads
        import random
        print(estimate_probability(lambda: random.choice([True, False])))
    """
    count = 0
    for _ in range(trials):
        if event_func():
            count += 1
    return count / trials

def card_game_event():
    """
    Example event for simulating a simple “red vs. black” card game.
    Returns True if red deck wins.
    Usage:
        print(estimate_probability(card_game_event, 50000))
    """
    # prepare decks
    red = list(range(26))
    black = list(range(26, 52))
    random.shuffle(red)
    random.shuffle(black)
    # draw until one empties
    while red and black:
        if red.pop() > black.pop():
            return True
    return bool(red)

#// PATTERNS AND FORMATTING \\------------------------------------------------------------------

def glowing_tree(n):
    """
    Print a centered pyramid (“glowing tree”) of height n using '*' characters.
    Usage:
        glowing_tree(3)
        #   *
        #  ***
        # *****
    """
    for i in range(n):
        print(' '*(n - i - 1) + '*'*(2*i + 1))
        
def generate_index(words):
    """
    Given a list of words, build and print a word→sorted list of line numbers.
    Usage:
        generate_index(["apple", "banana", "apple"])
        # apple: 1, 3
        # banana: 2
    """
    idx = defaultdict(list)
    for lineno, word in enumerate(words, 1):
        idx[word].append(lineno)
    for word in sorted(idx):
        print(f"{word}: {', '.join(map(str, idx[word]))}")

#// SORTING AND SEARCHING \\------------------------------------------------------------------

def linear_search(arr, target):
    """
    Linear (sequential) search.
    Time: O(n), Space: O(1)
    Returns the index of the first occurrence of target in arr, or -1 if not found.
    """
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1

def binary_search(arr, target):
    """
    Iterative binary search on a sorted array.
    Time: O(log n), Space: O(1)
    Returns the index of target, or -1 if not found.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def binary_search_recursive(arr, target, lo=0, hi=None):
    """
    Recursive binary search.
    Time: O(log n), Space: O(log n) due to recursion.
    """
    if hi is None:
        hi = len(arr) - 1
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid+1, hi)
    else:
        return binary_search_recursive(arr, target, lo, mid-1)
    
def jump_search(arr, target):
    """
    Jump search on a sorted array.
    Time: O(√n), Space: O(1)
    """
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    # find block
    while prev < n and arr[min(n-1, prev+step)] < target:
        prev += step
    # linear search within block
    for i in range(prev, min(prev+step, n)):
        if arr[i] == target:
            return i
    return -1

def interpolation_search(arr, target):
    """
    Interpolation search on uniformly distributed sorted array.
    Time avg: O(log log n), worst O(n), Space: O(1)
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi and target >= arr[lo] and target <= arr[hi]:
        # avoid division by zero
        if arr[hi] == arr[lo]:
            if arr[lo] == target:
                return lo
            break
        # probe
        pos = lo + ((target - arr[lo]) * (hi - lo)) // (arr[hi] - arr[lo])
        if pos < lo or pos > hi:
            break
        if arr[pos] == target:
            return pos
        if arr[pos] < target:
            lo = pos + 1
        else:
            hi = pos - 1
    return -1

def bubble_sort(arr):
    """
    Bubble sort.
    Time: O(n²), Space: O(1)
    Returns a new sorted list.
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if a[j] > a[j+1]:
                a[j], a[j+1] = a[j+1], a[j]
                swapped = True
        if not swapped:
            break
    return a

def selection_sort(arr):
    """
    Selection sort.
    Time: O(n²), Space: O(1)
    """
    a = arr.copy()
    n = len(a)
    for i in range(n):
        # find min in a[i:]
        m = i
        for j in range(i+1, n):
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    return a

def insertion_sort(arr):
    """
    Insertion sort.
    Time: O(n²), Space: O(1)
    """
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
        a[j+1] = key
    return a

def merge_sort(arr):
    """
    Merge sort.
    Time: O(n log n), Space: O(n)
    """
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr)//2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def quick_sort(arr):
    """
    Quick sort (in-place Lomuto partition).
    Time avg: O(n log n), worst O(n²), Space: O(log n) recursion.
    """
    a = arr.copy()
    def _qs(lo, hi):
        if lo < hi:
            p = partition(lo, hi)
            _qs(lo, p-1)
            _qs(p+1, hi)
    def partition(lo, hi):
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] < pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        return i
    _qs(0, len(a)-1)
    return a

def heap_sort(arr):
    """
    Heap sort using heapq.
    Time: O(n log n), Space: O(n)
    """
    a = arr.copy()
    heapq.heapify(a)
    return [ heapq.heappop(a) for _ in range(len(a)) ]