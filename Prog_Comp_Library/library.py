import sys
import re
import string
import random
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