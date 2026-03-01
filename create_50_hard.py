import os
from datasets import Dataset

tasks_data = [
    ("dp_fib", "Write a Python function `solve(n: int) -> int` that returns the n-th Fibonacci number modulo 1000000007. solve(0)=0, solve(1)=1. n can be up to 10^5.", "assert solve(10) == 55\n    assert solve(100) == 687995182"),
    ("dp_lis", "Write a Python function `solve(arr: list[int]) -> int` that returns the length of the longest strictly increasing subsequence in `arr`.", "assert solve([10,9,2,5,3,7,101,18]) == 4"),
    ("dp_lcs", "Write a Python function `solve(s1: str, s2: str) -> int` returning the length of the longest common subsequence.", "assert solve('abcde', 'ace') == 3"),
    ("dp_edit", "Write a Python function `solve(word1: str, word2: str) -> int` returning the minimum number of operations (insert, delete, replace) to convert word1 to word2.", "assert solve('horse', 'ros') == 3"),
    ("dp_knapsack", "Write a Python function `solve(weights: list[int], values: list[int], W: int) -> int` returning the max value for 0/1 knapsack with capacity W.", "assert solve([1,2,3], [6,10,12], 5) == 22"),
    ("dp_coin", "Write a Python function `solve(coins: list[int], amount: int) -> int` returning the fewest number of coins to make `amount`. If not possible, return -1.", "assert solve([1,2,5], 11) == 3\n    assert solve([2], 3) == -1"),
    ("dp_ways", "Write a Python function `solve(coins: list[int], amount: int) -> int` returning the number of combinations that make up that amount.", "assert solve([1,2,5], 5) == 4"),
    ("dp_maxsub", "Write a Python function `solve(nums: list[int]) -> int` returning the sum of the contiguous subarray with the largest sum.", "assert solve([-2,1,-3,4,-1,2,1,-5,4]) == 6"),
    ("dp_rob", "Write a Python function `solve(nums: list[int]) -> int` returning max money you can rob from houses without robbing adjacent ones.", "assert solve([2,7,9,3,1]) == 12"),
    ("dp_climb", "Write a Python function `solve(n: int) -> int` returning number of distinct ways to climb n steps (1 or 2 at a time).", "assert solve(3) == 3\n    assert solve(5) == 8"),
    ("graph_islands", "Write a Python function `solve(grid: list[list[str]]) -> int` returning the number of islands ('1's connected 4-directionally).", "assert solve([['1','1','0'],['0','1','0'],['0','0','1']]) == 2"),
    ("graph_components", "Write a Python function `solve(n: int, edges: list[list[int]]) -> int` returning number of connected components in an undirected graph.", "assert solve(5, [[0,1], [1,2], [3,4]]) == 2"),
    ("graph_cycle", "Write a Python function `solve(numCourses: int, prerequisites: list[list[int]]) -> bool` returning True if you can finish all courses (no cycles in directed graph), else False.", "assert solve(2, [[1,0]]) == True\n    assert solve(2, [[1,0],[0,1]]) == False"),
    ("graph_topo", "Write a Python function `solve(numCourses: int, prerequisites: list[list[int]]) -> list[int]` returning a valid topological sort. If impossible, return [].", "ans = solve(2, [[1,0]]); assert ans == [0,1]"),
    ("graph_bipartite", "Write a Python function `solve(graph: list[list[int]]) -> bool` returning True if the undirected graph is bipartite.", "assert solve([[1,2,3],[0,2],[0,1,3],[0,2]]) == False"),
    ("str_anagram", "Write a Python function `solve(s: str, t: str) -> bool` returning True if t is an anagram of s.", "assert solve('anagram', 'nagaram') == True"),
    ("str_palin", "Write a Python function `solve(s: str) -> bool` returning True if s is a palindrome considering only alphanumeric characters and ignoring cases.", "assert solve('A man, a plan, a canal: Panama') == True"),
    ("str_longest_palin", "Write a Python function `solve(s: str) -> str` returning the longest palindromic substring.", "assert solve('babad') in ['bab', 'aba']"),
    ("str_prefix", "Write a Python function `solve(strs: list[str]) -> str` returning the longest common prefix.", "assert solve(['flower','flow','flight']) == 'fl'"),
    ("str_substring", "Write a Python function `solve(s: str) -> int` returning the length of the longest substring without repeating characters.", "assert solve('abcabcbb') == 3"),
    ("math_prime", "Write a Python function `solve(n: int) -> int` returning the number of prime numbers strictly less than n.", "assert solve(10) == 4"),
    ("math_gcd", "Write a Python function `solve(a: int, b: int) -> int` computing the greatest common divisor.", "assert solve(12, 18) == 6"),
    ("math_pow", "Write a Python function `solve(x: float, n: int) -> float` computing x raised to the power n. Assume valid ranges.", "assert abs(solve(2.0, 10) - 1024.0) < 1e-5"),
    ("math_sqrt", "Write a Python function `solve(x: int) -> int` computing the square root rounded down to the nearest integer.", "assert solve(8) == 2"),
    ("math_factorial", "Write a Python function `solve(n: int) -> int` returning the trailing zeroes in n!.", "assert solve(5) == 1\n    assert solve(10) == 2"),
    ("arr_two_sum", "Write a Python function `solve(nums: list[int], target: int) -> list[int]` returning indices of the two numbers that add up to target.", "assert sorted(solve([2,7,11,15], 9)) == [0,1]"),
    ("arr_three_sum", "Write a Python function `solve(nums: list[int]) -> list[list[int]]` returning all unique triplets [nums[i], nums[j], nums[k]] that sum to 0. Order doesn't matter.", "res = solve([-1,0,1,2,-1,-4]); assert sorted([sorted(x) for x in res]) == [[-1,-1,2], [-1,0,1]]"),
    ("arr_water", "Write a Python function `solve(height: list[int]) -> int` returning the max area of water a container can store.", "assert solve([1,8,6,2,5,4,8,3,7]) == 49"),
    ("arr_trap", "Write a Python function `solve(height: list[int]) -> int` returning how much trapped rainwater can be caught.", "assert solve([0,1,0,2,1,0,1,3,2,1,2,1]) == 6"),
    ("arr_product", "Write a Python function `solve(nums: list[int]) -> list[int]` returning an array such that answer[i] is equal to product of all elements except nums[i].", "assert solve([1,2,3,4]) == [24,12,8,6]"),
    ("arr_merge", "Write a Python function `solve(intervals: list[list[int]]) -> list[list[int]]` merging all overlapping intervals.", "assert solve([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]"),
    ("arr_insert", "Write a Python function `solve(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]` inserting newInterval and merging if needed.", "assert solve([[1,3],[6,9]], [2,5]) == [[1,5],[6,9]]"),
    ("arr_rotate", "Write a Python function `solve(nums: list[int], k: int) -> list[int]` returning array shifted right by k steps.", "assert solve([1,2,3,4,5,6,7], 3) == [5,6,7,1,2,3,4]"),
    ("arr_search", "Write a Python function `solve(nums: list[int], target: int) -> int` returning the index of target in a sorted ascending array, or -1.", "assert solve([-1,0,3,5,9,12], 9) == 4"),
    ("arr_search_rot", "Write a Python function `solve(nums: list[int], target: int) -> int` returning index of target in rotated sorted array.", "assert solve([4,5,6,7,0,1,2], 0) == 4"),
    ("bit_single", "Write a Python function `solve(nums: list[int]) -> int` finding the single element that doesn't appear twice.", "assert solve([4,1,2,1,2]) == 4"),
    ("bit_single_two", "Write a Python function `solve(nums: list[int]) -> list[int]` finding the two elements that appear only once. Return sorted.", "assert solve([1,2,1,3,2,5]) == [3,5]"),
    ("bit_ones", "Write a Python function `solve(n: int) -> int` returning the number of '1' bits in the unsigned 32-bit integer.", "assert solve(11) == 3"),
    ("bit_reverse", "Write a Python function `solve(n: int) -> int` reversing the bits of a 32-bit unsigned integer.", "assert solve(43261596) == 964176192"),
    ("bit_missing", "Write a Python function `solve(nums: list[int]) -> int` returning the only number missing from the range [0, len(nums)].", "assert solve([3,0,1]) == 2"),
    ("tree_bst", "Write a Python function `solve(preorder: list[int]) -> bool` checking if it can represent the preorder traversal of a Binary Search Tree.", "assert solve([5,2,1,3,6]) == True\n    assert solve([5,2,6,1,3]) == False"),
    ("str_paren", "Write a Python function `solve(s: str) -> bool` returning True if parentheses (), {}, [] are balanced.", "assert solve('()[]{}') == True\n    assert solve('(]') == False"),
    ("str_gen_paren", "Write a Python function `solve(n: int) -> list[str]` returning all combinations of n pairs of well-formed parentheses.", "assert sorted(solve(3)) == sorted(['((()))', '(()())', '(())()', '()(())', '()()()'])"),
    ("num_roman", "Write a Python function `solve(s: str) -> int` converting Roman numeral to integer.", "assert solve('MCMXCIV') == 1994"),
    ("num_int_roman", "Write a Python function `solve(num: int) -> str` converting integer to Roman numeral.", "assert solve(1994) == 'MCMXCIV'"),
    ("misc_lru", "Write a Python class `LRUCache` with methods `__init__(capacity)`. `get(key)` returns value or -1. `put(key, value)` adds or updates. Do not use an externally defined function, just the class.", "c = LRUCache(2)\n    c.put(1,1)\n    c.put(2,2)\n    assert c.get(1) == 1\n    c.put(3,3)\n    assert c.get(2) == -1"),
    ("misc_trie", "Write a Python class `Trie` with `__init__()`, `insert(word)`, `search(word)->bool`, and `startsWith(prefix)->bool`.", "t = Trie()\n    t.insert('apple')\n    assert t.search('apple') == True\n    assert t.search('app') == False\n    assert t.startsWith('app') == True"),
    ("misc_median", "Write a Python class `MedianFinder` with `addNum(n)` and `findMedian()->float`.", "m = MedianFinder()\n    m.addNum(1)\n    m.addNum(2)\n    assert m.findMedian() == 1.5\n    m.addNum(3)\n    assert m.findMedian() == 2.0"),
    ("misc_sudoku", "Write a Python function `solve(board: list[list[str]]) -> bool` returning True if a 9x9 partially filled Sudoku board is valid.", "board = [['5','3','.','.','7','.','.','.','.'],['6','.','.','1','9','5','.','.','.'],['.','9','8','.','.','.','.','6','.'],['8','.','.','.','6','.','.','.','3'],['4','.','.','8','.','3','.','.','1'],['7','.','.','.','2','.','.','.','6'],['.','6','.','.','.','.','2','8','.'],['.','.','.','4','1','9','.','.','5'],['.','.','.','.','8','.','.','7','9']]; assert solve(board) == True"),
    ("misc_word_search", "Write a Python function `solve(board: list[list[str]], word: str) -> bool` returning True if word exists in the grid (adjacent vertically/horizontally).", "board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']]; assert solve(board, 'ABCCED') == True\n    assert solve(board, 'ABCB') == False")
]

tasks = []
for idx, (t_id, prompt, tests) in enumerate(tasks_data):
    if "LRUCache" in prompt or "Trie" in prompt or "MedianFinder" in prompt:
         tests_str = f"def test():\n    {tests}\ntest()"
    else:
         tests_str = f"def test():\n    {tests}\ntest()"
         
    tasks.append({
        "id": t_id,
        "prompt": prompt,
        "tests": tests_str
    })

ds = Dataset.from_list(tasks)
ds.save_to_disk('./custom_50_hard')
print('Created custom_50_hard dataset with 50 distinct algorithmic challenges.')
