from functools import lru_cache
import cProfile

# @lru_cache(maxsize=128)  # Limit the cache to 128 results
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
print(fibonacci(100))  # Output: 55

num = 42
print(f"The memory address of num is: {id(num)}")