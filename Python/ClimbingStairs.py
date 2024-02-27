# You are climbing a staircase. It takes n steps to reach the top.

# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

# Example 1:

# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps
# Example 2:

# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
 

# Constraints:

# 1 <= n <= 45

# https://leetcode.com/problems/climbing-stairs/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions


# This Solution uses dynamic programming and a bottom up approach to resolve it
# One can only take 1 or 2 steps at once
# Assuming from the last step you can only go to the top with 1 step
# and from the one prior, there's 2 ways, we can use two variables first and second
# to store the possibilities. It resembles the fibonnaci sequence
class Solution:
    def climbStairs(self, n: int) -> int:
        first = 1
        second = 1
        for i in range(n-1): 
            temp = first
            first = first + second
            second = temp
        return first