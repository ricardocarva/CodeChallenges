# Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

# You must implement a solution with a linear runtime complexity and use only constant extra space.

 

# Example 1:

# Input: nums = [2,2,1]
# Output: 1
# Example 2:

# Input: nums = [4,1,2,1,2]
# Output: 4
# Example 3:

# Input: nums = [1]
# Output: 1
 

# Constraints:

# 1 <= nums.length <= 3 * 104
# -3 * 104 <= nums[i] <= 3 * 104
# Each element in the array appears twice except for one element which appears only once.

# https://leetcode.com/problems/single-number/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            # XOR assignment operator
            result ^= num
        return result
    

    Initial result is 0 (in binary: 0000 for simplicity, assuming we're only considering 4-bit representations for demonstration).

# XOR with the first element (4):

# result = 0000
# num = 0100 (binary for 4)
# Perform 0000 ^ 0100 = 0100 (result is now 0100 which is 4 in decimal)
# XOR with the second element (1):

# result = 0100
# num = 0001 (binary for 1)
# Perform 0100 ^ 0001 = 0101 (result is now 0101 which is 5 in decimal)
# XOR with the third element (2):

# result = 0101
# num = 0010 (binary for 2)
# Perform 0101 ^ 0010 = 0111 (result is now 0111 which is 7 in decimal)
# XOR with the fourth element (1 again):

# result = 0111
# num = 0001 (binary for 1)
# Perform 0111 ^ 0001 = 0110 (result is now 0110 which is 6 in decimal; the effect of the first 1 is cancelled)
# XOR with the fifth element (2 again):

# result = 0110
# num = 0010 (binary for 2)
# Perform 0110 ^ 0010 = 0100 (result is now 0100 which is 4 in decimal; the effect of both 2s is cancelled)
# In the end, result holds the binary 0100, which is 4 in decimal, revealing the single number that doesn't appear twice.