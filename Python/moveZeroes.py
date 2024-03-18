# Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

# Note that you must do this in-place without making a copy of the array.

 

# Example 1:

# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
# Example 2:

# Input: nums = [0]
# Output: [0]
 

# Constraints:

# 1 <= nums.length <= 104
# -231 <= nums[i] <= 231 - 1
 

# Follow up: Could you minimize the total number of operations done?

# https://leetcode.com/problems/move-zeroes/description/?envType=study-plan-v2&envId=leetcode-75

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return nums
        left = 0
        right = 1
        while right < len(nums):
            if nums[left] == 0 and nums[right] != 0:
                nums[left] = nums[right]
                nums[right] = 0
            if nums[left] != 0:
                left = left +1
            right += 1
        return nums
