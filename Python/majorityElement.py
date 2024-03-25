# Given an array nums of size n, return the majority element.

# The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

 

# Example 1:

# Input: nums = [3,2,3]
# Output: 3
# Example 2:

# Input: nums = [2,2,1,1,1,2,2]
# Output: 2
 

# Constraints:

# n == nums.length
# 1 <= n <= 5 * 104
# -109 <= nums[i] <= 109
 

# Follow-up: Could you solve the problem in linear time and in O(1) space?

# https://leetcode.com/problems/majority-element/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        size = len(nums)
        dic = {}
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        return max(dic, key=dic.get)
    
    # Best solution:
        # count = 0
        # candidate = 0
        
        # for num in nums:
        #     if count == 0:
        #         candidate = num
            
        #     if num == candidate:
        #         count += 1
        #     else:
        #         count -= 1
        
        # return candidate