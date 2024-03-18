# You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

# Find two lines that together with the x-axis form a container, such that the container contains the most water.

# Return the maximum amount of water a container can store.

# Notice that you may not slant the container.

 

# Example 1:


# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49
# Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
# Example 2:

# Input: height = [1,1]

# https://leetcode.com/problems/container-with-most-water/description/?envType=study-plan-v2&envId=leetcode-75


class Solution:
    def maxArea(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        minH = left if height[left] < height[right] else right
        mArea = height[minH] * (right-left)
        while left < right:
            minH = left if height[left] < height[right] else right
            currArea = height[minH] * (right-left)
            if currArea > mArea:
                mArea = currArea
            if minH == left:
                left += 1
            else:
                right -= 1
        return mArea
        