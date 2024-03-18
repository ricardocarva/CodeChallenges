# Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

# In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


 

# Example 1:

# Input: rowIndex = 3
# Output: [1,3,3,1]
# Example 2:

# Input: rowIndex = 0
# Output: [1]
# Example 3:

# Input: rowIndex = 1
# Output: [1,1]
 

# Constraints:

# 0 <= rowIndex <= 33

# https://leetcode.com/problems/pascals-triangle-ii/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        pt = []
        for row in range(1, rowIndex+2):
            arr = []
            for i in range(row):
                if i == 0 or i == row - 1:
                    arr.append(1)
                else:
                    if i > 0:
                        val1 = pt[-1][i-1]
                        val2 = pt[-1][i]
                        arr.append(val1+val2)
            pt.append(arr)
        print(pt)
        return pt[rowIndex]