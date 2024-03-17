# Given an integer numRows, return the first numRows of Pascal's triangle.

# In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


 

# Example 1:

# Input: numRows = 5
# Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
# Example 2:

# Input: numRows = 1
# Output: [[1]]

# https://leetcode.com/problems/pascals-triangle/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        pt = []
        for row in range(1, numRows+1):
            arr = []
            for i in range(0, row):
                if i == 0 or i == row-1:
                    arr.append(1)
                else:
                    if row > 1:
                        v1 = pt[-1][i-1]
                        v2 = pt[-1][i]
                        arr.append(v1 + v2)
            pt.append(arr)
        return pt