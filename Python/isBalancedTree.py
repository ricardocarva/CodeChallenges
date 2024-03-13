# Given a binary tree, determine if it is 
# height-balanced
# .

 

# Example 1:


# Input: root = [3,9,20,null,null,15,7]
# Output: true
# Example 2:


# Input: root = [1,2,2,3,3,null,null,4,4]
# Output: false
# Example 3:

# Input: root = []
# Output: true
 

# Constraints:

# The number of nodes in the tree is in the range [0, 5000].
# -104 <= Node.val <= 104

# https://leetcode.com/problems/balanced-binary-tree/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root == None:
            return True
        balance = self.getHeight(root)
        if balance != -1:
            return True
        return False
        
    def getHeight(self, root:Optional[TreeNode]) -> int:
        if root == None:
            return 0
        left = self.getHeight(root.left)
        if left == -1:
            return -1
        right = self.getHeight(root.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return - 1
        return max(left, right) + 1