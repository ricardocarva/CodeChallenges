# Given the root of a binary tree, return the preorder traversal of its nodes' values.

 

# Example 1:


# Input: root = [1,null,2,3]
# Output: [1,2,3]
# Example 2:

# Input: root = []
# Output: []
# Example 3:

# Input: root = [1]
# Output: [1]
 

# Constraints:

# The number of nodes in the tree is in the range [0, 100].
# -100 <= Node.val <= 100

# https://leetcode.com/problems/binary-tree-preorder-traversal/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # Recursive
        # nodes = []
        # if root is not None:
        #     nodes.append(root.val)
        #     nodes += self.preorderTraversal(root.left)
        #     nodes += self.preorderTraversal(root.right)
        # return nodes

        # Iterative
        nodes = []
        stack = []
        stack.append(root)
        curr = None
        while stack:
            curr = stack.pop()
            if curr is not None:
                nodes.append(curr.val)
                if curr.right is not None:
                    stack.append(curr.right)
                if curr.left is not None:
                    stack.append(curr.left)
        return nodes
