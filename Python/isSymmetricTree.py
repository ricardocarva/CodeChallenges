# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

 

# Example 1:


# Input: root = [1,2,2,3,4,4,3]
# Output: true
# Example 2:


# Input: root = [1,2,2,null,3,null,3]
# Output: false
 

# Constraints:

# The number of nodes in the tree is in the range [1, 1000].
# -100 <= Node.val <= 100
 

# Follow up: Could you solve it both recursively and iteratively?
# https://leetcode.com/problems/symmetric-tree/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
            if root is None:
                return False
            q = []
            q.append(root.left)
            q.append(root.right)
            while len(q) > 0:
                l = q.pop(0)
                r = q.pop(0)
                if l is None and r is None:
                    continue
                elif l is None or r is None or l.val != r.val:
                    return False
                else:
                    q.append(l.left)
                    q.append(r.right)
                    q.append(l.right)
                    q.append(r.left)
            return True
