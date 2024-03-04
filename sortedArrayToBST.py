# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helperBST(self, nums: List[int], start: int, end: int) -> Optional[TreeNode]:
        if start > end:
            return None
        mid = (start + end)//2
        curr = TreeNode(nums[mid])
        curr.left = self.helperBST(nums, start, mid-1)
        curr.right = self.helperBST(nums, mid+1, end)
        return curr
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        size = len(nums)
        if size == 0:
            return None
        else:
            root = self.helperBST(nums, 0, size-1)
            return root


