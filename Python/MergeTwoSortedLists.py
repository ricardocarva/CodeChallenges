# ou are given the heads of two sorted linked lists list1 and list2.

# Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

# Return the head of the merged linked list.

 

# Example 1:


# Input: list1 = [1,2,4], list2 = [1,3,4]
# Output: [1,1,2,3,4,4]
# Example 2:

# Input: list1 = [], list2 = []
# Output: []
# Example 3:

# Input: list1 = [], list2 = [0]
# Output: [0]
 

# Constraints:

# The number of nodes in both lists is in the range [0, 50].
# -100 <= Node.val <= 100
# Both list1 and list2 are sorted in non-decreasing order.

# https://leetcode.com/problems/merge-two-sorted-lists/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        mergedList = ListNode()
        tempVal = mergedList
        if list1 == None:
            return list2
        elif list2 == None:
            return list1
        elif list1 == None and list2 == None:
            return []
        while list1 and list2:
            if list1.val <= list2.val:
                tempVal.next = list1
                list1 = list1.next
            else:
                tempVal.next = list2
                list2 = list2.next
            tempVal = tempVal.next
        tempVal.next = list1 if list1 else list2
        return mergedList.next