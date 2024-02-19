# Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:

# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
# Every close bracket has a corresponding open bracket of the same type.
 

# Example 1:

# Input: s = "()"
# Output: true
# Example 2:

# Input: s = "()[]{}"
# Output: true
# Example 3:

# Input: s = "(]"
# Output: false
 

# Constraints:

# 1 <= s.length <= 104
# s consists of parentheses only '()[]{}'.

# https://leetcode.com/problems/valid-parentheses/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False
        stack = []
        dict = {')':'(', ']':'[', '}':'{'}
        for i in s:
            if i in ['(','[', '{']:
                stack.append(i)
            elif i in [')', ']', '}']:
                if len(stack) > 0:
                    if dict[i] == stack[-1]:
                        stack.pop()
                    else: 
                        return False
                else: 
                    return False
            else:
                return False
        if len(stack) > 0:
            return False
        return True