# Given a string s, reverse only all the vowels in the string and return it.

# The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower and upper cases, more than once.

 

# Example 1:

# Input: s = "hello"
# Output: "holle"
# Example 2:

# Input: s = "leetcode"
# Output: "leotcede"
 

# Constraints:

# 1 <= s.length <= 3 * 105
# s consist of printable ASCII characters.

# https://leetcode.com/problems/reverse-vowels-of-a-string/?envType=study-plan-v2&envId=leetcode-75

class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o','u']
        left = 0
        right = len(s) - 1 
        result = list(s)
        while left < right:
            if s[left].lower() not in vowels:
                left = left + 1
                continue
            if s[right].lower() not in vowels:
                right = right - 1
                continue
            result[left], result[right] = result[right], result[left]
            left = left + 1
            right = right - 1     
        return ''.join(result)
