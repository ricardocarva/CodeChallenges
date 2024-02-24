

# Given two binary strings a and b, return their sum as a binary string.

 

# Example 1:

# Input: a = "11", b = "1"
# Output: "100"
# Example 2:

# Input: a = "1010", b = "1011"
# Output: "10101"
 

# Constraints:

# 1 <= a.length, b.length <= 104
# a and b consist only of '0' or '1' characters.
# Each string does not contain leading zeros except for the zero itself.

# https://leetcode.com/problems/add-binary/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        r = ""
        val = 0
        remainder = 0
        ia = len(a)-1
        ib = len(b)-1
        while ia >= 0 or ib >= 0:
            if ia >= 0 and ib >=0 :
                val = int(a[ia]) + int(b[ib]) + remainder
                ia = ia - 1
                ib = ib - 1
            elif ia == -1 and ib >= 0:
                val = int(b[ib]) + remainder
                ib = ib - 1
            elif ia >= 0 and ib == -1:
                val = int(a[ia]) + remainder
                ia = ia - 1
            if val > 2:
                remainder = 1
                r ="1" + r
            elif val > 1:
                remainder = 1
                r ="0" + r
            else:
                remainder = 0
                r = str(val) + r
        if remainder == 1:
            r = str(remainder) + r 
        return r

