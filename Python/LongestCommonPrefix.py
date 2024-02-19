# Write a function to find the longest common prefix string amongst an array of strings.

# If there is no common prefix, return an empty string "".

 

# Example 1:

# Input: strs = ["flower","flow","flight"]
# Output: "fl"
# Example 2:

# Input: strs = ["dog","racecar","car"]
# Output: ""
# Explanation: There is no common prefix among the input strings.
 

# Constraints:

# 1 <= strs.length <= 200
# 0 <= strs[i].length <= 200
# strs[i] consists of only lowercase English letters.

# https://leetcode.com/problems/longest-common-prefix/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        # Sort them
        strs.sort()
        # Assume the first is the prefix as its the shortest
        prefix = strs[0]
        # Loops through the range of the minimum between the assumed prefix and the last value in the array of strings
        for i in range(min(len(prefix),len(strs[-1]))):
            # If not found at the start, it returns -1. Otherwise, it returns 0
            while strs[-1].find(prefix) != 0:
                # Keep shortening the prefix
                prefix = prefix[:-1]        
                #If prefix got empty, there is no common prefix        
                if not prefix:
                    return ""    
        return prefix
