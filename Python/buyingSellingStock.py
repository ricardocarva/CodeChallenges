# ou are given an array prices where prices[i] is the price of a given stock on the ith day.

# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

# Example 1:

# Input: prices = [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
# Example 2:

# Input: prices = [7,6,4,3,1]
# Output: 0
# Explanation: In this case, no transactions are done and the max profit = 0.
 

# Constraints:

# 1 <= prices.length <= 105

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/?envType=featured-list&envId=top-amazon-questions?envType=featured-list&envId=top-amazon-questions

class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        #Solution 1 better run time
        maxProfit = 0
        minVal = float('inf')
        for price in prices:
            if price < minVal:
                minVal = price
            if price - minVal > maxProfit:
                maxProfit = price - minVal
        return maxProfit
    
    #Solution 2
        # if len(prices) > 1:
        #     candidateMin = 0
        #     low = 0
        #     high = 1
        #     for i in range(1,len(prices)):
        #         if prices[i] < prices[low]:
        #             if prices[i] < prices[candidateMin]:
        #                 candidateMin = i
        #         if prices[i] - prices[candidateMin] > prices[high] - prices[low]:
        #             high = i
        #             low = candidateMin
        #     return prices[high] - prices[low]
        # else:
        #     return 0