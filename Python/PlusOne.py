class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        sum = 0
        num = 0
        order = 1
        for i in range(len(digits)-1, -1, -1):
            num = digits[i] * order
            order  = order * 10
            sum = sum + num
        sum = sum + 1
        print(sum)
        r = []
        while sum > 0:
            r.insert(0, sum % 10)
            sum = sum // 10
        return r
