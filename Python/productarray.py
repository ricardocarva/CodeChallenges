class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        answer = [1] * length
        # Left products
        left_product = 1
        for i in range(1, length):
            left_product *= nums[i - 1]
            answer[i] *= left_product
        right_product = 1
        for i in reversed(range(length - 1)):
            right_product *= nums[i + 1]
            answer[i] *= right_product
        return answer


