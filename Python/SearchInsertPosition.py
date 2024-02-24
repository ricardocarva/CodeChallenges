class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if not nums:
            return 0
        start = 0
        end = len(nums)-1
        mid = 0
        while start <= end:
            mid = (start + end)//2
            if nums[mid] == target:
                return mid
            elif target > nums[mid]:
                start = mid+1
            else:
                end = mid-1
        return start
