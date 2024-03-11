class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        result = ""
        while columnNumber > 0:
            # To align with the way we do calculations in base-26, we subtract 1 from the column number so that A maps to 0, B to 1, ..., and Z to 25 within our calculation context
            columnNumber -= 1  # Adjusting for the lack of zero in Excel columns
            # ord gets the int value of a char, and chr gets the ascii value of the int
            result = chr(columnNumber % 26 + ord('A')) + result  # Convert to letter and prepend to result
            columnNumber //= 26  # Move to the next digit
        return result

