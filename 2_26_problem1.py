class Solution:
    def isPalindrome(self, s: str) -> bool:
        temp = ""
        for letter in s:
            if letter.isalnum():
                temp += letter.lower()
        print(temp)
        return temp == temp[::-1]


sol = Solution()

print(sol.isPalindrome("0P"))