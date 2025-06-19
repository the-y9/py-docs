#%%
class Solution:
    def maxDifference(self, s: str) -> int:
        hmap = {}
        for i in s:
            if i in hmap:
                hmap[i] += 1
            else:
                hmap[i] = 1
        maxodd = float('-inf')
        mineve = float('inf')
        for char, count in hmap:
            print(char, count)
            if (count%2==1) and (count > maxodd):
                maxodd = count
            elif (count%2==0) and (count < mineve):
                mineve = count
        return maxodd - mineve

if __name__ == '__main__':
    s = Solution()
    print(s.maxDifference("aabccba"))  