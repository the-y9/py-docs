#%%
s = []
x = 0
flag = 0
sum = 0
s.append(x)
while len(s)!=0:
    if flag == 0:
        x = x + 1
        s.append(x)
    if x==8:
        flag = 1
    if flag == 1:
        x = s.pop()
        if x%2==1:
            s.pop()
        sum = sum + x
print(sum)