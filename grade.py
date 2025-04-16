#%%
def p(n):
    if n >= 90:
        return 'S', 10
    elif n >= 80:
        return 'A', 9
    elif n >= 70:
        return 'B', 8
    elif n >= 60:
        return 'C', 7
    elif n >= 50:
        return 'D', 6
    elif n >= 40:
        return 'E', 4
    else:
        return 'U', 0
#%% ME
gaa = 92
q1 = 60
q2 = 58
# bonus = 6
f = 82
total = 0.15*gaa + max(0.2*q1 + 0.2*q2+ 0.45*f, 0.5*f + 0.25*max(q1,q2))
if total > 40:
    print(f"ME: {total}- {p(total)}") # ME: 74.3- ('B', 8)

#%% CF
gaa = 72
q1 = 68
q2 = 40
f = 67
total = 0.1*gaa + 0.2*q1 + 0.3*q2 + 0.4*f
if total > 40:
    print(f"CF: {total}- {p(total)}") # CF: 59.599999999999994- ('D', 6)
#%% DLP
gaa = 63
q1 = 72
q2 = 49
n1 = 90.27
n2 = 0
n3 = 80
q3 = 48
total = 0.2*gaa + 0.15*q1 + 0.15*q2 + 0.5*q3 + 0.2*max(n1, n3) + 0.15*min(n1,n3) + 0.1*0
maxi = 0.2 + 0.15 + 0.15 + 0.5 + 0.2 + 0.15 + 0.1
tf = total / maxi 
if tf > 40:
    print(f"DLP: {tf}- {p(tf)}") # DLP: 58.48551724137931- ('D', 6)