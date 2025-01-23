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
#%% LLM
gaa = 73
q1 = 35
q2 = 82
bonus = 6
f = 72
total = 0.1*gaa + 0.4*f + 0.25*q1 + 0.25*q2
if total > 40:
    print(f"LLM + bonus {bonus}: {total}- {p(total)}") # C-7
# print(f"LLM: {total- p(total)}")
# %%
#%% SE
gaa = 84
q2 = 82
f = 65
gp1 = 95
gp2 = 99
pp = 8
cp = 2
total = 0.05*gaa + 0.4*f + 0.2*q2 + 0.1*gp1 + 0.1*gp2 +0.1*pp + 0.05*cp
print(f"SE: {total}- {p(total)}") # C-7
# %% C
w = [80.00,100.00,86.00,94.00,95.00,95.00,95.00,100.00,100.00,0,0]
gaa = sum(sorted(w)[-10:])/10
gaap = 100
q1 = 72
oppe1 = 100
oppe2 = 8
f = 78
t = 0.05*gaa + 0.1* gaap + 0.15 * q1 + 0.2*oppe1 + 0.2*oppe2 + 0.3*f
print(f"C: {t}- {p(t)}") # B-8
# %% CSD
w =[100.00,100.00,100.00,40.00,11.00,80.00,100.00,0,0,0]
gaa = sum(w)/10
f = 82
q1 = 44
q2 = 70
t = 0.1*gaa + 0.4*f + 0.2*q1 + 0.3*q2
print(f"CSD: {t}- {p(t)}") # C-7