# 请在...补充一行或多行代码
def judge(n):
    if n%int(n)==0:
        return n
    else:
        return int(n)

def prime(m):
    flag = True
    while (flag):
        j = 0
        for i in range(m-1):
            if (m % (i + 1) == 0):
                j+=1
        m += 1
        if (j == 1):
            flag = False
            return m

while(True):
    n = eval(input())
    n=judge(n)
    for i in range(5):
        n = prime(n)
        if i<4:
            print(n-1, end=",")
        else:
            print(n-1,end="")
