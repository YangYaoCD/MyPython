sum=0
for i in range(2,100):
    k=0
    for j in range(1,i):
        if i%j==0:
            k+=1
    if k==1:
        sum+=i
print(sum)
for i in (1,2,3):
    print(i)
P=10
P=-P
print(P)