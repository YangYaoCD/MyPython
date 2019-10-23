re=open("C:\\Users\\yangyao\\desktop\\讲话.txt","rt")
for i in re:
    i=i.replace("\n","")
    print(i)
# i=re.readline().strip()
# print(i,end="")
# while i!="":
#     i=re.readline()
#     print(i,end="")
#print(re.readlines())
# print(re.readline(5))
re.close()