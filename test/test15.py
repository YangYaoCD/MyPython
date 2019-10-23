fo=open("output.txt",'w+')
ls=["中国","法国","美国"]
fo.writelines(ls)
fo.seek(0)
for line in fo:
    print(line)
fo.close()