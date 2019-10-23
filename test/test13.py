import jieba
def getText(s):
    try:
        txt=open(s,"r",encoding="gb18030").read()
        for ch in '，。？！/*-+=——、|；’"“”：……（）{}【】·~':
            txt = txt.replace(ch, "")
        txt = jieba.lcut(txt)
        return txt
    except:
        print("输入有误")
flag1=True
while(flag1):
    print("请输入txt文件路径(//):")
    s=input()
    print("请输入正序排列(1)还是倒序排列(2)：")
    n1=input()
    s2=(True if eval(n1)==1 else False)
    flag2 = True
    while(flag2):
        print("请输入需要输出的个数：")
        n2=eval(input())
        hamletTxt=getText(s)
        counts={}
        for word in hamletTxt:
            if len(word)==1:
                continue
            counts[word] = counts.get(word, 0) + 1
        items=list(counts.items())
        items.sort(key=lambda  x:x[1],reverse=s2)
        if n2<len(items):
            for i in range(n2):
                word, count = items[i]
                print("{0:<10}{1:>5}".format(word, count))
            print("输入t退出程序！")
            if (input() == "t"):
                flag1 = False
            flag2 = False
        else:
            print("需要输出的个数出错！")