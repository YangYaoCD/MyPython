import jieba
def getText():
    txt=open("C:\\Users\\yangyao\\Desktop\\讲话.txt","r",encoding="gb18030").read()
    for ch in '，。？！/*-+=——、|；’"“”：……（）{}【】·~':
        txt = txt.replace(ch, "")
    txt=jieba.lcut(txt)
    return txt
hamletTxt=getText()
excludes={"将军","却说","不敢","魏兵","二人","不可","陛下","不知","人马","主公","荆州","不能","如此","商议","如何","军士","左右","军马","引兵","次日","大喜","天下","东吴","于是","今日"}
counts={}
for word in hamletTxt:
    if len(word)==1 or word in excludes:
        continue
    elif word=="玄德曰" or word=="玄德":
        word="刘备"
    elif word=="孔明曰" or word=="诸葛亮":
        word="孔明"
    elif word=="都督":
        word="周瑜"
    elif word=="丞相" or word=="孟德":
        word="曹操"
    elif word=="关公":
        word="关羽"
    counts[word] = counts.get(word, 0) + 1
items=list(counts.items())
items.sort(key=lambda  x:x[1],reverse=True)
for i in range(10):
    word,count=items[i]
    print("{0:<10}{1:>5}".format(word,count))