s={"中国":"cncities","美国":"uscities"}
cncities={"大连","北京","上海","成都"}
print("cncities"in s)
print(s.keys())
print(s.values())
for t in s.values():
    print(t)
print(s.get("中国1","巴基斯坦"))
print(len(s))
print(s.popitem())
print(len(s))