import random
import os.path as op
def printIntro():
    print("这是一个比赛程序！")

def getInputs():
    n=eval(input("请输入场数："))
    probA = eval(input("请输入球员A的能力值："))
    probB = eval(input("请输入球员B的能力值："))
    return probA,probB,n


def simOneGame(probA,probB,winsA,winsB):
    a = random.randint(0, probA)
    b = random.randint(0, probB)
    if (a >= b):
        winsA += 1
    elif (a < b):
        winsB += 1
    return  winsA,winsB


def gameIsOver(winsA,winsB,n,i):
    if winsA > winsB and winsA > (winsB + n - i - 1):
        return True
    elif winsA < winsB and winsB > (winsA + n - i - 1):
        return True


def simNGames(n, probA, probB):
    winsA, winsB = 0, 0
    for i in range(n+1):
        winsA,winsB=simOneGame(probA,probB,winsA,winsB)
        if gameIsOver(winsA,winsB,n,i):
            break
    print("进行了{}局游戏".format(i+1))
    return winsA, winsB


def printLastInf(winsA, winsB):
    print("球员A赢了{}局，球员B赢了{}局。".format(winsA, winsB))
    if winsA > winsB:
        result = "A赢了"
    elif winsA < winsB:
        result = "B赢了"
    else:
        result = "平局"
    print(result)


def main():
    printIntro()
    probA,probB,n=getInputs()
    winsA,winsB=simNGames(n,probA,probB)
    printLastInf(winsA,winsB)

while True:
    main()
    url=op.normpath("py1.png")
    print(url)
