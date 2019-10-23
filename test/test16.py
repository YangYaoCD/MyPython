import turtle

fi=open("C:\\Users\\yangyao\\desktop\\te.txt","rt")
for line in fi:
    S=line.split(",")
    if len(S)==6:
        RGB1 = eval(S[3])
        RGB2 = eval(S[4])
        RGB3 = eval(S[5])
        turtle.pencolor(RGB1, RGB2, RGB3)
        path=eval(S[0])
        ia=eval(S[1])
        angle=eval(S[2])
        if ia==0:
            turtle.left(angle)
        elif ia==1:
            turtle.right(angle)
        else:
            print("错误")
            continue
        turtle.fd(path)
fi.close()
turtle.done()

