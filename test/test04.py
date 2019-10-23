import time
import turtle

def drawLine(draw):
    turtle.pendown() if draw else turtle.up()
    turtle.fd(40)
    turtle.right(90)
def gap():
    turtle.seth(0)
    turtle.up()
    turtle.fd(60)
def down():
    turtle.penup()
    turtle.seth(270)
    turtle.fd(40)
    turtle.seth(0)
def up():
    turtle.penup()
    turtle.seth(90)
    turtle.fd(40)
    turtle.seth(0)
def drawDigit(digit):
    drawLine(True) if digit in [2,3,4,5,6,8,9] else drawLine(False)
    drawLine(True) if digit in [0,1,3,4,5,6,7,8,9] else drawLine(False)
    drawLine(True) if digit in [0,2,3,5,6,8] else drawLine(False)
    drawLine(True) if digit in [0,2,6,8] else drawLine(False)
    turtle.left(90)
    drawLine(True) if digit in [0,4, 5, 6, 8, 9] else drawLine(False)
    drawLine(True) if digit in [0,2, 3, 5, 6, 7, 8, 9] else drawLine(False)
    drawLine(True) if digit in [0,1,2,3, 4, 7, 8, 9] else drawLine(False)
    turtle.up()
    turtle.left(180)
    turtle.fd(20)
def drawDate(date):
    if date in ['=']:
        down()
        turtle.write("年",font=("Arial",40,"normal"))
        gap()
        up()
        turtle.pencolor("purple")
    elif date in ['+']:
        down()
        turtle.write("月",font=("Arial",40,"normal"))
        gap()
        up()
        turtle.pencolor("blue")
    elif date in ['-']:
        down()
        turtle.write("日",font=("Arial",40,"normal"))
        gap()
        up()
        turtle.pencolor("yellow")
    elif date in ['*']:
        down()
        turtle.write("时",font=("Arial",40,"normal"))
        gap()
        up()
        turtle.pencolor("pink")
    elif date in ['/']:
        down()
        turtle.write("分",font=("Arial",40,"normal"))
        gap()
        up()
        turtle.pencolor("green")
    elif date in ['(']:
        down()
        turtle.write("秒",font=("Arial",40,"normal"))
        up()
        turtle.pencolor("orange")
    else:
        drawDigit(eval(date))
def main():
    turtle.reset()
    turtle.speed(10)
    Date=time.strftime("%Y=%m+%d-%H*%M/%S(",time.localtime())
    turtle.pencolor("red")
    turtle.pensize(5)
    turtle.penup()
    turtle.bk(600)
    try:
        for i in Date:
            drawDate(i)
    except:
        pass
    turtle.hideturtle()
    #turtle.done()
turtle.setup(width=1400,height=600, startx=100, starty=100)
turtle.hideturtle()
while True:
    main()
    time.sleep(2)

