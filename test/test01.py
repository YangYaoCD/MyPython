b=1
while(b):
    try:
        mass=eval(input("请输入体重(kg)："))
        body = eval(input("请输入身高(m)："))
        BMI=mass/(body**2)
        print("BMI指数为：{}".format(BMI))
        if BMI>=30:
            who1,who2="肥胖","肥胖"
        elif BMI>=28and BMI<30:
            who1,who2="偏胖","肥胖"
        elif BMI>=25 and BMI<28:
            who1, who2 = "偏胖", "偏胖"
        elif BMI>=24 and BMI<25:
            who1, who2 = "正常", "偏胖"
        elif BMI>=18.5 and BMI<24:
            who1, who2 = "正常", "正常"
        else:
            who1, who2 = "偏瘦", "偏瘦"
    except:
        print("输入格式错误")
        b=0
    else:
        print("国际标准：{}\n国内标准：{}".format(who1,who2))
