import tensorflow as tf

print("version:"+str(tf.version))
# 创建图，启动图

# 创建一个常量op
x=tf.constant(0.)
y=tf.constant(1.)
for iteration in range(50):
    x=x+y
    y=y/2
print(x.numpy())
