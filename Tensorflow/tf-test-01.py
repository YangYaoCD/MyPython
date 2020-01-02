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
s=tf.constant([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
softmax = tf.nn.softplus(s)
print (softmax)
