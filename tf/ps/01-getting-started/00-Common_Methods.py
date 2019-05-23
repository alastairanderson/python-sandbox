#region packages
import tensorflow as tf
#endregion

'''
TODO: Remove this from the Pluralsight folder eventually
      Should have this in an easier to access location
'''

#region get_shape(), dtype, rank, reshape(), cast()
x = tf.placeholder(tf.float32, shape=[None, 784])

print(x.get_shape())
print(x.dtype)
print(tf.rank(x))


values = [1, 2, 3, 4]
result = tf.reshape(values, [2, 2])
print(tf.Session().run(result))
print(result.dtype)

result2 = tf.cast(result, tf.float32)
print(tf.Session().run(result2))
print(result2.dtype)
#endregion


#region Activation functions
'''
https://alexisalulema.com/2017/10/15/activation-functions-in-tensorflow/
'''


#endregion
