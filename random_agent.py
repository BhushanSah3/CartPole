import gym, cv2
import tensorflow as tf

#tf can use gpus to accelerate
#https://gymnasium.farama.org/environments/classic_control/cart_pole/
env = gym.make("CartPole-v1")

for episode in range(5):
    done = False
    state = env.reset()

    while not done:
        frame = env.render(mode= "rgb_array")  #this will return numpy array
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        #just converts into numpy
        state , reward, done, _ = env.step(action)
env.close()