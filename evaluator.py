import gym, cv2
import tensorflow as tf
from keras.models import load_model
  #to load our q -neetwork

env = gym.make("CartPole-v1")
q_net = load_model("sarsa_q_net")
#q_net = load_model("q_learning_q_net")


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action


for episode in range(5):
    done = False
    state = tf.convert_to_tensor([env.reset()])
    while not done:
        frame = env.render(mode="rgb_array")
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        action = policy(state)
        state, reward, done, _ = env.step(action.numpy())
        state = tf.convert_to_tensor([state])
env.close()