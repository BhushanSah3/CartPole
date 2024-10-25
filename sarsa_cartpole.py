import gym
import tensorflow as tf

from keras import Model, Input
from keras.layers import Dense

env = gym.make("CartPole-v1")

# Q Network
net_input = Input(shape=(4,)) # our observation is the vactor of 4
#forming layers
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)

#why 2  coz we have two actions ???
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

# Parameters
ALPHA = 0.001
EPSILON = 1.0
#why the epsilon is fixed here , that would make it bias so have epsilon dcay

EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)  #default is float 32
    #above was for optimal
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action


for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0
    #bathing heeere coz
    state = tf.convert_to_tensor([env.reset()])

    action = policy(state, EPSILON)
    while not done:
        next_state, reward, done, _ = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        #selecting action
        next_action = policy(next_state, EPSILON)


        target = reward + GAMMA * q_net(next_state)[0][next_action]
        if done:
            target = reward

        with tf.GradientTape() as tape:
            #taking curent value after passing with q network
            current = q_net(state)

    #calculating graddient  and grads is a list so we loop over the list below
        grads = tape.gradient(current, q_net.trainable_weights)
       #takignthe differednce
        delta = target - current[0][action]
        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])

 #updating
        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
    print("Episode:", episode, "Length:", episode_length, "Rewards:", total_reward, "Epsilon:", EPSILON)
    #decaying the epsilon
    EPSILON /= EPSILON_DECAY

q_net.save("sarsa_q_net")

env.close()


# the lenght should be 500  i.e it should be able to keep the cart for 500 , so our agent didnt learn
# so have epsilon decay it would give better results than prevuious 