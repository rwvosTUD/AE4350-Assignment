import tensorflow as tf 
#from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque
import math
import os


class Actor:
    '''
    Class describing the actor model for the DRL portfolio management system
    This script is part of the assignment for the 
    course AE4350-Bio_inspiredIntelligenceAndLearning
    
    Created on Thu May 12 13:22:50 2022
    @author: Reinier Vos, 4663160-TUD
    '''
    def __init__(self, state_size, action_size):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        
        net = tf.keras.layers.Dense(units=16,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(states)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

        actions = tf.keras.layers.Dense(units=self.action_size, activation='softmax', name = 'actions')(net)
        self.model = tf.keras.models.Model(inputs=states, outputs=actions)
        self.optimizer = tf.keras.optimizers.Adam(lr=.00001)
        
        
        
    def train_fn(self, states, action_gradients):
        #action_gradients = tf.keras.layers.Input(shape=(self.action_size,))
        #loss = K.mean(-action_gradients * actions)
        states = tf.convert_to_tensor(states)
        action_gradients = tf.convert_to_tensor(action_gradients)
        params = self.model.trainable_weights
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(states)
            tape.watch(action_gradients)
            actions = self.model(states, training = True)
            
            actions_log = tf.math.log(actions)
            loss = tf.math.reduce_sum(-action_gradients * actions_log)
        
            #loss = tf.math.reduce_mean(-action_gradients * actions)
        
        grads = tape.gradient(loss, params)
        grads_and_vars = list(zip(grads, params))
        self.optimizer._assert_valid_dtypes([
            v for g, v in grads_and_vars
            if g is not None and v.dtype != tf.resource
        ])
        self.optimizer.apply_gradients(grads_and_vars)
        '''
        actions = self.model(states, training = True)
        loss = tf.math.reduce_mean(-action_gradients * actions)
        updates_op = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        '''
        # updates should already be applied
        
        #self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
        #    outputs=[],
        #    updates=updates_op)
        '''
        # original function in documentation
        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)
            grads_and_vars = list(zip(grads, params))
            self._assert_valid_dtypes([
                v for g, v in grads_and_vars
                if g is not None and v.dtype != tf.resource
            ])
            return [self.apply_gradients(grads_and_vars)]
        '''

#%%
class Critic:
    '''
    Class describing the critic model for the DRL portfolio management system
    This script is part of the assignment for the 
    course AE4350-Bio_inspiredIntelligenceAndLearning
    
    Created on Thu May 12 13:22:50 2022
    @author: Reinier Vos, 4663160-TUD
    '''
    def __init__(self, state_size, action_size):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):
        # define inputs
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')
        
        # 
        net_states = tf.keras.layers.Dense(units=16,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(states)
        net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(net_states)
        
        # 
        net_actions = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(actions)
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)
        
        Q_values = tf.keras.layers.Dense(units=1, name='q_values',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)
        
        self.model = tf.keras.models.Model(inputs=[states, actions],outputs=Q_values)
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse') #experimental_run_tf_function=False)
        
    def get_action_gradients(self, states,actions):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(actions)
            Q_values = self.model([states, actions], training= False) # notice training = false
            #print(type(Q_values))
        #Q_values = self.model([states, actions], training = False)
        #action_gradients = tf.gradients(Q_values, actions)
        action_gradients = tape.gradient(Q_values, actions)
        #self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],outputs=action_gradients)
        #return action_gradients.numpy()
        return action_gradients
        
        
#%%
class ReplayBuffer:
    #Fixed sized buffer to stay experience tuples
    def __init__(self, buffer_size, batch_size):
        #Initialize a replay buffer object.
        #parameters
        #buffer_size: maximum size of buffer. Batch size: size of each batch
        self.memory = deque(maxlen = buffer_size) #memory size of replay buffer
        self.batch_size = batch_size #Training batch size for Neural nets
        self.experience = namedtuple("Experience", field_names = ["state","action", "reward", "next_state", "done"]) #Tuple containing experienced replay

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size = 32):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

#%%
class Agent:
    '''
    Class describing the agent model for the DRL portfolio management system
    This script is part of the assignment for the 
    course AE4350-Bio_inspiredIntelligenceAndLearning
    
    Created on Thu May 12 13:22:50 2022
    @author: Reinier Vos, 4663160-TUD
    '''    
    def __init__(self, state_size, batch_size, checkpoint_dir: str, is_eval = False):
        self.state_size = state_size #
        self.action_size = 3
        self.buffer_size = 1000000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.inventory = []
        self.is_eval = is_eval
        self.gamma = 0.99
        self.tau = 0.001

        self.checkpoint_dir = "/" + checkpoint_dir
        self.checkpoint_path = os.path.join(os.getcwd(),self.checkpoint_dir)
        # check if directory exists
        if os.path.exists(self.checkpoint_path):
            raise Exception("Checkpoint directory already exists, please adjust")
        else:
            os.mkdir(self.checkpoint_path)
            os.mkdir(os.path.join(self.checkpoint_path,"results"))
        print("Models will be saved to {}".format(self.checkpoint_path))
        
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
    def act(self, state):
        '''
         Returns an action, given a state, using the actor (policy network) and
        the output of the softmax layer of the actor-network, returning the
        probability for each action.

        '''
        options = self.actor_local.model.predict(state)
        self.last_state = state
        if not self.is_eval:
            return choice(range(3), p = options[0])
        return np.argmax(options[0])

    def step(self, action, reward, next_state, done):
        '''
         Returns a stochastic policy, based on the action probabilities in the
        training model and a deterministic action corresponding to the maximum
        probability during testing. There is a set of actions to be carried out by
        the agent at every step of the episode.
        '''
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            #print("==============YES============")
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            self.last_state = next_state
     
    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(states, actions),(-1,self.action_size))
        self.actor_local.train_fn(states, action_gradients)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        self.soft_update(self.critic_local.model, self.critic_target.model)
        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights)
        new_weights = self.tau * local_weights + (1 - self.tau)*target_weights
        target_model.set_weights(new_weights)
        
        
    def save_models(self, episode:int):
        os.path.mkdir(os.path.join(self.checkpoint_path,"e{}".format(episode)))
        self.actor_local.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'actor_local.h5'))
        self.actor_target.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'actor_target.h5'))
        self.critic_local.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'critic_local.h5'))
        self.critic_target.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'critic_target.h5'))
        # TODO; also save (hyper)parameters
        print("Succesfully saved models for episode {}".format(episode))
        
    def load_models(self, checkpoint_dir: str, episode:int):
        checkpoint_path = os.path.join(os.getcwd(),'/',checkpoint_dir)
        self.actor_local.model.load_weights(os.path.join(checkpoint_path, 'actor_local_e{}'.format(episode)))
        self.actor_target.model.load_weights(os.path.join(checkpoint_path, 'actor_target_e{}'.format(episode)))
        self.critic_local.model.load_weights(os.path.join(checkpoint_path, 'critic_local_e{}'.format(episode)))
        self.critic_target.model.load_weights(os.path.join(checkpoint_path, 'critic_target_e{}'.format(episode)))
        # TODO; also load (hyper)parameters
        print("Succesfully loaded models for run {0} and episode {1}".format(checkpoint_dir,episode))
        
        
#%%
def formatPrice(n):
    if n>=0:
        curr = "$"
    else:
        curr = "-$"
    return (curr +"{0:.2f}".format(abs(n)))

def getStockData(key):
    datavec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in reversed(lines[1:]): # reversed because first date is latest
        datavec.append(float(line.split(",")[6])) # close price
    return datavec


def getState(data, t, window):
    if t - window >= -1:
        vec = data[t - window+ 1:t+ 1]
    else:
        vec = -(t-window+1)*[data[0]]+data[0: t + 1]
    scaled_state = []
    for i in range(window - 1):
        scaled_state.append(1/(1 + math.exp(vec[i] - vec[i+1])))
    return np.array([scaled_state])

















