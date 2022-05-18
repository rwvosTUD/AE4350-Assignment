import tensorflow as tf 
#from tensorflow.keras.optimizers import Adam
from keras import backend as K
import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque
import math
import os
import pandas as pd
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import plotly.io as pio
import copy 
import json

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
        
        net = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(states)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dense(units=64,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

        actions = tf.keras.layers.Dense(units=self.action_size, activation='softmax', name = 'actions')(net)
        self.model = tf.keras.models.Model(inputs=states, outputs=actions)
        self.optimizer = tf.keras.optimizers.Adam(lr=.00001)
        
    @tf.autograph.experimental.do_not_convert
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
            
            #actions_log = tf.math.log(actions)
            #loss = tf.math.reduce_sum(-action_gradients * actions_log)
        
            loss = tf.math.reduce_mean(-action_gradients * actions)
        
        grads = tape.gradient(loss, params)
        grads_and_vars = list(zip(grads, params))
        '''
        # check, but does not seem required
        self.optimizer._assert_valid_dtypes([
            v for g, v in grads_and_vars
            if g is not None and v.dtype != tf.resource
        ])
        '''
        self.optimizer.apply_gradients(grads_and_vars)
        return loss.numpy()
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
        net_states = tf.keras.layers.Dense(units=32,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(states)
        net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=64,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(net_states)
        
        # 
        net_actions = tf.keras.layers.Dense(units=64,kernel_regularizer=tf.keras.regularizers.l2(1e-6))(actions)
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)
        
        Q_values = tf.keras.layers.Dense(units=1, name='q_values',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)
        
        self.model = tf.keras.models.Model(inputs=[states, actions],outputs=Q_values)
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse') #experimental_run_tf_function=False)
        
    @tf.autograph.experimental.do_not_convert
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
    '''
    Replay buffer to store transitions, optimized compared to original
    '''
    def __init__(self,state_size, action_size, buffer_size, batch_size):

        self.memory_size = buffer_size
        self.batch_size = batch_size #Training batch size for Neural nets
        self.state_size = state_size
        self.action_size = action_size
        self.memory_counter = 0 
        
        self.memory_state = np.zeros((self.memory_size,state_size))
        self.memory_nextState = np.zeros((self.memory_size,state_size))
        self.memory_action = np.zeros((self.memory_size,action_size))
        self.memory_reward = np.zeros((self.memory_size))
        self.memory_dones = np.zeros((self.memory_size), dtype=np.bool)


    def add(self, state, action, reward, next_state, done):
        ind = self.memory_counter % self.memory_size
        
        self.memory_state[ind] =  state
        self.memory_nextState[ind] = next_state
        self.memory_action[ind] = action
        self.memory_reward[ind] = reward
        self.memory_dones[ind] = done
        
        self.memory_counter += 1

    def sample(self, batch_size = 32):
        max_choice = min(self.memory_size,self.memory_counter)
        batch = np.random.choice(max_choice, batch_size)
        
        states = self.memory_state[batch].astype(np.float32).reshape(-1,self.state_size)
        nextStates = self.memory_nextState[batch].astype(np.float32).reshape(-1,self.state_size)
        actions = self.memory_action[batch].astype(np.float32).reshape(-1,self.action_size) 
        rewards = self.memory_reward[batch].astype(np.float32).reshape(-1,1)
        dones = self.memory_dones[batch].astype(np.float32).reshape(-1,1)
        
        return [states, actions, rewards, nextStates, dones]

    def __len__(self):
        # return current length
        return self.memory_counter

#%%
class Agent:
    '''
    Class describing the agent model for the DRL portfolio management system
    This script is part of the assignment for the 
    course AE4350-Bio_inspiredIntelligenceAndLearning
    
    Created on Thu May 12 13:22:50 2022
    @author: Reinier Vos, 4663160-TUD
    '''    
    def __init__(self, state_size, batch_size, start_price, n_budget, is_terminal_threshold, 
                 checkpoint_dir: str,rewardType = 1, data_extraWindow = 1, is_eval = False):
        self.state_size = state_size+3 # +3 for the balance, inventory and portfolio state
        self.action_size = 3
        self.buffer_size = 1000000
        self.batch_size = batch_size
        
        self.n_budget = n_budget # number of stocks in budget
        self.budget = self.n_budget*start_price
        self.balance = 0. # balance parameter will be adapted
        
        assert is_terminal_threshold <= 1 and is_terminal_threshold >= 0, "is_terminal_threshold should be in %"
        self.is_terminal_threshold = is_terminal_threshold*self.budget
        self.is_eval = is_eval
        self.gamma = 0.99
        self.tau = 0.002 #0.001
        self.actor_local_loss = 1.
        self.rewardType = rewardType
        self.data_extraWindow = data_extraWindow
        self.attr_dct = copy.deepcopy(self.__dict__) # setup attirbute dictionary thusfar
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(os.getcwd(),self.checkpoint_dir)
        # check if directory exists
        if os.path.exists(self.checkpoint_path):
            dirList = os.listdir(self.checkpoint_path)
            if len(dirList) > 2: # >2 because results folder & param file (end of __init) is probably created
                raise Exception("Checkpoint directory already exists and is not empty, please adjust")
        else:
            os.mkdir(self.checkpoint_path)
            os.mkdir(os.path.join(self.checkpoint_path,"results"))
        print("Models will be saved to {}".format(self.checkpoint_path))
        
        # initialize models and rest of system
        self.reset(start_price)
        self.set_rewardtype(self.rewardType)
        self.memory = ReplayBuffer(self.state_size, self.action_size, self.buffer_size, self.batch_size)
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        self.save_attributes() # save all simple attibutes
        
    def setup_validation(self):
        os.mkdir(os.path.join(self.checkpoint_path,"validation"))
        
    
    def update_balance(self, change: float):
        self.balance += change
    
    def update_inventory(self,cur_price: float):
        self.inventory_value = len(self.inventory)*cur_price # current value of sum of stocks
        
    def check_threshold(self):
        is_terminal = False
        a = 2
        if a ==3: # wont trigger
        #if (self.balance+self.inventory_value) < self.is_terminal_threshold or self.balance < 0:
            # terminal state reached, raise flag
            is_terminal = True
            print("=== Terminating trial === ")
        return is_terminal 
        
    def reset(self, start_price):
        self.balance = 0.
        self.inventory = [start_price]*self.n_budget
        self.inventory_value = start_price*self.n_budget
        self.inventory_ma = []
        
    '''
    ======================== MODEL RELATED ===============================
    ''' 
        
    def act(self, state):
        '''
         Returns an action, given a state, using the actor (policy network) and
        the output of the softmax layer of the actor-network, returning the
        probability for each action.

        '''
        actions_prob = self.actor_local.model.predict(state)
        self.last_state = state
        if not self.is_eval:
            action = choice(range(3), p = actions_prob[0])
        else:
            action = np.argmax(actions_prob[0])
        return action, actions_prob

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
        return self.actor_local_loss
    
    @tf.autograph.experimental.do_not_convert
    def learn(self, experiences):
        '''
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        '''
        states, actions, rewards, next_states, dones = experiences 
        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(states, actions),(-1,self.action_size))
        self.actor_local_loss = self.actor_local.train_fn(states, action_gradients)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        self.soft_update(self.critic_local.model, self.critic_target.model)
        
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights)
        new_weights = self.tau * local_weights + (1 - self.tau)*target_weights
        target_model.set_weights(new_weights)
        
    '''
    ======================== SAVING/LOADING ===============================
    '''  
    def save_models(self, episode:int):
        os.mkdir(os.path.join(self.checkpoint_path,"e{}".format(episode)))
        self.actor_local.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'actor_local.h5'))
        self.actor_target.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'actor_target.h5'))
        self.critic_local.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'critic_local.h5'))
        self.critic_target.model.save_weights(os.path.join(self.checkpoint_path,"e{}".format(episode), 'critic_target.h5'))
        # TODO; also save (hyper)parameters
        print("Succesfully saved models for episode {}".format(episode))
        
    def load_models(self, checkpoint_dir: str, episode:int):
        checkpoint_path = os.path.join(os.getcwd(),checkpoint_dir)
        self.actor_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_local.h5'))
        self.actor_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_target.h5'))
        self.critic_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_local.h5'))
        self.critic_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_target.h5'))
        # TODO; also load (hyper)parameters
        print("Succesfully loaded models from folder {0} and episode {1}".format(checkpoint_dir,episode))
        

    def save_attributes(self):
        with open('./{0}/agent_parameters.json'.format(self.checkpoint_dir), 'w') as fp:
            json.dump(self.attr_dct, fp)
        print("Succesfully saved model parameters to folder {0}".format(self.checkpoint_dir))

    def load_attributes(self,checkpoint_dir: str):
        with open('./{0}/agent_parameters.json'.format(checkpoint_dir), 'r') as fp:
            self.attr_dct = json.load(fp)
        for key in self.attr_dct:
            setattr(self, key, self.attr_dct[key])
        print("Succesfully loaded model parameters from folder {0}".format(checkpoint_dir))
        
        
    '''
    =========================== REWARDS ======================================
    LINKS:
        https://ai.stackexchange.com/questions/22851/what-are-some-best-practices-when-trying-to-design-a-reward-function
    '''
    def set_rewardtype(self, rewardType: int):
        if rewardType == 0:
            msg = "basic reward function of format max(profit,0)"
            self.get_reward = self._reward_type0
        elif rewardType == 1:
            msg = "unclamped basic reward i.e. positive and negative profits possible"
            self.get_reward = self._reward_type1
        elif rewardType == 2:
            msg = "reward neutralizing unclosed positions and rewarding profits"
            self.get_reward = self._reward_type2
        elif rewardType == 3:
            msg = "terminal reward and penalty for buy hold, no intermediary"
            self.get_reward = self._reward_type3
        elif rewardType == 4:
            msg = "Terminal reward and intermediary rewards"
            self.get_reward = self._reward_type4
        elif rewardType == 5:
            msg = "Only intermediate rewards no terminal reward"
            self.get_reward = self._reward_type5
        print("Reward function description: "+msg)
    
    def _reward_type0(self, profit: float, 
                      util_lst: list, last: bool):
        '''
        naive function returning the clamped profit of a sale as reward
        '''
        reward = max(profit,0)
        return reward
    
    def _reward_type1(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        naive function returning the unclamped profit of a sale as reward
        '''
        reward = profit
        return reward

    def _reward_type2(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        reward neutralizing unclosed positions and rewarding profits
        '''
        price = util_lst[0]
        reward = profit # notice, unclamped!
        if last:
            # close positions based on current price
            closed = np.sum(np.array(agent.inventory)-price)
            reward = profit + closed
        return reward
    
    def _reward_type3(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        Terminal reward only, no intermediate rewards
        '''
        pt = util_lst[0]
        reward = 0
        if last:
            reward = agent.balance+agent.inventory_value - agent.n_budget*pt
            if reward > 0:
                reward = reward*2  # to reward a bit more, this might overfit though!
            elif reward == 0:
                reward = -1000 # to avoid buy and hold itself
        return reward

    def _reward_type4(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        Terminal reward and intermediary rewards
        '''
        pt = util_lst[0] 
        pt1 = util_lst[1]
        ptn = util_lst[2]
        at = util_lst[3]
        if at == 2:
            # a sale should be -1 according to docs
            at = -1
        
        reward = (1+at*(pt-pt1)/pt1)*(pt1/ptn)
        
        if last:
            reward = agent.balance+agent.inventory_value - agent.n_budget*pt
            if reward > 0:
                reward = -1000 # to avoid buy and hold itself
        return reward
    
    def _reward_type5(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        Only intermediate rewards no terminal reward
        EXPERIMENTAL; also some small intermediary rewards based on 
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7384672/
        '''
        pt = util_lst[0] 
        pt1 = util_lst[1]
        ptn = util_lst[2]
        at = util_lst[3]
        if at == 2:
            # a sale should be -1 according to docs
            at = -1
        
        reward = (1+at*(pt-pt1)/pt1)*(pt1/ptn)

        '''
        todo; maybe add the final reward, i.e. if last statement
        '''
        return reward

#%%
def formatPrice(n):
    if n>=0:
        curr = "+$"
    else:
        curr = "-$"
    return (curr +"{0:.2f}".format(abs(n)))

def getData(key: str, window: int, colab = False) -> np.array:
    if colab:
        data = pd.read_csv("AE4350_Assignment/data/" + key + ".csv")
    else:
        data = pd.read_csv("data/" + key + ".csv")
    data = data["Close*"].to_numpy()
    data = data[::-1] # reverse because first date is latest
    # moving average 
    '''
    data_maAdjusted = np.convolve(data, np.ones(window), 'valid')/window  
    len_mismatch = len(data)-len(data_maAdjusted)
    data_maAdjusted = np.pad(data_maAdjusted,(len_mismatch,0), "constant",constant_values=(data[:len_mismatch],0))
    data_maAdjusted = data-data_maAdjusted
    # cutoff irrelevant ma part
    data_maAdjusted = data_maAdjusted[window:]
    data = data[window:]
    data_maAdjusted = data_maAdjusted[window:]
    return data, data_maAdjusted
    '''
    # window is cutoff window
    data = data[window:]
    predata = data[:window]
    return data, predata


def getState(agent, data: np.array, t: int, window: int, use_rtn = True) -> np.array:
    if t - window >= -1:
        state = data[t - window+ 1:t+ 1]
    else:
        state = np.pad(data[0:t+1],(-(t-window+1),0),'constant',constant_values=(data[0],np.nan))
    # scaling of state (standardization of input)
    if use_rtn:
        # use returns to scale, NOTE that we loose one data entry this way
        state = state[1:]-state[:-1] # returns
    else:
        state = state[1:] # consistent length
        
    state = 1/(1+np.exp(state))
    
    pvalue = agent.inventory_value+agent.balance # portfolio value
    pvalue = (pvalue-agent.n_budget*data[t])/(agent.n_budget*data[t]) # standardize
    append = [agent.balance/agent.budget,agent.inventory_value/agent.budget,pvalue]
    state = np.append(state,append) # TODO, maybe clip these to max of 1?
    
    state = np.expand_dims(state,axis = 0)
    return state


    












