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
    def __init__(self, stateTS_size, stateUT_size, action_size, model_hyper):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.stateTS_size = stateTS_size
        self.stateUT_size = stateUT_size
        self.action_size = action_size
        self.model_hyper = model_hyper
        self.regularizer = model_hyper["actor_regularizer"]
        self.use_batchNorm = model_hyper["use_batchNorm"]
        self.build_model()
        

    def build_model(self):
        
        
        '''
        states_ts = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        
        net = tf.keras.layers.Dense(units=self.hidden_units[0][0],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(states)
        if self.use_batchNorm:
            net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dense(units=self.hidden_units[0][1],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        if self.use_batchNorm:
            net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        # additional layers

        net = tf.keras.layers.Dense(units=self.hidden_units[0][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        if self.use_batchNorm:
            net = tf.keras.layers.BatchNormalization()(net)
        
        net = tf.keras.layers.Activation("relu")(net)
        '''
        # Define inputs
        states_ts = tf.keras.layers.Input(shape=(self.stateTS_size,1,), name='states_TS')
        states_ut = tf.keras.layers.Input(shape=(self.stateUT_size,), name='states_UT')
        net_ts = states_ts
        net_ut = states_ut
        
        # timeseries track
        ts_conv1dLayers = self.model_hyper["actor_ts_convLayers"]
        ts_denseLayers = self.model_hyper["actor_ts_dLayers"]
        ts_maxPoolsize = self.model_hyper["actor_ts_maxPoolsize"]
        ts_maxPoolstrides = self.model_hyper["actor_ts_maxPoolstrides"]
        
        for layerUnit in ts_conv1dLayers:
            filters = layerUnit[0]
            kernel_size = layerUnit[1]
            net_ts = tf.keras.layers.Conv1D(filters, kernel_size, 
                                            activation = "relu")(net_ts)
            if ts_maxPoolsize != 0:
                net_ts = tf.keras.layers.MaxPool1D(pool_size=ts_maxPoolsize, 
                                                   strides = ts_maxPoolstrides)(net_ts)
            if self.use_batchNorm:
                net_ts = tf.keras.layers.BatchNormalization()(net_ts)
                
            
        net_ts = tf.keras.layers.Flatten()(net_ts) # flatten before dense
        for layerUnit in ts_denseLayers:
            net_ts = tf.keras.layers.Dense(units = layerUnit, 
                                           activation = 'relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_ts)
            
        
        
        # utilities track
        util_denseLayers = self.model_hyper["actor_util_dLayers"]
        for layerUnit in util_denseLayers:
            net_ut = tf.keras.layers.Dense(units = layerUnit, activation = 'relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_ut)
        
        
        # combined track 
        net = tf.keras.layers.Concatenate()([net_ts,net_ut])
        
        comb_denselayers = self.model_hyper["actor_comb_dLayers"]
        for layerUnit in comb_denselayers:
            net = tf.keras.layers.Dense(units = layerUnit, activation = 'relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        '''
        TODO REGULARIZERS OR DROPOUT?
        '''
        
        # output layer
        actions = tf.keras.layers.Dense(units=self.action_size, activation='softmax', name = 'actions')(net)
        
        # setup
        self.model = tf.keras.models.Model(inputs=[states_ts,states_ut], outputs=actions)
        self.optimizer = tf.keras.optimizers.Adam(lr=.00001)
        
    @tf.autograph.experimental.do_not_convert
    def train_fn(self, states_ts, states_ut, action_gradients):
        #action_gradients = tf.keras.layers.Input(shape=(self.action_size,))
        #loss = K.mean(-action_gradients * actions)        
        states_ts = tf.convert_to_tensor(states_ts)
        states_ut = tf.convert_to_tensor(states_ut)
        action_gradients = tf.convert_to_tensor(action_gradients)
        params = self.model.trainable_weights
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(states_ts)
            tape.watch(states_ut)
            tape.watch(action_gradients)
            actions = self.model([states_ts,states_ut], training = True)
            
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
    def __init__(self, stateTS_size, stateUT_size, action_size, model_hyper):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.stateTS_size = stateTS_size
        self.stateUT_size = stateUT_size
        self.action_size = action_size
        self.model_hyper = model_hyper
        self.regularizer = model_hyper["critic_regularizer"]
        self.use_batchNorm = model_hyper["use_batchNorm"]
        self.build_model()

    def build_model(self):        
        '''
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')
        
        # 
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][0],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(states)
        if self.use_batchNorm:
            net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][1],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_states)

        # == additional layer
        if self.use_batchNorm:
            net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_states)
        # ==
        
        # 
        net_actions = tf.keras.layers.Dense(units=self.hidden_units[1][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(actions)
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)
        
        '''
        
        # Define inputs
        states_ts = tf.keras.layers.Input(shape=(self.stateTS_size,1,), name='states_TS')
        states_ut = tf.keras.layers.Input(shape=(self.stateUT_size,), name='states_UT')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')
        net_ts = states_ts
        net_ut = states_ut
        net_actions = actions
        
        # timeseries track
        ts_conv1dLayers = self.model_hyper["critic_ts_convLayers"]
        ts_denseLayers = self.model_hyper["critic_ts_dLayers"]
        ts_maxPoolsize = self.model_hyper["critic_ts_maxPoolsize"]
        ts_maxPoolstrides = self.model_hyper["critic_ts_maxPoolstrides"]
        
        for layerUnit in ts_conv1dLayers:
            filters = layerUnit[0]
            kernel_size = layerUnit[1]
            net_ts = tf.keras.layers.Conv1D(filters, kernel_size, 
                                            activation = "relu")(net_ts)
            if ts_maxPoolsize != 0:
                net_ts = tf.keras.layers.MaxPool1D(pool_size=ts_maxPoolsize, 
                                                   strides = ts_maxPoolstrides)(net_ts)
            if self.use_batchNorm:
                net_ts = tf.keras.layers.BatchNormalization()(net_ts)
                
            
        net_ts = tf.keras.layers.Flatten()(net_ts) # flatten before dense
        for layerUnit in ts_denseLayers:
            net_ts = tf.keras.layers.Dense(units = layerUnit, 
                                           activation = 'relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_ts)
            
        
        
        # utilities track
        util_denseLayers = self.model_hyper["critic_util_dLayers"]
        for layerUnit in util_denseLayers:
            net_ut = tf.keras.layers.Dense(units = layerUnit, activation = 'relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_ut)
        
        
        # combined track 
        net_states = tf.keras.layers.Concatenate()([net_ts,net_ut])
        
        comb_denselayers = self.model_hyper["critic_comb_dLayers"]
        for layerUnit in comb_denselayers: 
            net_states = tf.keras.layers.Dense(units = layerUnit, activation = 'relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_states)

        # action related layers before addition
        action_denselayers = self.model_hyper["critic_action_dLayers"]
        for layerUnit in action_denselayers:
            net_actions = tf.keras.layers.Dense(units=layerUnit, activation = 'relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_actions)
        
        # adding of states and actions & final layers        
        final_denselayers = self.model_hyper["critic_final_dLayers"]
        net = tf.keras.layers.Concatenate()([net_states, net_actions])  # ADD LAYERS
        for layerUnit in final_denselayers:
            net = tf.keras.layers.Dense(units=layerUnit, activation = 'relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        
        
        
        Q_values = tf.keras.layers.Dense(units=1, name='q_values',kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)
        
        '''
        TODO REGULARIZERS OR DROPOUT?
        '''
        
        self.model = tf.keras.models.Model(inputs=[states_ts, states_ut, actions],outputs=Q_values)
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse') #experimental_run_tf_function=False)
        
    @tf.autograph.experimental.do_not_convert
    def get_action_gradients(self, states_ts, states_ut, actions):
        states_ts = tf.convert_to_tensor(states_ts)
        states_ut = tf.convert_to_tensor(states_ut)
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(actions)
            Q_values = self.model([states_ts,states_ut, actions], training= False) # notice training = false
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
        
        self.memory_state = np.zeros((self.memory_size,state_size,1))
        self.memory_nextState = np.zeros((self.memory_size,state_size,1))
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
        
        states = self.memory_state[batch].astype(np.float32).reshape(-1,self.state_size,1)
        nextStates = self.memory_nextState[batch].astype(np.float32).reshape(-1,self.state_size,1)
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
    def __init__(self, modelParams, start_price,
                 checkpoint_dir: str,rewardParams: dict, extraParams: dict, is_eval = False):
        # model general related attributes
        for key in modelParams:
            setattr(self, key, modelParams[key])
            
        #self.stateTS_size = stateTS_size
        #self.stateUT_size = stateUT_size # already in dictionary
        self.action_size = 2
        self.buffer_size = 1000000
        self.budget = self.n_budget*start_price
        self.balance = 0. # balance parameter will be adapted
        self.is_eval = is_eval
        
        self.stateTS_size, self.stateUT_size
        
        # reward function related
        for key in rewardParams:
            setattr(self, key, rewardParams[key])
            
        for key in extraParams:
            setattr(self, key, extraParams[key])
        
        self.attr_dct = copy.deepcopy(self.__dict__) # setup attirbute dictionary thusfar
        self.actor_local_loss = 1. #initiliaze
        
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
        self.memory = ReplayBuffer((self.stateTS_size+self.stateUT_size), self.action_size, self.buffer_size, self.batch_size)
        self.actor_local = Actor(self.stateTS_size, self.stateUT_size, 
                                 self.action_size, self.model_hyper)
        self.actor_target = Actor(self.stateTS_size, self.stateUT_size, 
                                  self.action_size,  self.model_hyper)
        self.critic_local = Critic(self.stateTS_size, self.stateUT_size, 
                                   self.action_size,  self.model_hyper)
        self.critic_target = Critic(self.stateTS_size, self.stateUT_size, 
                                    self.action_size,  self.model_hyper)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        self.save_attributes() # save all simple attibutes
        
    def setup_validation(self, validation_dir):
        os.mkdir(os.path.join(os.path.join(os.getcwd(),validation_dir),"validation"))
        
    
    def update_balance(self, change: float):
        self.balance += change
    
    def update_inventory(self,cur_price: float):
        self.inventory_value = len(self.inventory)*cur_price # current value of sum of stocks
        # notice that we dont change the buy prices of stocks in self.inventory!
        
    def check_threshold(self, utils, terminateFunc_on = False):
        n_impossible = utils[0]
        min_futurePrice = utils[1]
        
        is_terminal = False
        terminal_message = "n/a"
        if terminateFunc_on:
            if n_impossible >= self.is_terminal_threshold: # too many impossibles
            #if (self.balance+self.inventory_value) < self.is_terminal_threshold or self.balance < 0:
                # terminal state reached, raise flag
                is_terminal = True
                terminal_message = "too many impossibles"
            elif not bool(self.inventory) and (self.balance-self.trade_cost) < min_futurePrice:
                # we have no stock and balance is lower than the stock will ever go
                is_terminal = True
                terminal_message = "too low balance for rest of trial"
            else:
                self.override_reward = False

        return is_terminal, terminal_message 
        
    def reset(self, start_price):
        '''
        Resets all relevant attributes for next run
        '''
        self.balance = 0.
        self.inventory = [start_price]*self.n_budget
        self.budget = self.n_budget*start_price # for validation cell this also has to be reset
        # inventory contains price at which BOUGHT, and is not updated which cur_price
        self.inventory_value = start_price*self.n_budget
        self.inventory_conj = [] 
        # conjugate of inventory, contains price at which we SOLD
        self.r_util = np.zeros(10)
        self.override_reward = False
        
    def validation_extraCash(self, extraCash):
        '''
        In the validation case the growth of the stock can prevent the system 
        from buying and thereby participating in only part of the run.
        This function provides an additional cash infusion which should not 
        alter the profit obtained as the budget is adjusted as well
        
        Note: function must be called after reset!
        '''
        self.balance = extraCash # instead of zero
        self.budget += extraCash 
    '''
    ======================== MODEL RELATED ===============================
    ''' 
        
    def act(self, state, utils: list):
        '''
         Returns an action, given a state, using the actor (policy network) and
        the output of the softmax layer of the actor-network, returning the
        probability for each action.


        NOTE; because some actions are impossible but proposed through 
        the exploration sheme it is important to ensure that if the system does 
        not follow the greedy action (the output of argmax), the explorative 
        action is at least possible! otherwise we rake up reward penalties
        '''
        states_ts = state[:,:self.stateTS_size,:]
        states_ut = state[:,-self.stateUT_size:,0]
        actions_prob = self.actor_local.model.predict([states_ts, states_ut])
        self.last_state = state
        prob = utils[0]
        price = utils[1]
        
        if not self.is_eval:
            '''
            No evaluation so allow for exploration.
            Exploration is based on 2 principles; (1) if the greedy action is 
            bad, the system should observe its consequence [otherwise system 
            will not learn from mistakes]. (2) If the action 
            is changed to an explorative one, this explorative action 
            should not result in a penalty.
            '''
            
            #'''
            action = choice(range(self.action_size), p = actions_prob[0]) 
            

        else:
            action = np.argmax(actions_prob[0])
            
            
            
        if action == 1:
            if not bool(self.inventory): #and (self.balance-self.trade_cost) > price:
                # buy
                action = 1
                '''
                notice can still be impossible due to too low balance!
                '''
            elif bool(self.inventory):
                # sell
                action = 2
            '''
            else:
                # hold no stock and cannot buy so hold
                action = 0
            '''
        return action, actions_prob


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
        
        states_ts = np.expand_dims(states[:,:self.stateTS_size],axis = -1)
        states_ut = states[:,-self.stateUT_size:]
        states = [states_ts, states_ut]
        
        next_states_ts = next_states[:,:self.stateTS_size,:]
        next_states_ut = next_states[:,-self.stateUT_size:,0]
        next_states = [states_ts, states_ut]
        
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states_ts, next_states_ut, actions_next])
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        self.critic_local.model.train_on_batch(x = [states_ts, states_ut, actions], y = Q_targets)
        action_gradients = np.reshape(self.critic_local.get_action_gradients(states_ts, states_ut, actions),(-1,self.action_size))
        self.actor_local_loss = self.actor_local.train_fn(states_ts, states_ut, action_gradients)
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
        np.savez_compressed(os.path.join(self.checkpoint_path, 'Rbuffer.npz'), 
                    a = self.memory.memory_state,
                    b = self.memory.memory_nextState, 
                    c = self.memory.memory_action,
                    d = self.memory.memory_reward,
                    e = self.memory.memory_dones,
                    f = np.array(self.memory.memory_counter))
        print("Succesfully saved models for episode {}".format(episode))
        
    def load_models(self, checkpoint_dir: str, episode:int, 
                    actor = True, critic = True, buffer = False, using_colab = False):
        checkpoint_path = os.path.join(os.getcwd(),checkpoint_dir)
        if actor:
            self.actor_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_local.h5'))
            self.actor_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_target.h5'))
        if critic:
            self.critic_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_local.h5'))
            self.critic_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_target.h5'))
        if buffer:
            if using_colab:
                Rbuffer = np.load(os.path.join(checkpoint_path,'e{}'.format(episode),'Rbuffer.npz'))
            else:
                Rbuffer = np.load(os.path.join(checkpoint_path,'Rbuffer.npz'))
            self.memory.memory_state = Rbuffer['a']
            self.memory.memory_nextState = Rbuffer['b']
            self.memory.memory_action = Rbuffer['c']
            self.memory.memory_reward = Rbuffer['d']
            self.memory.memory_dones = Rbuffer['e']
            self.memory.memory_counter = int(Rbuffer['f']) 
            '''
            #int(episode*buffer_episode_size) # APPROXIMATION
            
            print("Set replay buffer memory counter to approx {0}*{1} = {2}".format(episode,
                                                                                    buffer_episode_size,
                                                                                    self.memory.memory_counter))
            '''
            
        # TODO; also load (hyper)parameters
        print("Succesfully loaded (actor:{2}|critic:{3}|buffer:{4}) models from folder {0} and episode {1}".format(checkpoint_dir,
                                                                                                        episode,
                                                                                                        actor,
                                                                                                        critic,
                                                                                                        buffer))

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
        
        self.r_util = np.zeros(10) # utility variable for every reward type if necessayr
        
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
        elif rewardType == 6:
            msg = "EXPERIMENTAL; reward 6 change description"
            self.get_reward = self._reward_type6
        elif rewardType == 7:
            msg = "EXPERIMENTAL; ONLY USE FOR N = 1"
            self.get_reward = self._reward_type7
        elif rewardType == 8:
            msg = "Reward to learn trading, avoiding impossible actions"
            self.get_reward = self._reward_type8
        elif rewardType == 9:
            msg = "Reward for profit and hold penalty no hold profit"
            self.get_reward = self._reward_type9
        print("Reward function description: "+msg)
    
    def switch_rewardType(self, switch: int, switch_episode: int, episode: int):
        '''
        Function to switch from reward type during training
        '''
        if episode == switch_episode:
            
            print("Switching from rewardtype {0} to {1}".format(self.rewardType,switch))
            self.set_rewardtype(switch)
            self.rewardType = switch
    
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
            if reward == 0:
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
    
    
    def _reward_type6(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        EXPERIMENTAL; change description
        '''
        pt = util_lst[0] # price
        pt1 = util_lst[1] # prev price
        ptn = util_lst[2] 
        at = util_lst[3] # action 
        at_prob = util_lst[4] # action probabilities
        n_trades = util_lst[5] # total number of trades during run
        n_holds = util_lst[6] # concurrent holds, resets after a sell/buy
        impossible = util_lst[7] # invalid action 
        l = util_lst[8] # length of data
        terminate = util_lst[9]
        
        '''
        penalty = -1000 # -10 # -1000000
        hold_scale = 10 #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = 14
        trade_cost = 3 #2.5 #7# 5 # transaction cost
        #hold_bonus = 1 #2.5 # 1.5
        '''
        penalty = self.penalty # -10 # -1000000
        hold_scale = self.hold_scale #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = self.trade_scale
        #trade_cost = self.trade_cost #2.5 #7# 5 # transaction cost
        max_holds = self.max_holds
        #hold_bonus = 1 #2.5 # 1.5
        
        reward = 0
        #prob = at_prob[at]**self.prob_power #**0.2 # probability of action transformed for better gradient
        
        
        at_argmx = np.argmax(at_prob)
        prob = at_prob[at_argmx]**self.prob_power #**0.2 # probability of action transformed for better gradient

        reward = 0
        
        if at == 1 or at == 2:
            reward = max(profit,0)*prob        
        return reward/1000
    
    
    def _reward_type7(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        EXPERIMENTAL; change description
        '''
        pt = util_lst[0] # price
        pt1 = util_lst[1] # prev price
        ptn = util_lst[2] 
        at = util_lst[3] # action 
        at_prob = util_lst[4] # action probabilities
        n_trades = util_lst[5] # total number of trades during run
        n_holds = util_lst[6] # concurrent holds, resets after a sell/buy
        impossible = util_lst[7] # invalid action 
        l = util_lst[8] # length of data
        terminate = util_lst[9]
        
        '''
        penalty = -1000 # -10 # -1000000
        hold_scale = 10 #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = 14
        trade_cost = 3 #2.5 #7# 5 # transaction cost
        #hold_bonus = 1 #2.5 # 1.5
        '''
        penalty = self.penalty # -10 # -1000000
        hold_scale = self.hold_scale #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = self.trade_scale
        #trade_cost = self.trade_cost #2.5 #7# 5 # transaction cost
        max_holds = self.max_holds
        #hold_bonus = 1 #2.5 # 1.5
        
        reward = 0
        #prob = at_prob[at]**self.prob_power #**0.2 # probability of action transformed for better gradient
        
        
        at_argmx = np.argmax(at_prob)
        prob = at_prob[at_argmx]**self.prob_power #**0.2 # probability of action transformed for better gradient

        
        if not terminate: 
            '''
            # first if statement can be extended to all undesired strategies, 
            (rewarding zero reward from the start), these include:
                - n_trades == 0: buy hold 
                - 
                - 
                - 
                - 
            '''
            
            
            #p_val = agent.balance + agent.inventory_value # portfolio value
            #bh_val = pt*agent.n_budget # buy hold value
            #ratio = (p_val/bh_val)**power # to distinguish between attempt, not necessarily within an attempt
            
            
            '''
            profit adjustment
            '''
            #profit_adj = profit - trade_cost # ALREADY DONE
            #hold_penalty = (-np.exp((n_holds-max_holds)/l*hold_scale)+1)
            #n_invent = len(agent.inventory) # stocks in inventory

            if at == 0 or impossible:
                # hold position; 
                n_invent = len(agent.inventory) # stocks in inventory
                hold_penalty = (-np.exp((n_holds-max_holds)/l*hold_scale)+1)
                
                if n_invent != 0:
                    # in case stock is held; reward growth
                    #reward = max((pt-pt1)*n_invent*ratio,0) + hold_penalty # notice if n_invent = 0, this equals zero
                    #reward = (pt-pt1)*n_invent*ratio + hold_penalty # notice if n_invent = 0, this equals zero
                    #reward = (pt-pt1)*n_invent*ratio*hold_bonus + hold_penalty # notice if n_invent = 0, this equals zero
                    #reward = ((pt-pt1)*n_invent + hold_penalty)*prob # notice if n_invent = 0, this equals zero
                    reward = ((ptn-pt)*n_invent + hold_penalty)*prob # notice if n_invent = 0, this equals zero

                    '''
                    if n_holds < max_holds:
                        #reward = ((pt-pt1)*n_invent)*prob # notice if n_invent = 0, this equals zero
                        #reward = max(((pt-pt1)*n_invent)*prob,0)*max(1-n_holds/max_holds,0) #*max(1-np.exp(n_holds/max_holds),0) # notice if n_invent = 0, this equals zero
                        reward = ((pt-pt1)*n_invent*prob)*max(1-n_holds/max_holds,0) #*max(1-np.exp(n_holds/max_holds),0) # notice if n_invent = 0, this equals zero

                    else:
                        reward = 0
                    #'''
                else:
                    # in case stock is NOT held; introduce an opportunity cost or reward
                    #reward = max(-1*(pt-pt1)*ratio,0)
                    #reward = -1*(pt-pt1)*ratio
                    #reward = -1*(pt-pt1)*ratio*hold_bonus
                    #reward = (-1*(pt-pt1) + hold_penalty)*prob
                    reward = (-1*(ptn-pt) + hold_penalty)*prob

                    '''
                    if n_holds < max_holds:
                        #reward = (-1*(pt-pt1))*prob # notice if n_invent = 0, this equals zero
                        #reward = max((-1*(pt-pt1))*prob,0)*max(1-n_holds/max_holds,0) #*max(1-np.exp(n_holds/max_holds),0) # notice if n_invent = 0, this equals zero
                        reward = (-1*(pt-pt1)*prob)*max(1-n_holds/max_holds,0) #*max(1-np.exp(n_holds/max_holds),0) # notice if n_invent = 0, this equals zero

                    else:
                        reward = 0 
                    #'''

                
            elif at == 1:
                if impossible:  
                    # buy action while we already had stock, IMPOSSIBLE
                    reward = penalty*prob
                    print("here")
                else:
                    # buy; reward the conjugate profit 
                    #reward = profit*ratio
                    #reward = profit*ratio+(-np.exp((n_trades)/l*trade_scale)+1)  # scaling to avoid the profit reward from overtaking the final reward
                    # ^reward of v6_tradePen
                    #reward = profit_adj*prob
                    reward = profit*prob #max(profit*prob,0)
                    #reward = max(profit_adj*ratio,0)
                    # reward of v6_prftAdj
                    #reward = profit*prob
                    
            elif at == 2:
                # sell action
                if impossible:
                    # sale action while we dont have a stock IMPOSSIBLE
                    reward = penalty*prob
                    # note, yes this could trigger with a sale as well, but higly unlikely
                else:
                    # sale; reward the profit 
                    #reward = profit*ratio
                    #reward = profit*ratio+(-np.exp((n_trades)/l*trade_scale)+1) # scaling to avoid the profit reward from overtaking the final reward
                    #reward = profit_adj*prob
                    reward = profit*prob
                    #reward = max(profit*prob,0)
                    #reward = profit*prob
            '''
            # TESTINGGGG ================
            if reward != penalty:
                # TODO REMOVE
                reward = max(0,reward)
            '''
            if impossible and reward > 0:
                reward *= 1/10
            if impossible and reward < 0:
                reward *= 1.1
                
            reward = max(reward, -50000) # clip to avoid numerical errors
        else:
            reward = -100000 #-10000

        return reward/1000
    
    
    
    def _reward_type8(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        Reward function focussed on learning agent trading is good, 
        but not to make impossible trades
        '''
        pt = util_lst[0] # price
        pt1 = util_lst[1] # prev price
        ptn = util_lst[2] 
        at = util_lst[3] # action 
        at_prob = util_lst[4] # action probs 
        n_trades = util_lst[5] # total number of trades during run
        n_holds = util_lst[6] # concurrent holds, resets after a sell/buy
        impossible = util_lst[7] # invalid action 
        l = util_lst[8] # length of data
        terminate = util_lst[9]
        
        penalty = self.penalty # -10 # -1000000
        hold_scale = self.hold_scale #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = self.trade_scale
        #trade_cost = self.trade_cost #2.5 #7# 5 # transaction cost
        max_holds = self.max_holds
        #self.r_util[0] += 1
        
        reward = 0 
        #prob = at_prob[at]**0.2
        prob = at_prob[at]**self.prob_power
        if at == 1 or at == 2:
            # a trade 
            if impossible:
                reward = penalty*prob
            else:
                reward = -1/20*penalty*prob #-1*penalty # positive reward
                #reward = max(profit,0)*prob #-1*penalty # positive reward
        else:
            hold_penalty = min((-np.exp((n_holds-max_holds)/l*hold_scale)+1),0)
            reward = hold_penalty*prob 
        '''
        TODO CHANGE REWARDTYPE HERE AFTER EPISODES
        this is 
        '''
        
        return reward/1000
    
    
    def _reward_type9(self, agent, profit: float, 
                      util_lst: list, last: bool):
        '''
        EXPERIMENTAL; change description
        '''
        pt = util_lst[0] # price
        pt1 = util_lst[1] # prev price
        ptn = util_lst[2] 
        at = util_lst[3] # action 
        at_prob = util_lst[4] # action probabilities
        n_trades = util_lst[5] # total number of trades during run
        n_holds = util_lst[6] # concurrent holds, resets after a sell/buy
        impossible = util_lst[7] # invalid action 
        l = util_lst[8] # length of data
        terminate = util_lst[9]
        
        '''
        penalty = -1000 # -10 # -1000000
        hold_scale = 10 #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = 14
        trade_cost = 3 #2.5 #7# 5 # transaction cost
        #hold_bonus = 1 #2.5 # 1.5
        '''
        penalty = self.penalty # -10 # -1000000
        hold_scale = self.hold_scale #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = self.trade_scale
        #trade_cost = self.trade_cost #2.5 #7# 5 # transaction cost
        max_holds = self.max_holds
        #hold_bonus = 1 #2.5 # 1.5
        
        reward = 0
        #prob = at_prob[at]**self.prob_power #**0.2 # probability of action transformed for better gradient
        
        
        at_argmx = np.argmax(at_prob)
        prob = at_prob[at_argmx]**self.prob_power #**0.2 # probability of action transformed for better gradient

        
        if not terminate: 
            '''
            # first if statement can be extended to all undesired strategies, 
            (rewarding zero reward from the start), these include:
                - n_trades == 0: buy hold 
                - 
                - 
                - 
                - 
            '''
            
            
            #p_val = agent.balance + agent.inventory_value # portfolio value
            #bh_val = pt*agent.n_budget # buy hold value
            #ratio = (p_val/bh_val)**power # to distinguish between attempt, not necessarily within an attempt
            
            
            '''
            profit adjustment
            '''
            #profit_adj = profit - trade_cost # ALREADY DONE
            if at == 0:
                # hold position; 
                n_invent = len(agent.inventory) # stocks in inventory
                hold_penalty = (-np.exp((n_holds-max_holds)/l*hold_scale)+1)
                if (not bool(agent.inventory) and agent.balance-agent.trade_cost > pt) or (bool(agent.inventory)): 
                    reward = -100*prob
                    if n_holds > max_holds:
                        reward = -1000*prob
                
            elif at == 1:
                if impossible:  
                    # buy action while we already had stock, IMPOSSIBLE
                    reward = penalty*prob
                else:
                    # buy; reward the conjugate profit 
                    reward = profit*prob
                    #reward = max(profit_adj*ratio,0)
                    # reward of v6_prftAdj
                    
            elif at == 2:
                # sell action
                if impossible:
                    # sale action while we dont have a stock IMPOSSIBLE
                    reward = penalty*prob
                    # note, yes this could trigger with a sale as well, but higly unlikely
                else:
                    # sale; reward the profit 
                    reward = profit*prob
            reward = max(reward, -50000) # clip to avoid numerical errors
        else:
            reward = -100000 #-10000

        return reward/1000
    

#%% Utility functions

class UtilFuncs:
    '''
    This class contains utility functions used throughout the main project script
    and which do not directly belong to main Agent class 
    '''
    def to_currency(n):
        if n>=0:
            curr = "+$"
        else:
            curr = "-$"
        return (curr +"{0:.2f}".format(abs(n)))
    
    def get_data(key: str, window: int, colab = False) -> np.array:
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
    
    
    def get_state(agent, data: np.array, t: int, window: int, utils: list, use_rtn = True) -> np.array:
        
        # unpack utils
        l = utils[0] # length of full data 
        n_holds = utils[1] # concurrent holds, resets after a sell/buy
        n_trades = utils[2] # total amount of trades does not reset
        tradeCost = utils[3]
        tanh_scale = agent.tanh_scale
        mask_input = agent.mask_input

        if t - window >= -1:
            state = data[t - window+ 1:t+ 1]
        else:
            state = np.pad(data[0:t+1],(-(t-window+1),0),'constant',constant_values=(data[0],np.nan))
        # scaling of state (standardization of input)
        if use_rtn:
            # use returns to scale, NOTE that we loose one data entry this way
            state = np.diff(state) # returns
        else:
            state = state[1:] # consistent length
            
        
        state = state/tanh_scale
        state = np.tanh(state)
        '''
        tanh scaling, scale is based on distributional plots of returns, default 80
        '''
        if mask_input:
            state_zeros = np.zeros(window-1)
            state_zeros[min(-1,-min(n_holds,window)):] = state[min(-1,-min(n_holds,window)):]
            state = state_zeros 
        
        '''
        Masking of inputs since last trade
        '''

        #balance_norm = (agent.balance-data[t])/data[t]
        if not bool(agent.inventory):
            balance_bool = float(agent.balance-tradeCost > data[t])
        else:
            # ensure that if system never thinks it can buy more than one
            balance_bool = 0. 
        nholds_norm = min(1,n_holds/max(agent.max_holds, 100)) #(l-window) # time duration of current hold position, resets at buy/sell
        holding = float(len(agent.inventory)) # binary, whether or not we have a stock
        if not bool(agent.inventory):
            # no stock held so no buy price 
            bought_price = 0
            sold_price = agent.inventory_conj[0]
            profit = sold_price - data[t] - tradeCost
            buy_bool = float(sold_price-tradeCost > data[t])
            sell_bool = 0 
        else:
            # no stock sold yet so no sell price
            bought_price = agent.inventory[0]
            sold_price = 0
            profit = data[t] - bought_price - tradeCost
            sell_bool = float(bought_price+tradeCost < data[t])
            buy_bool = 0
        
        #buy_norm = (data[t]-bought_price-tradeCost)/data[t]
        #buy_bool = float(sold_price-tradeCost > data[t])
        #sell_bool = float(bought_price+tradeCost < data[t])
        #sell_norm = (sold_price-data[t]-tradeCost)/data[t]
        profit_norm = profit/data[t]
        append = [balance_bool, nholds_norm, holding,
                  buy_bool, sell_bool, profit_norm]
        state = np.append(state,append) # TODO, maybe clip these to max of 1?
        
        state = np.expand_dims(state,axis = (0,2))
        return state


    def break_deadlock(agent,action: int, episode: int, utils, on = False):
        '''
        Function which changes the action proposed to encourage exploration
        and avoid a deadlock in which the prob for hold essentially destroys
        exploration
        Notice that we only adapt the action and not the probability of those 
        actions. Hence we 'hacked' the system by false presenting an 
        exploratory action which higher probability
        
        aim: is to allow for more exploration but avoid the 'impossible' 
        penalties thereby allow the system to raise the probabilities 
        of buy/sell actions 
        (suggesting actions that result in the impossible penalty would 
         actually have an adverse effect wrt to goal)
        '''
        
        if on and action == 0:
            prob = utils[0]
            price = utils[1]
            tradeCost = utils[2]
            
            action = choice(range(3), p = [1-2*prob, prob, prob]) #[2/3, 1/6, 1/6]
            if action == 1:                
                if len(agent.inventory) == 0 and (agent.balance-tradeCost) > price:
                    b = 1
                else:
                    
                    action = 0 
            elif action == 2:
                if len(agent.inventory) > 0:
                    b = 1
                else:
                    # otherwise would be impossible 
                    action = 0 
        return action
    
    
    def handle_action(agent, stats, action, data, t, flags, utils, training = True):
        # unpack
        use_terminateFunc = flags[0]
        terminateFunc_on = flags[1]
        action_prob = utils[0]
        action_argmx = np.argmax(action_prob) # preferred action by system
        
        # initialize
        profit = 0 
        change = 0 
        impossible = False

        
        if action == 0:
            stats.n_holds += 1
            
        #elif not training and action == 0 and agent.balance < np.min(data[(t+1):]) and not bool(agent.inventory):
        #stats.n_holds += 1
        #stats.extraCash += data[t] - agent.balance # extra cash required for purchase
        #stats.xtr_ind.append(t)
        #agent.reset(data[t+1]) # reset the portfolio
        
        
        elif action == 1:
            stats.n_1or2 += 1
            if (agent.balance-agent.trade_cost) > data[t] and not bool(agent.inventory): #max one stock 
                # BUYING stock, only if there is balance though
                agent.inventory.append(data[t])
                sold_price = agent.inventory_conj.pop(0)
                
                profit = sold_price - data[t] -agent.trade_cost
                
                change = -data[t]-agent.trade_cost
                stats.buy_ind.append(t)
                stats.n_trades += 1
                stats.n_holds = 0 # reset counter
            
            else:
                impossible = True
                stats.n_impossible += 1
                stats.imp_ind.append(t)
                stats.n_holds += 1 # effectively no buy is a hold
                if not use_terminateFunc:
                    terminate = True
                    term_msg = "impossibles"
            
        elif action == 2:
            stats.n_1or2 += 1
            if bool(agent.inventory): 
                # SELLING stock, only if there are stocks held

                bought_price = agent.inventory.pop(0)
                agent.inventory_conj.append(data[t])
                
                profit = data[t] - bought_price -agent.trade_cost

                change = data[t]-agent.trade_cost
                stats.sell_ind.append(t)
                stats.n_trades += 1
                stats.n_holds = 0 # reset counter
            else:
                impossible = True
                stats.n_impossible += 1
                stats.imp_ind.append(t)
                stats.n_holds += 1 # effectively no sell is a hold
                if not use_terminateFunc:
                    terminate = True
                    term_msg = "impossibles"
        
        if not training and (agent.balance-agent.trade_cost) < data[t] and not bool(agent.inventory) and action_argmx == 1:
            '''
            In this statement extra cash required is recorded for the validation case
            This is done as to not hinder the validation process due to a single 
            bad trade
            
            notice the sign of agent.balance < data is reversed
            '''

            stats.extraCash += data[t] - agent.balance - agent.trade_cost # extra cash required for purchase
            stats.xtr_ind.append(t)
            agent.reset(data[t]) # reset the portfolio
            profit = 0 
            
            #stats.buy_ind.append(t)
            stats.n_trades += 1
            stats.n_holds = 0 # reset counter
        
        if profit > 0:
            # good trade made 
            stats.n_posiProfits += 1

        
        # update and check termination condition
        agent.update_balance(change)
        agent.update_inventory(data[t])
        if use_terminateFunc and training:
            utils_term = [stats.n_impossible, np.min(data[(t+1):])]
            terminate, term_msg = agent.check_threshold(utils_term, terminateFunc_on= terminateFunc_on)
        else:
            terminate = False
            term_msg = ""
            
        return action, profit, impossible, terminate, term_msg
    
    
    def plot_data(agent, data, data_extra, data_extraWindow, window_size, training = True):
        if training:
            msg = "Training"
        else:
            msg = 'Test'
        fig = pgo.Figure()
        fig.update_layout(showlegend=True, title_text ="{} data".format(msg))
        fig.add_trace(pgo.Scatter(x=np.arange(len(data)), y=data,
                            mode='lines',
                            name='stock growth'))
        fig.add_trace(pgo.Scatter(x=np.arange(len(data_extra)), y=data_extra,
                            mode='lines',
                            name='predata W ={}'.format(data_extraWindow)))
        fig.show()
        growth_buyhold_per = (data[-1]-data[window_size])/data[window_size]
        
        print("Naive buy & hold strategy on {1} data has a portfolio growth of {0}% per asset bought".format(round(growth_buyhold_per,3),msg))
        growth_buyhold_cash = agent.n_budget*data[window_size]*growth_buyhold_per
        growth_buyhold = (agent.n_budget*(data-data[window_size]))[window_size:-1]
        print("For current budget of {0}, this means {1} stocks bought results in a final portfolio growth of {2} (i.e. final value ={3})".format(UtilFuncs.to_currency(agent.n_budget*data[window_size]),
                                                                                            agent.n_budget,
                                                                                            UtilFuncs.to_currency(growth_buyhold_cash),
                                                                                            UtilFuncs.to_currency(growth_buyhold_cash+agent.n_budget*data[window_size])))
        return growth_buyhold

    def get_episodeStart(agent, expand_i, expand, utils):
        
        # unpack
        l = utils[0]
        offset = utils[1]
        episode_window = utils[2]
    
        #
        window_size = agent.stateTS_size
        
        start = max(window_size,offset) 
        extra = min(offset-expand_i*expand,0)*-1

        choice_start = max(start-expand_i*expand,window_size) 
        choice_end  = min(start+expand_i*expand + extra, 
                          l-episode_window-1) # -1 in case we use rwrd func that uses p_t+1
        
        #episode_start = np.random.randint(choice_start,choice_end)
        choice_prob = np.ones((choice_end-choice_start))/(choice_end-choice_start+expand*2*min(expand_i,10)) 
        if choice_start != window_size and choice_end != l-episode_window-1:
            if extra == 0:
                choice_prob[:expand] *= min(expand_i,10)
                choice_prob[-expand:] *= min(expand_i,10)
            else:
                # front is not added
                choice_prob[-(expand*2):] *= min(expand_i,10)
        choice_prob /= np.sum(choice_prob) # additional normalization 
        
        episode_start = np.random.choice(np.arange(choice_start,
                                                   choice_end),
                                         p = choice_prob)
        
        return episode_start
    
    
    
#%% Statistics container
class Statistics:
    '''
    This class contains all counters, containers etc. needed to keep track 
    of performance and other useful statistics and helps avoid cluttering of 
    the code
    '''
    def __init__(self, checkpoint_dir, training = True):
        self.checkpoint_dir = checkpoint_dir
        self.training = training
        
    def reset_episode(self):
        '''
        Function that resets all relevant counters and containers
        '''
        self.total_reward = 0. # total profit resets every epsiode 
        self.n_trades = 0 
        self.n_posiProfits = 0  # positive profits made 
        self.n_impossible = 0 
        self.n_holds = 0 
        self.n_1or2 = 1 # 1 not zero because we cant have division by zero 
        self.extraCash = 0.
        
        self.profits = []
        self.balances = []
        self.rewards = []
        self.inventories = [] # inventory value (only stocks)
        self.actor_local_losses = []
        self.buy_ind = []
        self.sell_ind = []
        self.xtr_ind = []
        self.imp_ind = []
        self.growth =[]
        self.compete = []
        self.trades_list = []
        self.actions = []
        
            
    def reset_all(self,budget: float, growth_buyhold: np.array):
        self.budget = budget
        self.growth_buyhold = growth_buyhold.tolist() # used later
        self.totalReward_list = []
        self.lastLosses_list=[]
        self.impossible_list = []
        self.tradeRatio_list = []
        self.everyProfit_dct = {}
        self.everyBalance_dct = {}
        self.everyReward_dct = {}
        self.everyAction_dct = {}
        self.everyInventory_dct = {}
        self.everyGrowth_dct = {}
        self.everyGrowth_dct["buyhold"] = self.growth_buyhold
        self.everyCompete_dct = {}
        self.everyLoss_dct = {}
        
        self.reset_episode() # reset all other lists as well


    def pad_on_terminate(self, utils):
        '''
        This function ensures consistency for length of lists by padding 
        '''
        # unpack
        l = utils[0]
        t = utils[1]
        
        # pad lists
        self.balances = np.pad(self.balances,(0,l-t-1),'constant',constant_values=(0,self.balances[-1])).tolist()
        self.inventories = np.pad(self.inventories,(0,l-t-1),'constant',constant_values=(0,self.inventories[-1])).tolist()
        self.profits = np.pad(self.profits,(0,l-t-1),'constant',constant_values=(0,0)).tolist()
        self.rewards = np.pad(self.rewards,(0,l-t-1),'constant',constant_values=(0,0)).tolist()
        self.actor_local_losses = np.pad(self.actor_local_losses,(0,l-t-1),'constant',constant_values=(0,self.actor_local_losses[-1])).tolist()
        self.actions = np.pad(self.actions,(0,l-t-1),'constant',constant_values=(0,-1)).tolist() # -1 as to easily recognize

    
    def collect_iteration(self,agent,utils):
        # unpack
        profit = utils[0]
        reward = utils[1]
        actor_local_loss = utils[2]
        action = utils[3]
        t = utils[4]
        
        # append
        self.balances.append(agent.balance)
        self.inventories.append(agent.inventory_value)
        self.profits.append(profit)
        self.rewards.append(reward)
        self.actor_local_losses.append(float(actor_local_loss))
        self.actions.append(int(action))
        growth = agent.balance+agent.inventory_value-self.budget-self.extraCash
        self.growth.append(growth)
        self.compete.append(growth-self.growth_buyhold[t]) 

        
    def collect_episode(self,agent,episode, utils):
        #self.growth = (np.array(self.balances)+np.array(self.inventories)-agent.budget).tolist() # 
        #self.compete = (np.array(self.growth)-np.array(self.growth_buyhold)).tolist() # compete vs buyhold

        self.totalReward_list.append(self.total_reward)
        self.lastLosses_list.append(self.actor_local_losses[-1])
        self.impossible_list.append(int(self.n_impossible))
        self.trades_list.append(int(self.n_trades))
        self.tradeRatio_list.append(float(self.n_posiProfits/max(1,self.n_trades)))
        
        episode_name = "e{}".format(episode)
        self.everyProfit_dct[episode_name] = self.profits
        self.everyBalance_dct[episode_name] = self.balances
        self.everyReward_dct[episode_name] = self.rewards
        self.everyInventory_dct[episode_name] = self.inventories
        self.everyGrowth_dct[episode_name] = self.growth
        self.everyCompete_dct[episode_name] = self.compete
        self.everyLoss_dct[episode_name] = self.actor_local_losses
        self.everyAction_dct[episode_name] = self.actions


    '''
    ============= LOAD/SAVE/PLOT RELATED =====================
    '''
    def save_statistics(self, episode: int):
        self.reset_episode() # reset all other lists to save memory
        attr_dct = self.__dict__ 
        with open('./{0}/statistics.json'.format(self.checkpoint_dir,episode), 'w') as fp:
            json.dump(attr_dct, fp) # this will overwrite!
        print("Succesfully saved e{0} statistics to results folder".format(episode))

        '''
        with open(f'./{checkpoint_dir}/e{e}/totalReward.npy', 'wb') as f:
                np.save(f, np.array(totalReward_list))
        with open(f'./{checkpoint_dir}/e{e}/losses.npy', 'wb') as f:
            np.save(f, np.array(lastLosses_list))
        with open(f'./{checkpoint_dir}/e{e}/impossibles.npy', 'wb') as f:
            np.save(f, np.array(impossible_list))
        with open(f'./{checkpoint_dir}/e{e}/tradeRatios.npy', 'wb') as f:
            np.save(f, np.array(tradeRatio_list))
        with open(f'./{checkpoint_dir}/e{e}/everyProfit.json', 'w') as fp:
            json.dump(everyProfit_dct, fp)
        with open(f'./{checkpoint_dir}/e{e}/everyBalance.json', 'w') as fp:
            json.dump(everyBalance_dct, fp)
        with open(f'./{checkpoint_dir}/e{e}/everyReward.json', 'w') as fp:
            json.dump(everyReward_dct, fp)
        with open(f'./{checkpoint_dir}/e{e}/everyInventory.json', 'w') as fp:
            json.dump(everyInventory_dct, fp)
        with open(f'./{checkpoint_dir}/e{e}/everyLoss.json', 'w') as fp:
            json.dump(everyLoss_dct, fp)
        '''
        
        
    def plot_figure(self,data,episode, utils, show_figs = False):
        l = utils[0]
        window_size = utils[1]
        fig = pgo.Figure() # figure 
        fig.update_layout(showlegend=True, xaxis_range=[window_size, l], 
                          title_text = "E{3} final profit RL: {0} vs buyhold: {1}, difference = {2}| +trades/Alltrades ={4}/{5}={6} | extracash = {7}" \
                                  .format(round(self.growth[-1],2),
                                                round(self.growth_buyhold[-1],2),
                                                round((self.compete[-1]),2),episode,
                                                self.n_posiProfits,
                                                self.n_trades,
                                                round(self.tradeRatio_list[-1],2),
                                                round(self.extraCash,2)))
        # buy/sell traces imposed on data line
        x = np.arange(len(data))
        fig.add_trace(pgo.Scatter(x=x, y=data,
                            mode='lines',
                            name='data',
                            legendgroup = '1'))
        fig.add_trace(pgo.Scatter(x=self.buy_ind, y=data[self.buy_ind], marker_color = "green",
                            mode='markers',
                            name='buy',
                            legendgroup = '1'))
        fig.add_trace(pgo.Scatter(x=self.sell_ind, y=data[self.sell_ind], marker_color = "red",
                            mode='markers',
                            name='sell',
                            legendgroup = '1'))
        # impossible/extra cash traces imposed on data line
        fig.add_trace(pgo.Scatter(x=self.xtr_ind, y=data[self.xtr_ind], marker_color = "black",
                            mode='markers',
                            name='extraCash',
                            legendgroup = '2'))
        fig.add_trace(pgo.Scatter(x=self.imp_ind, y=data[self.imp_ind], marker_color = "orange",
                            mode='markers',
                            name='impossibles',
                            legendgroup = '2'))
        
        # growth traces
        x = np.arange(window_size,window_size+len(data))
        fig.add_trace(pgo.Scatter(x=x, y=np.array(self.growth), marker_color = "red",
                            mode='lines',
                            name='growth-RL',
                            legendgroup = '3'))
        fig.add_trace(pgo.Scatter(x=x, y=np.array(self.growth_buyhold), marker_color = "green",
                            mode='lines',
                            name='growth-buyhold',
                            legendgroup = '3'))
        #
        
        if show_figs:
            fig.show(render = "browser")
            
        if self.training:
            fig.write_html("./{0}/results/e{1}_trades.html".format(self.checkpoint_dir,episode))
        else:
            # validation set
            fig.write_html("./{0}/results/e{1}_TestTrades.html".format(self.checkpoint_dir,episode))
        fig.data = [] # reset traces