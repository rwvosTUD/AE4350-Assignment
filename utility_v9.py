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
    def __init__(self, state_size, action_size, hidden_units, regularizer):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.regularizer = regularizer
        self.build_model()

    def build_model(self):
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        
        net = tf.keras.layers.Dense(units=self.hidden_units[0][0],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(states)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Dense(units=self.hidden_units[0][1],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        # additional layers

        net = tf.keras.layers.Dense(units=self.hidden_units[0][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
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
    def __init__(self, state_size, action_size, hidden_units, regularizer):
        '''
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.regularizer = regularizer
        self.build_model()

    def build_model(self):
        # define inputs
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')
        
        # 
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][0],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(states)
        net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][1],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_states)

        # == additional layer
        net_states = tf.keras.layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Activation("relu")(net_states)
        net_states = tf.keras.layers.Dense(units=self.hidden_units[1][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net_states)
        # ==
        
        # 
        net_actions = tf.keras.layers.Dense(units=self.hidden_units[1][2],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(actions)
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('relu')(net)
        
        # additional layer
        '''
        # no additional layers because notice that there isnt even activation on 
        net = tf.keras.layers.Dense(units=self.hidden_units[1][3],kernel_regularizer=tf.keras.regularizers.l2(self.regularizer))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('relu')(net)
        '''

        
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
    def __init__(self, state_size, batch_size,
                 hidden_units, regularizer,
                 start_price, n_budget, is_terminal_threshold, 
                 checkpoint_dir: str,rewardType = 1, data_extraWindow = 1, is_eval = False):
        self.state_size = state_size+6 #+8 # +5 for additional states, see get_state
        self.action_size = 3
        self.buffer_size = 1000000
        self.batch_size = batch_size
        
        self.n_budget = n_budget # number of stocks in budget
        self.budget = self.n_budget*start_price
        self.balance = 0. # balance parameter will be adapted
        
        self.is_terminal_threshold = is_terminal_threshold # howmany impossibles
        self.is_eval = is_eval
        self.gamma = 0.99
        self.tau = 0.001 #0.001 #2 #0.001
        self.actor_local_loss = 1.
        self.rewardType = rewardType
        self.data_extraWindow = data_extraWindow
        self.hidden_units = hidden_units
        self.regularizer = regularizer
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
        self.actor_local = Actor(self.state_size, self.action_size, self.hidden_units, self.regularizer)
        self.actor_target = Actor(self.state_size, self.action_size,  self.hidden_units, self.regularizer)

        self.critic_local = Critic(self.state_size, self.action_size,  self.hidden_units, self.regularizer)
        self.critic_target = Critic(self.state_size, self.action_size,  self.hidden_units, self.regularizer)
        
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
            elif not bool(self.inventory) and self.balance < min_futurePrice:
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
        actions_prob = self.actor_local.model.predict(state)
        action = np.argmax(actions_prob[0]) # greedy action
        self.last_state = state
        if not self.is_eval:
            '''
            No evaluation so allow for exploration.
            Exploration is based on 2 principles; (1) if the greedy action is 
            bad, the system should observe its consequence [otherwise system 
            will not learn from mistakes]. (2) If the action 
            is changed to an explorative one, this explorative action 
            should not result in a penalty.
            '''
            prob = utils[0]
            price = utils[1]
            
            action_explore = choice(range(3), p = actions_prob[0]) 
            
            if action != action_explore:
                # if equal, action passes right through even if impossible
                if action_explore == 1:                
                    if len(self.inventory) == 0 and self.balance > price:
                        action = action_explore
                    else:
                        # otherwise would be impossible 
                        action = 0 
                elif action_explore == 2:
                    if len(self.inventory) > 0:
                        action = action_explore
                    else:
                        # otherwise would be impossible 
                        action = 0 
                else:
                    # action_explore = 0
                    action = 0

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
        
    def load_models(self, checkpoint_dir: str, episode:int, actor = True, critic = True):
        checkpoint_path = os.path.join(os.getcwd(),checkpoint_dir)
        if actor:
            self.actor_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_local.h5'))
            self.actor_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'actor_target.h5'))
        if critic:
            self.critic_local.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_local.h5'))
            self.critic_target.model.load_weights(os.path.join(checkpoint_path, 'e{}'.format(episode),'critic_target.h5'))
        # TODO; also load (hyper)parameters
        print("Succesfully loaded (actor:{2}|critic:{3}) models from folder {0} and episode {1}".format(checkpoint_dir,
                                                                                                        episode,
                                                                                                        actor,
                                                                                                        critic))

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
            msg = "Reward to avoid impossible actions, with negative bonus for buyhold and small reward for trades"
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
        pt = util_lst[0] 
        pt1 = util_lst[1]
        ptn = util_lst[2]
        at = util_lst[3]
        n_trades = util_lst[4]
        
        reward = 0

        '''
        ratio = (p_val/bh_val)
        
        if ratio <= 1:
            reward = max(0,profit)
        else:
            reward = profit
        '''
        
        reward = profit*0.1 # scaling to avoid the profit reward from overtaking the final reward
        if last:
            if n_trades == 0:
                reward = -10000 # buy hold
            else:
                p_val = agent.balance + agent.inventory_value # portfolio value
                bh_val = pt*agent.n_budget # buy hold value
                reward = p_val-bh_val
        return reward
    
    
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
        
        penalty = -1000 # -10 # -1000000
        #trades_threshold = 3 # at least howmany sales (buys not counted)!
        power = 0
        hold_scale = 10 #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        trade_scale = 14
        trade_cost = 3 #2.5 #7# 5 # transaction cost
        #hold_bonus = 1 #2.5 # 1.5
        reward = 0
        #if n_trades != 0:
            
        prob = at_prob[at]**0.2
        if not terminate: 
            '''
            first if statement can be extended to all undesired strategies, 
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
            profit_adj = profit - trade_cost
            if at == 0:
                # hold position; 
                n_invent = len(agent.inventory) # stocks in inventory
                hold_penalty = (-np.exp((n_holds)/l*hold_scale)+1)
                
                if n_invent != 0:
                    # in case stock is held; reward growth
                    #reward = max((pt-pt1)*n_invent*ratio,0) + hold_penalty # notice if n_invent = 0, this equals zero
                    #reward = (pt-pt1)*n_invent*ratio + hold_penalty # notice if n_invent = 0, this equals zero
                    #reward = (pt-pt1)*n_invent*ratio*hold_bonus + hold_penalty # notice if n_invent = 0, this equals zero
                    reward = ((pt-pt1)*n_invent + hold_penalty)*prob # notice if n_invent = 0, this equals zero

                else:
                    # in case stock is NOT held; introduce an opportunity cost or reward
                    #reward = max(-1*(pt-pt1)*ratio,0)
                    #reward = -1*(pt-pt1)*ratio
                    #reward = -1*(pt-pt1)*ratio*hold_bonus
                    reward = -1*(pt-pt1)*prob

                
            elif at == 1:
                if impossible:  
                    # buy action while we already had stock, IMPOSSIBLE
                    reward = penalty*prob
                else:
                    # buy; reward the conjugate profit 
                    #reward = profit*ratio
                    #reward = profit*ratio+(-np.exp((n_trades)/l*trade_scale)+1)  # scaling to avoid the profit reward from overtaking the final reward
                    # ^reward of v6_tradePen
                    reward = profit_adj*prob
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
                    #reward = profit*ratio
                    #reward = profit*ratio+(-np.exp((n_trades)/l*trade_scale)+1) # scaling to avoid the profit reward from overtaking the final reward
                    reward = profit_adj*prob
                    #reward = max(profit_adj*ratio,0)
            '''
            # TESTINGGGG ================
            if reward != penalty:
                # TODO REMOVE
                reward = max(0,reward)
            '''

        else:
            reward = -10000
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
        
        penalty = -10000 # -10 # try to keep this equal to that in R7!
        hold_scale = 10
        #self.r_util[0] += 1
        
        reward = 0 
        prob = at_prob[at]**0.2
        if at == 1 or at == 2:
            # a trade 
            if impossible:
                reward = penalty*prob
            else:
                reward = -1/20*penalty*prob #-1*penalty # positive reward
        else:
            hold_penalty = min((-np.exp((n_holds-100)/l*hold_scale)+1),0)
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
        at_prob = util_lst[4] # action probs 
        n_trades = util_lst[5] # total number of trades during run
        n_holds = util_lst[6] # concurrent holds, resets after a sell/buy
        impossible = util_lst[7] # invalid action 
        l = util_lst[8] # length of data
        terminate = util_lst[9]

        hold_scale = 10 #10 # higher means heavier penalty, default at 10 = -35 at n_hold = 800
        reward = 0
        #if n_trades != 0:
        if not terminate: 
            if at == 1 or at == 2:
                reward = 10
            else:
                hold_penalty = (-np.exp((n_holds)/l*hold_scale)+1)
                reward = 1 + hold_penalty # notice if n_invent = 0, this equals zero

        else:
            reward = -10000
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
        '''
        pvalue = agent.inventory_value+agent.balance # portfolio value
        pvalue = (pvalue-agent.n_budget*data[t])/(agent.n_budget*data[t]) # standardize
        append = [agent.balance/agent.budget,agent.inventory_value/agent.budget,pvalue]
        '''
        '''
        pvalue = agent.inventory_value+agent.balance # portfolio value
        pvalue_norm = (pvalue-agent.n_budget*data[t])/(agent.n_budget*data[t]) # standardize
        #balance_norm = (agent.balance-data[t])/pvalue
        #invent_norm =  (agent.inventory_value-data[t])/pvalue
        balance_norm = float(agent.balance>data[t])
        invent_norm =  float(agent.inventory_value>data[t])
        nholds_norm = n_holds/(l-window) # time duration of current hold position, resets at buy/sell
        holding = float(len(agent.inventory)) # binary, whether or not we have a stock
        ntrades_norm = n_trades/(l-window) 
        if not bool(agent.inventory):
            # no stock sold yet so no sell price
            bought_price = 0
            sold_price = agent.inventory_conj[0]
        else:
            # no stock held so no buy price 
            bought_price = agent.inventory[0]
            sold_price = 0
        
        buy_norm = (bought_price-data[t])/data[t]
        sell_norm = (sold_price-data[t])/data[t]
        append = [balance_norm,invent_norm,pvalue_norm, 
                  nholds_norm, holding, ntrades_norm,
                  buy_norm, sell_norm]
        
        '''
        balance_norm = (agent.balance-data[t])/data[t]
        nholds_norm = n_holds/(l-window) # time duration of current hold position, resets at buy/sell
        holding = float(len(agent.inventory)) # binary, whether or not we have a stock
        ntrades_norm = n_trades/(l-window) 
        if not bool(agent.inventory):
            # no stock sold yet so no sell price
            bought_price = 0
            sold_price = agent.inventory_conj[0]
        else:
            # no stock held so no buy price 
            bought_price = agent.inventory[0]
            sold_price = 0
        
        buy_norm = (bought_price-data[t])/data[t]
        sell_norm = (sold_price-data[t])/data[t]
        append = [balance_norm, nholds_norm, holding, ntrades_norm,
                  buy_norm, sell_norm]
        state = np.append(state,append) # TODO, maybe clip these to max of 1?
        
        state = np.expand_dims(state,axis = 0)
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
        
        if on and episode < 100 and action == 0:
            prob = utils[0]
            price = utils[1]
            
            action = choice(range(3), p = [1-2*prob, prob, prob]) #[2/3, 1/6, 1/6]
            if action == 1:                
                if len(agent.inventory) == 0 and agent.balance > price:
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
    
    
    def handle_action(agent, stats, action, data, t, flags):
        # unpack
        use_terminateFunc = flags[0]
        terminateFunc_on = flags[1]
        
        # initialize
        profit = 0 
        change = 0 
        impossible = False

        
        if action == 0:
            stats.n_holds += 1

        elif action == 1:
            stats.n_1or2 += 1
            if agent.balance > data[t] and len(agent.inventory) == 0: #max one stock 
                # BUYING stock, only if there is balance though
                agent.inventory.append(data[t])
                sold_price = agent.inventory_conj.pop(0)
                
                profit = sold_price - data[t]
                
                change = -data[t]
                stats.buy_ind.append(t)
                stats.n_trades += 1
                stats.n_holds = 0 # reset counter
            else:
                impossible = True
                stats.n_impossible += 1
                stats.n_holds += 1 # effectively no buy is a hold
                if not use_terminateFunc:
                    terminate = True
                    term_msg = "impossibles"
            
        elif action == 2:
            stats.n_1or2 += 1
            if len(agent.inventory) > 0: 
                # SELLING stock, only if there are stocks held

                bought_price = agent.inventory.pop(0)
                agent.inventory_conj.append(data[t])
                
                profit = data[t] - bought_price 

                change = data[t]
                stats.sell_ind.append(t)
                stats.n_trades += 1
                stats.n_holds = 0 # reset counter
            else:
                impossible = True
                stats.n_impossible += 1
                stats.n_holds += 1 # effectively no sell is a hold
                if not use_terminateFunc:
                    terminate = True
                    term_msg = "impossibles"
        
        # update and check termination condition
        agent.update_balance(change)
        agent.update_inventory(data[t])
        if use_terminateFunc:
            utils_term = [stats.n_impossible, np.min(data[(t+1):])]
            terminate, term_msg = agent.check_threshold(utils_term, terminateFunc_on= terminateFunc_on)
    
        return action, profit, impossible, terminate, term_msg

#%% Statistics container
class Statistics:
    '''
    This class contains all counters, containers etc. needed to keep track 
    of performance and other useful statistics and helps avoid cluttering of 
    the code
    '''
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        
    def reset_episode(self):
        '''
        Function that resets all relevant counters and containers
        '''
        self.total_reward = 0 # total profit resets every epsiode 
        self.n_trades = 0 
        self.n_impossible = 0 
        self.n_holds = 0 
        self.n_1or2 = 1 # 1 not zero because we cant have division by zero 
        
        self.profits = []
        self.balances = []
        self.rewards = []
        self.inventories = [] # inventory value (only stocks)
        self.actor_local_losses = []
        self.buy_ind = []
        self.sell_ind = []
        self.growth =[]
        self.compete = []
        self.trades_list = []
        self.actions = []
            
    def reset_all(self,growth_buyhold: np.array):
        self.growth_buyhold = growth_buyhold.tolist() # used later
        self.totalReward_list = []
        self.lastLosses_list=[]
        self.impossible_list = []
        self.tradeRatio_list = []
        self.everyProfit_dct = {}
        self.everyBalance_dct = {}
        self.everyReward_dct = {}
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
        '''
        todo include actions?
        '''
    
    def collect_iteration(self,agent,utils):
        # unpack
        profit = utils[0]
        reward = utils[1]
        actor_local_loss = utils[2]
        action = utils[3]
        
        # append
        self.balances.append(agent.balance)
        self.inventories.append(agent.inventory_value)
        self.profits.append(profit)
        self.rewards.append(reward)
        self.actor_local_losses.append(float(actor_local_loss))
        self.actions.append(action)
        
    def collect_episode(self,agent,episode, utils):
        self.growth = (np.array(self.balances)+np.array(self.inventories)-agent.budget).tolist() # 
        self.compete = (np.array(self.growth)-np.array(self.growth_buyhold)).tolist() # compete vs buyhold

        self.totalReward_list.append(self.total_reward)
        self.lastLosses_list.append(self.actor_local_losses[-1])
        self.impossible_list.append(self.n_impossible)
        self.trades_list.append(self.n_impossible)
        self.tradeRatio_list.append(self.n_impossible/self.n_1or2)
        
        episode_name = "e{}".format(episode)
        self.everyProfit_dct[episode_name] = self.profits
        self.everyBalance_dct[episode_name] = self.balances
        self.everyReward_dct[episode_name] = self.rewards
        self.everyInventory_dct[episode_name] = self.inventories
        self.everyGrowth_dct[episode_name] = self.growth
        self.everyCompete_dct[episode_name] = self.compete
        self.everyLoss_dct[episode_name] = self.actor_local_losses


    '''
    ============= LOAD/SAVE RELATED =====================
    '''
    def save_statistics(self, episode: int):
        attr_dct = self.__dict__ 
        with open('./{0}/e{1}/e{1}_statistics.json'.format(self.checkpoint_dir,episode), 'w') as fp:
            json.dump(attr_dct, fp)
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
                          title_text = "E{3} final profit RL: {0} vs buyhold: {1}, difference = {2}| impossible/trades ={4}/{5}={6}".format(round(self.growth[-1],2),
                                                                                                                    round(self.growth_buyhold[-1],2),
                                                                                                                    round((self.growth[-1]-self.growth_buyhold[-1]),2),episode,
                                                                                                                    self.trades_list[-1]-1,
                                                                                                                    self.impossible_list[-1],
                                                                                                                    round(self.tradeRatio_list[-1],2)))
        # buy/sell traces
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
        # growth traces
        fig.add_trace(pgo.Scatter(x=x, y=np.array(self.growth), marker_color = "red",
                            mode='lines',
                            name='growth-RL',
                            legendgroup = '2'))
        fig.add_trace(pgo.Scatter(x=x, y=np.array(self.growth_buyhold), marker_color = "green",
                            mode='lines',
                            name='growth-buyhold',
                            legendgroup = '2'))
        #
        
        if show_figs:
            fig.show(render = "browser")
        fig.write_html("./{0}/results/e{1}_trades.html".format(self.checkpoint_dir,episode))
        fig.data = [] # reset traces