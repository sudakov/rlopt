import numpy as np
import random
import json
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym
from ortools.linear_solver import pywraplp
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict, Optional, TYPE_CHECKING
from ray.rllib.policy import Policy

T = 3
upperBound = 1

path = 'data.json'


class PrepData:

  def __init__(self, json_path):
    self.json_path = json_path
    self.vars = []
    self.coefs = []
    self.upperBound = None

    with open(self.json_path) as file:
      self.data = json.load(file)

  
  def set_goal_coef(self):
    ex = {}
    T = self.data["Periods"]
    years = ['year' + '_' + str(i) for i in range(1, T + 1)]
    self.upperBound = self.data["The maximum number of basketball courts in the region per year"]

    w_dict = {i: {
        j: self.data["Regions"][i]["Type of basketball court"][j]["Priority"]
        for
        j in list(self.data["Regions"][i]["Type of basketball court"].keys())
    } for i in
        list(self.data["Regions"].keys())}

    p = {i: self.data["Regions"][i]["Rank"] for i in
         list(self.data["Regions"].keys())}

    # Sorting
    p = dict(sorted(p.items(), key=lambda x: int(x[0][x[0].index('_') + 1])))
    w_dict = dict(sorted({i: dict(sorted(w_dict[i].items(), key=lambda x: int(x[0][x[0].index('_') + 1]))) for i in
                          list(w_dict.keys())}.items(),
                         key=lambda x: int(x[0][x[0].index('_') + 1])))
    
    # Creating vars
    for t in range(1, T + 1):
      for key in list(w_dict.keys()):
        for k in list(w_dict[key].keys()):
          self.vars.append(key + '.' + k + '.' + years[t - 1])
          self.coefs.append((p[key] + w_dict[key][k]) * ((T + 1 - t) / T))
          ex[key + '.' + k + '.' + years[t - 1]] = (p[key] + w_dict[key][k]) * ((T + 1 - t) / T)
    
    return ex

  def get_constraints(self):
    constraits = []
    numberOfRegs, typesOfPlaces = len(self.data["Regions"]), len(self.data["Types of basketball courts"])

    maxNumberCourts = self.data["The maximum number of basketball courts all types for each region"]

    cost = {
        i: {j: self.data["Regions"][i]["Type of basketball court"][j]["cost"]
            for
            j in list(self.data["Regions"][i]["Type of basketball court"].keys())
            } for i in
        list(self.data["Regions"].keys())}

    b = {i: self.data["Regions"][i]["Number of players"] for i in
         list(self.data["Regions"].keys())}
    
    e = {i: self.data["Types of basketball courts"][i]["Capacity"] for i in list(self.data["Types of basketball courts"].keys())}

    u = {
        i: {j: self.data["Regions"][i]["Type of basketball court"][j]["Regional costs"]
            for
            j in list(self.data["Regions"][i]["Type of basketball court"].keys())
            } for i in
        list(self.data["Regions"].keys())}
    
    a = {i: self.data["Regions"][i]["Regs budget"] for i in
         list(self.data["Regions"].keys())}
    
    cost = dict(sorted({i: dict(sorted(cost[i].items(), key=lambda x: int(x[0][x[0].index('_') + 1]))) for i in
                        list(cost.keys())}.items(),
                       key=lambda x: int(x[0][x[0].index('_') + 1])))
    
    b = dict(sorted(b.items(), key=lambda x: int(x[0][x[0].index('_') + 1])))
    e = dict(sorted(e.items(), key=lambda x: int(x[0][x[0].index('_') + 1])))
    a = dict(sorted(a.items(), key=lambda x: int(x[0][x[0].index('_') + 1])))
    u = dict(sorted({i: dict(sorted(u[i].items(), key=lambda x: int(x[0][x[0].index('_') + 1]))) for i in
                        list(u.keys())}.items(),
                       key=lambda x: int(x[0][x[0].index('_') + 1])))
    
    cost = [[*list((list(cost[key].values())))] for key in list(cost.keys())]
    b = [b[i] for i in list(b.keys())]
    e = [e[i] for i in list(e.keys())]
    u = [[*list((list(u[key].values())))] for key in list(u.keys())]
    a = [a[i] for i in list(a.keys())]

    x, X = {i: 0 for i in self.vars}, []

    totalProjPerYear = self.data["Limit on the number of projects per year"]
    totalBudget = self.data["Total budget"]

    # Ограничения на количесвто объектов в год
    for i in range(0, len(self.vars), numberOfRegs * typesOfPlaces):
      for j in self.vars[i:i + (numberOfRegs * typesOfPlaces)]:
        x[j] = 1
      X.append(x)
      constraits.append(totalProjPerYear)
      
      x = {i: 0 for i in self.vars}

    # Ограничения на стоиммость объектов за T лет
    for i in range(0, len(self.vars), typesOfPlaces):
      l = self.vars[i: i + typesOfPlaces]
      for j in range(len(l)):
        x[l[j]] = cost[int(i / typesOfPlaces) % numberOfRegs][j]
    
    X.append(x)
    constraits.append(totalBudget)
    x = {i: 0 for i in self.vars}

    # Ограничение на максимальное количесвто площадок в каждом регионе
    vars = sorted(self.vars)
    for i in range(0, len(vars), T * typesOfPlaces):
      for j in vars[i: i + (T * typesOfPlaces)]:
        x[j] = 1
      X.append(x)
      constraits.append(maxNumberCourts)

      x = {i: 0 for i in self.vars}
    
    # Ограничение на колиество баскетболистов
    for i in range(0, len(vars), T * typesOfPlaces):
      v = vars[i: i + (T * typesOfPlaces)]
      for j in range(0, len(v), T):
        for k in v[j:j + T]:
          x[k] = e[int(j / T) % T]

      X.append(x)
      constraits.append(b[int(i / (T * typesOfPlaces))])
      x = {i: 0 for i in self.vars}
    
    # Ограничение на затраты регионов
    for i in range(0, len(vars), T * typesOfPlaces):
      v = vars[i: i + (T * typesOfPlaces)]
      for j in range(0, len(v), T):
        for k in v[j:j + T]:
          x[k] = u[int(i / (T * typesOfPlaces) % (T * typesOfPlaces))][int(j / T)]

      X.append(x)
      constraits.append(a[int(i / (T * typesOfPlaces))])
      x = {i: 0 for i in self.vars}
    
    result = [list(d.values()) for d in X]

      
    return X, constraits

obj = PrepData(path)
goal_coef = obj.set_goal_coef()
constraits = obj.get_constraints()[0]
bounds = obj.get_constraints()[1]
p = []
for k in sorted(list(goal_coef.keys())):
  p.append(goal_coef[k])
p = np.array(p)

c = []
for d in constraits:
  r = []
  for k in sorted(list(d.keys())):
    r.append(d[k])
  c.append(r)
c = np.array(c)

b = np.array(bounds).astype(np.float64)

ubound = upperBound
m = len(p)
n = len(c)


solver = pywraplp.Solver.CreateSolver('SCIP')
x = {}
for j in range(m):
    x[j] = solver.IntVar(0, ubound, f"x[{j}]")
for i in range(n):
    constraint_expr = [c[i,j] * x[j] for j in range(m)]
    solver.Add(sum(constraint_expr) <= b[i])
obj_expr = [p[j] * x[j] for j in range(m)]
solver.Maximize(solver.Sum(obj_expr))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in range(m):
        print(x[j].name(), ' = ', x[j].solution_value())
    print()
    print(f"Problem solved in {solver.wall_time()} milliseconds")
    print(f"Problem solved in {solver.iterations()} iterations")
    print(f"Problem solved in {solver.nodes()} branch-and-bound nodes")
else:
    print("The problem does not have an optimal solution.")


ray.shutdown()
ray.init()

class SampleCallback(DefaultCallbacks):
    
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        self.best_reward = -666666666
        self.best_actions = []
        self.legacy_callbacks = legacy_callbacks_dict or {}
        
    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: Episode,
            agent_id: AgentID, policy_id: PolicyID,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        
        sample_obj = original_batches[agent_id][1]
        rewards = sample_obj.columns(['rewards'])[0]
        total_reward = np.sum(rewards)
        actions = sample_obj.columns(['actions'])[0]
        
        if total_reward > self.best_reward and len(actions[rewards >= 0])==m:
            # print('ku', total_reward, rewards, actions)
            actions = actions[rewards >= 0]
            total_reward = np.sum(actions * p)
            self.best_actions = actions
            self.best_reward = total_reward
            episode.hist_data["best_actions"] = [actions]
            episode.hist_data["best_reward"] = [total_reward]
            # print('ku_', total_reward, actions)
            
config =  ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["framework"] = "torch"
config["env_config"] = {}
# config['kl_coeff'] = 0.0
config["callbacks"] = SampleCallback
config["log_level"] = "ERROR"

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(upperBound + 1)
        self.observation_space = gym.spaces.Dict({
            'rem': gym.spaces.Box(low=np.zeros(len(b)), high=b, dtype=np.float64), 
            'j': gym.spaces.Discrete(c.shape[1])})
        self.state = {'rem': np.array(b), 'j': 0}
        self.done = False

    def reset(self):
        self.state = {'rem': np.array(b), 'j': 0}
        self.done = False
        return self.state

    def step(self, action):
        #print('current state:', self.state)   
        #print('action taken:', action)
        j = self.state['j']
        rem = self.state['rem'] - c[:,j] * action
        if np.any(rem < 0):
            self.reward = -1
        else:
            self.reward = action * p[j]
            self.state['rem'] = rem
            j += 1
            if j == c.shape[1]: 
                self.done = True
            else:
                self.state['j'] = j
                self.done = False
            
        # print('reward:', self.reward)
        # print('next state:', self.state)
            
        return self.state, self.reward, self.done, {}

agent3 = ppo.PPOTrainer(config=config, env=MyEnv)

best_g = 0
best_actions = []
for i in range(21):
   # Perform one iteration of training the policy with PPO
   result = agent3.train()
   if 'best_reward' in result['hist_stats'] and len(result['hist_stats']['best_reward']) > 0 and \
       (best_g < result['hist_stats']['best_reward'][-1] or best_actions == []):
       best_g = result['hist_stats']['best_reward'][-1]
       best_actions = result['hist_stats']['best_actions'][-1]
   if i % 10 == 0:
       #print(pretty_print(result))
       print('i: ', i)
       print('mean episode length:', result['episode_len_mean'])
       print('max episode reward:', result['episode_reward_max'])
       print('mean episode reward:', result['episode_reward_mean'])
       print('min episode reward:', result['episode_reward_min'])
       print('total episodes:', result['episodes_total'])
       print('solution:', best_g, best_actions)
          
       checkpoint = agent3.save()
       # print("checkpoint saved at", checkpoint)


"""env = MyEnv(config)
state = env.reset()
g = 0
done = False
reward = 0
while not done:
  action = agent3.compute_action(state, explore = False)
  state, reward, done, info = env.step(action)
  print(f"j = {state['j']} action = {action} reward = {reward}")
  g += reward
print(g)"""

for j, x in enumerate(best_actions):
    print(f"j = {j} action = {x} reward = {x*p[j]}")
print(best_g)