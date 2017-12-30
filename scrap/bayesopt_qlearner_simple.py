from tris.rules import Match
from tris.agents import RandomAgent, QLearningAgent
import numpy as np
from GPyOpt.methods import BayesianOptimization
from tqdm import tqdm

bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0,1)},
    {'name': 'discount', 'type': 'continuous', 'domain': (0, 1)},
    {'name': 'start_temperature', 'type': 'continuous', 'domain': (1,1000)},
    {'name': 'cooling_down_rate', 'type': 'continuous', 'domain': (1, 2)},
]

def model_parameters_efficiency(
        learning_rate,
        discount,
        start_temperature,
        cooling_down_rate,
):
    # initialize agents & data structures
    random_agent = RandomAgent()
    deepq_learner = QLearningAgent(
            discount=discount,
            learning_rate=learning_rate,
            temperature=start_temperature)
    # train
    for _ in tqdm(range(5000)):
        match = Match(deepq_learner, random_agent)
        match.play()
    for _ in tqdm(range(5000)):
        if not _%100:
            deepq_learner.temperature = deepq_learner.temperature / cooling_down_rate
        match = Match(deepq_learner, deepq_learner.spawn_self_player())
        match.play()
    all_results = []
    for _ in range(500):
        match = Match(deepq_learner, random_agent)
        all_results.append(match.play())
    return np.mean(all_results)

def function_wrapper(function):
    def wrapped_fcn(x):
        x = np.atleast_2d(x)
        results = np.zeros((x.shape[0],1))
        for j in range(x.shape[0]):
            results[j] = function(*x[j,:])
        return results
    return wrapped_fcn

model_parameters_efficiency = function_wrapper(model_parameters_efficiency)



myBopt = BayesianOptimization(f=model_parameters_efficiency,  # Objective function
                              domain=bounds,  # Box-constraints of the problem
                              # initial_design_numdata = 5,   # Number data initial design
                              acquisition_type='EI',  # Expected Improvement
                              # exact_feval = True
                              maximize=True,
                              #initial_design_numdata=1,
                              batch_size=1,
                              # X=X,
                              )
n_hours = 1/12
n_minutes = 60*n_hours
max_iter = 20  ## maximum number of iterations
max_time = int(60 * n_minutes)  ## maximum allowed time
eps = 1e-6  ## tolerance, max distance between consicutive evaluations.

myBopt.run_optimization(
        max_iter=9999,
        max_time=max_time,
        eps=eps)

myBopt.plot_convergence()