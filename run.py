import numpy as np
import matplotlib.pyplot as plt
from env import Env


E = Env(30, 30)
x_start = np.array([28, 27])# E.gen_random_state()
x_goal = np.array([3, 2])# E.gen_random_state()
E.set_start_state(x_start)
E.set_g_func(E.cost_score)
# E.plot(True)

sol = E.plan(x_start, x_goal)
path = sol['path']
E.plot_plan(path = path)
exit(1)

x_current = np.copy(x_start) # The real current state but unknown
actions_prev = []
P = []
count_plans = 0
while not np.all(x_current == x_goal) or E.get_max_prob() < 1.0:
    p_max = E.get_max_prob()
    print("Max probability: ", p_max)

    # Plan from current belief
    x_belief = E.sample_state_distribution()
    print("x_belief: ", x_belief, "x_real: ", x_current)
    print("Planning...")
    sol = E.plan(x_belief, x_goal)
    count_plans += 1
    if sol:
        path = sol['path']
        actions = sol['actions']
        actions_prev = np.copy(actions[1:])
    else:
        actions = actions_prev.copy()

    S = []
    while not len(S) or p_max == 1.0:
        try:
            a = actions.pop(0)
        except:
            print("End of plan.")
            break
        x_current = E.action(a, loc = x_current)
        print("a: ", a, x_current)
        P.append(E.get_max_prob())

        # Sense cell
        S = E.sense(x_current)
        print("Senses %d obstacles." % len(S))

    actions_prev = actions.copy()
        
    # Update belief map
    for s in S:
        E.update_prob(s)
    P.append(E.get_max_prob())

print("\n*** Reached goal! ***")
print("Number of plans: ", count_plans)
print("Number of steps: ", len(P) - count_plans)

plt.figure()
plt.plot(P)

E.plot(True, P)


# x = np.copy(x_start)
# for a in actions:
#     S = E.sense(x)
#     for s in S:
#         E.update_prob(s)
#     x = E.action(a, loc = x)
    
