import numpy as np
import matplotlib.pyplot as plt
import os
import DQN
import Snake

if __name__ == "__main__":
    env = Snake.Game(13)
    action_dim = env.action_dim
    state_dim = env.state_dim

    chickpoint = "./checkpoint/comple.pt"

    agent = DQN.Agent(state_dim, action_dim, dueling=True, prioritized=True, noisy=True)
    if os.path.exists(chickpoint):
        agent.load(chickpoint)
    
    r_log = []

    for ep in range(10):
        s = env.reset()
        while not env.is_end:
            title = f'episod: {ep}    score: {env.score}'
            env.set_title(title)
            env.render()

            a = agent.choose_action(s, False)

            s_, s_next, r, done = env.step(a)
            
            s = s_next

            env.wait()
        r_log.append(env.score)
    env.quit()
    
    x = np.linspace(1, len(r_log), len(r_log))
    plt.plot(x, r_log)
    plt.show()