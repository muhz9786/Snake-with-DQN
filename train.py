import Snake
import matplotlib.pyplot as plt
import DQN
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    env = Snake.Game(13)
    action_dim = env.action_dim
    state_dim = env.state_dim

    cfg = "./config.json"

    chickpoint = "./checkpoint/comple.pt"

    agent = DQN.Agent(state_dim, action_dim, cuda=True, dueling=True, prioritized=True, noisy=True)
    agent.read_config(cfg)

    if os.path.exists(chickpoint):
        agent.load(chickpoint)
    loss_log = []

    for ep in tqdm(range(2000)):
        s = env.reset()
        while not env.is_end:
            #env.title = f'episod: {ep}    scoal: {env.rewards}'
            #env.render()
            a = agent.choose_action(s)

            s_, s_next, r, done = env.step(a)

            agent.store(s, a, r, s_, done)

            loss = agent.update()
            if loss is not None:
                loss_log.append(loss)
            s = s_next
    env.quit()

    agent.save(chickpoint)
    
    x = np.linspace(1, len(loss_log), len(loss_log))
    plt.plot(x, loss_log)
    plt.show()