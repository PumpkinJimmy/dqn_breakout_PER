from collections import deque
import os
import random
from tqdm import tqdm
from glob import glob
import sys

import torch
import visdom

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 200_000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10000
WARM_STEPS = 0
MAX_STEPS = 1_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
cont = False
if os.path.exists(SAVE_PREFIX):
    choice = input("./models文件夹已存在，是否继续训练？[y/n]")
    if choice == 'y':
        restore = glob(os.path.join('models', 'model_*'))[-1]
        print(f"从检查点：{restore}恢复")
        EPS_START = EPS_END
        cont = True
    else:
        print("结束")
        sys.exit(0)
else:
    restore = None
    os.mkdir(SAVE_PREFIX)


torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Device: {device}")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    restore=restore
)

obs_queue = deque(maxlen=5)
agent_eval = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    0, 0, 1,
    restore=restore
)
avg_reward, frames = env.evaluate(obs_queue, agent_eval)
print(f"Avg. Reward: {avg_reward:.1f}")

memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####

print("开始训练")
obs_queue: deque = deque(maxlen=5)
done = True

vis = visdom.Visdom()
win = None

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        loss = agent.learn(memory, BATCH_SIZE)
        if win is None:
            win = vis.line([loss], [step])
        else:
            vis.line([loss], [step], win=win, update='append')


    if step % TARGET_UPDATE == 0:
        # print(f"#{step}: 同步估值网络")
        
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        print(f"#{step}: 评估模型")
        
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        agent_eval = Agent(
            env.get_action_dim(),
            device,
            GAMMA,
            new_seed(),
            0, 0, 1,
            restore=os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}")
        )
        avg_reward, frames = env.evaluate(obs_queue, agent_eval, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        
        done = True
