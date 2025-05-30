from env import GridEnv, GridEnvGUI
from Agent import DQNAgent, ReplayAgent
from matplotlib import pyplot as plt
import wandb
#from wandb.integration.sb3 import WandbCallback
import numpy as np

env = GridEnv(5, normalize_obs=True)
#env = GridEnv(5, normalize_obs=True, render_mode="human")
Agent1 = ReplayAgent(state_size=8, action_size=4)
Agent2 = ReplayAgent(state_size=8, action_size=4)
# Agent1 = DQNAgent(state_size=8, action_size=4)
# Agent2 = DQNAgent(state_size=8, action_size=4)
tot_reward = 0
tot_timestep = 0
printing = 100
action = [0, 0]
reward = [0, 0]
time_list = []
reward_list = []
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "GridEnv" 
}

run = wandb.init(
    project="grid_env_dqn",
    name = "DQN_PER_with_+strategy",
    monitor_gym=False, 
    config=config,
    sync_tensorboard=False,
    save_code=False,
)
#save, load시 target model, epsilon도 같이.
try:
    Agent1.load_model("model/test/14123000_Agent1_DQN_Replay.pth")
    Agent2.load_model("model/test/14123000_Agent2_DQN_Replay.pth")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model not found or error occurred: {e}. Starting training from scratch.")
        
for i in range(100000000):
    state, _= env.reset()
    terminated = False
    while not terminated:
        env.render()
        #DQN, PPO, SAC, MPO

        #DDQN, DQN, PER, Replay 비교하기
        action[0] = Agent1.choose_action(state)
        action[1] = Agent2.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        Agent1.replay_buffer.add(state, action[0], reward[0], next_state, terminated) 
        Agent2.replay_buffer.add(state, action[1], reward[1], next_state, terminated)

        #reward 연산 해결하기
        tot_reward += (reward[0] + reward[1])
        #epsilon 수정하기

        state = next_state.copy()
        Agent1.update()
        Agent2.update()
        

    if terminated:
        #print(tot_reward, env.get_timestep())
        #print(1)
        timestep = env.get_timestep()
        time_list.append(timestep)
        reward_list.append(tot_reward)
        tot_reward = 0
        ReplayAgent.epsilon_current = max(ReplayAgent.epsilon_min, ReplayAgent.epsilon_start - (ReplayAgent.epsilon_start - ReplayAgent.epsilon_min) * (i / 5000))  # Epsilon decay
        # DQNAgent.epsilon_current = max(DQNAgent.epsilon_min, DQNAgent.epsilon_start - (DQNAgent.epsilon_start - DQNAgent.epsilon_min) * (i / 2500))  # Epsilon decay

    if (i+1) % 50 == 0 and terminated:
            avg_reward = np.mean(reward_list[-50:])
            avg_timestep = np.mean(time_list[-50:])
            #print(wandb.run.summary)
            wandb.log({
                "avg_reward": avg_reward,
                "avg_timestep": avg_timestep
            }, step=i)
            if not (np.isnan(avg_reward) or np.isnan(avg_timestep)):
                print(f"에피소드: {i+1}, Reward: {avg_reward:.2f}, Timestep: {avg_timestep:.2f}, Epsilon: {ReplayAgent.epsilon_current:.3f}")
            else:
                print(f"에피소드: {i+1}, 평균 계산 불가(데이터 부족)")
                   
    if (i + 1) % 100 == 0:
        Agent1.update_target_model() 
        Agent2.update_target_model()


    if (i + 1) % 1000 == 0:
        Agent1.save_model(f"model/{i+1}_Agent1_DQN_Replay.pth")
        Agent2.save_model(f"model/{i+1}_Agent2_DQN_Replay.pth")
        print(f"Model saved at episode {i+1}")
    