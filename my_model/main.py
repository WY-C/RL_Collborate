from env import GridEnv, GridEnvGUI
from Agent import DQNAgent, DQN_Replay, DDQN_Replay
from matplotlib import pyplot as plt
import wandb
#from wandb.integration.sb3 import WandbCallback
import numpy as np

env = GridEnv(5, normalize_obs=True)
#env = GridEnv(5, normalize_obs=True, render_mode="human")
Agent1 = DDQN_Replay(state_size=6, action_size=4)
Agent2 = DDQN_Replay(state_size=6, action_size=4)
# Agent1 = DQNAgent(state_size=6, action_size=4)
# Agent2 = DQNAgent(state_size=6, action_size=4)
tot_reward = 0
best_reward = -99
reward_1000 = 0
tot_timestep = 0
printing = 100
time_list = []
reward_list = []
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "GridEnv" 
}

run = wandb.init(
    project="grid_env_dqn",
    name = "DDQN_Replay_soloprobsolving_learningrate/2per10000",
    monitor_gym=False, 
    config=config,
    sync_tensorboard=False,
    save_code=False,
)
#save, load시 target model, epsilon도 같이.
try:
    Agent1.load_model("/home/cau/바탕화면/grid/my_model/model/21233000_Agent1_DQN_Replay_20000.pth")
    Agent2.load_model("/home/cau/바탕화면/grid/my_model/model/23122000_Agent2_DQN_Replay_20000.pth")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model not found or error occurred: {e}.\n Starting training from scratch.")
for i in range(100000000):  
    state, _= env.reset()
    action = [0, 0]
    reward = [0, 0]
    terminated = False
    while not terminated:
        env.render()
        #DQN, PPO, SAC, MPO

        #DDQN, DQN, PER, Replay 비교하기
        action[0] = Agent1.choose_action(state[0])
        action[1] = Agent2.choose_action(state[1])
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        Agent1.replay_buffer.add(state[0], action[0], reward[0], next_state[0], terminated) 
        Agent2.replay_buffer.add(state[1], action[1], reward[1], next_state[1], terminated)

        #reward 연산 해결하기
        tot_reward += (reward[0] + reward[1])
        #epsilon 수정하기

        state = next_state.copy()
        Agent1.update()
        Agent2.update()
        

    if terminated:
        reward_1000 += tot_reward
        #print(tot_reward, env.get_timestep())
        #print(1)
        timestep = env.get_timestep()
        time_list.append(timestep)
        reward_list.append(tot_reward)
        tot_reward = 0
        DDQN_Replay.epsilon_current = max(DDQN_Replay.epsilon_min, DDQN_Replay.epsilon_start - (DDQN_Replay.epsilon_start - DDQN_Replay.epsilon_min) * (i / 20000))  # Epsilon decay
        #DQNAgent.epsilon_current = max(DQNAgent.epsilon_min, DQNAgent.epsilon_start - (DQNAgent.epsilon_start - DQNAgent.epsilon_min) * (i / 10000))  # Epsilon decay

    if (i+1) % 50 == 0 and terminated:
            avg_reward = np.mean(reward_list[-50:])
            avg_timestep = np.mean(time_list[-50:])
            #print(wandb.run.summary)
            wandb.log({
                "avg_reward": avg_reward,
                "avg_timestep": avg_timestep
            }, step=i)
            if not (np.isnan(avg_reward) or np.isnan(avg_timestep)):
                print(f"에피소드: {i+1}, Reward: {avg_reward:.2f}, Timestep: {avg_timestep:.2f}, Epsilon: {DDQN_Replay.epsilon_current:.3f}")           
                #print(f"에피소드: {i+1}, Reward: {avg_reward:.2f}, Timestep: {avg_timestep:.2f}, Epsilon: {DQNAgent.epsilon_current:.3f}")
            else:
                print(f"에피소드: {i+1}, 평균 계산 불가(데이터 부족)")
                   
    if (i + 1) % 100 == 0:
        Agent1.update_target_model() 
        Agent2.update_target_model()

    if (i+1) % 10000 == 0:
        DDQN_Replay.learning_rate /= 2
        print(f"Learning rate updated: {DDQN_Replay.learning_rate} at episode {i+1}")
    if (i + 1) % 1000 == 0:
        reward_1000 /= 1000
        if best_reward < reward_1000:
            #1000개 reward로 수정해야함
            best_reward = reward_1000
            Agent1.save_model(rf"/home/cau/바탕화면/grid/my_model/model/{i+1}_Agent1_DDQN_Replay_20000.pth")
            Agent2.save_model(rf"/home/cau/바탕화면/grid/my_model/model/{i+1}_Agent2_DDQN_Replay_20000.pth")       
            print(f"reward updated: {best_reward} at episode {i+1}")
        reward_1000 = 0
    