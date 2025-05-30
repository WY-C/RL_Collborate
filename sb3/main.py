from env import GridEnv, GridEnvGUI
from Agent import DQNAgent, ReplayAgent
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import time


#DQN, ReplayBuffer
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []  # 각 에피소드의 총 보상 저장
        self.episode_lengths = []  # 각 에피소드의 길이 저장
        self.episode_count = 0
        self.printing = False
        self.avg_reward = []
        self.avg_timestep = []
    def _on_step(self) -> bool:
        """
        매 스텝마다 호출됩니다. 에피소드 종료 시 보상 기록.
        """
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    
                    # 에피소드 종료 시 보상 및 길이 저장
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    self.episode_count += 1
                    #print(f"에피소드: {self.episode_count}, Reward: {info['episode']['r']:.2f}, Timestep: {info['episode']['l']:.2f}")
                    if self.episode_count % 50 == 0:
                        avg_reward = np.mean(self.episode_rewards[-50:])
                        avg_timestep = np.mean(self.episode_lengths[-50:])

                        self.avg_reward.append(round(float(avg_reward), 3))

                        self.avg_timestep.append(round(float(avg_timestep), 3))
                        #print(wandb.run.summary)
                        wandb.log({
                            "avg_reward": avg_reward,
                            "avg_timestep": avg_timestep,
                            "epsilon": self.model.exploration_rate
                        }, step=self.episode_count)
                        if not (np.isnan(avg_reward) or np.isnan(avg_timestep)):
                            print(f"에피소드: {self.episode_count}, Reward: {avg_reward:.2f}, Timestep: {avg_timestep:.2f}, Step: {self.num_timesteps}, Epsilon: {self.model.exploration_rate:.3f}")
                        else:
                            print(f"에피소드: {self.episode_count}, 평균 계산 불가(데이터 부족)")
                    # if(len(self.avg_reward) >= 100 and len(self.avg_timestep) >= 100 and self.printing == False):
                    #     print(self.avg_reward)
                    #     print(self.avg_timestep)
                    #     self.printing = True
                    #     return False
                    
        
        return True  # 학습 계속 진행

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "GridEnv" 
}

run = wandb.init(
    project="grid_env_dqn",
    name = "SB3_-new_strategy_epsilon_5100",
    monitor_gym=False, 
    config=config,
    sync_tensorboard=False,
    save_code=False,
)

raw_env = GridEnv(5, normalize_obs=True)
env = Monitor(raw_env)


model = DQN("MlpPolicy", env, verbose=0)
try:
    model = DQN.load("1", env=env)
    print("Model loaded successfully.")
except:
    print("Model not found, starting training from scratch.")

reward_logger = RewardLoggerCallback(verbose=0)


num_episodes = 100000000  # 원하는 총 에피소드 수
for episode in range(num_episodes):
    #print(episode)
    env.render()
    #print(model.exploration_rate, episode)
    model.learn(total_timesteps=1000000,
                callback=[
                    WandbCallback(
                        gradient_save_freq=0,
                        model_save_path=None,
                        verbose=0
                        ),
                    reward_logger],
                reset_num_timesteps=False)
    if(episode + 1) % 1000 == 0:
        print(episode)
        model.save("dqn_grid")
model.save("dqn_grid")


###### render code
# raw_env = GridEnv(5,render_mode="human", normalize_obs=True)
# env = Monitor(raw_env)
# model = DQN("MlpPolicy", env, verbose=0)
# try:
#     model = DQN.load("model/sb3/0.1_-0.3_-0.3_60650", env=env)
#     print("Model loaded successfully.")
# except:
#     print("Model not found, starting training from scratch.")

# reward_logger = RewardLoggerCallback(verbose=0)
# num_episodes = 3000  # 원하는 총 에피소드 수
# for episode in range(num_episodes):
#     obs, _ = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action, _ = model.predict(obs, deterministic=False)
#         obs, reward, done, _, info = env.step(action)
#         # 필요하다면 logger에 info 전달
#         if reward_logger is not None:
#             reward_logger.locals = {"infos": [info]}
#             reward_logger._on_step()
#     if (episode+1) % 1000 == 0:
#         print(f"Episode {episode + 1} finished.")
#         model.save("dqn_grid")
# model.save("dqn_grid")

"""
env = GridEnv(5)
Agent1 = DQNAgent(state_size=8, action_size=16)
#Agent2 = ReplayAgent(state_size=8, action_size=4)

tot_reward = 0
tot_timestep = 0
printing = 100
#action = [0, 0]
#reward = [0, 0]


#save, load시 target model, epsilon도 같이.
try:
    Agent1.load_model("model/10000000_Agent1_mk4.pth")
    #Agent2.load_model("model/10000000_Agent2_mk4.pth")
    print("Model loaded successfully.")
except:
    print("Model not found, starting training from scratch.")
for i in range(100000000):
    state, _ = env.reset()
    terminated = False
    while not terminated:
        #다른 observasation
        #reward 는 r1,r2를 더하고
        #Action은 [0] + [1] concat
        #sb3 vs custom model
        #다른 env와 DQN 비교하기
        #DQN, PPO, SAC, MPO

        #DDQN, DQN, PER, Replay 비교하기

        #CHI 2025 재밌는 논문 몇 개 읽어오기
        action = Agent1.choose_action(state)
        #action[1] = Agent2.choose_action(state)
        next_state, reward, terminated, _, _ = env.step(action) #truncated 없음
        #print(state)
        Agent1.replay_buffer.add(state, action, reward, next_state, terminated, 1.0) #replay는 안쓰고, per은 씀.
        #Agent2.replay_buffer.add(state, action[1], reward[1], next_state, terminated)#, 1.0)

        #reward 연산 해결하기
        tot_reward += reward
        #epsilon 수정하기
        #env.render()
        #최대 tick 출력하기
        state = next_state.copy()
        Agent1.update()
        #Agent2.update()
        

    if terminated:
        #print(tot_reward, env.get_timestep())
        timestep = env.get_timestep()
        tot_timestep += timestep

    if (i + 1) % 100 == 0:
        Agent1.update_target_model() 
    if terminated and (i + 1) % printing == 0:
        DQNAgent.epsilon = max(DQNAgent.epsilon_min, DQNAgent.epsilon * DQNAgent.epsilon_decay)  
        print(f"Episode {i+1} finished with reward: {tot_reward/printing:.2f}, timestep: {tot_timestep/printing:.3f}, Epsilon: {DQNAgent.epsilon:.3f}")
        tot_timestep = 0
        tot_reward = 0
    
    if (i + 1) % 1000 == 0:
        Agent1.save_model(f"model/{i+1}_Agent1_SB.pth")
        #Agent2.save_model(f"model/{i+1}_Agent2_DDQN_Replay.pth")
        print(f"Model saved at episode {i+1}")
    
        

"""