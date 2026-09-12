import gymnasium as gym
from stable_baselines3 import SAC
import time


#make a custom reward wrapper for the cheetah environment
class CustomCheetahRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, action):
        # Let the base environment run the step
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        anfronTip = obs[1] #angle of the front tip

        # Calculate the custom reward
        torso_z = obs[0]

        # Penalize sinking into the floor / dragging chest
        height_penalty = 0.0
        if torso_z < -0.15:  
            height_penalty = -5.0 * ((-0.15 - torso_z) ** 2)

        tilt = max(0.0, abs(anfronTip) - 0.5 )
        tiltingpenalty = -5 * (tilt ** 2)  # Penalty for tilting beyond a certain angle 

        pen = 0.0
        if abs(anfronTip) > 1.6:  # past 90 degrees
            terminated = True
            pen -= 50.0
        
        
        custom_reward = base_reward + tiltingpenalty + pen + height_penalty

        return obs, float(custom_reward), terminated, truncated, info



# Create the Cheetah environment
env = gym.make("HalfCheetah-v5")
env = CustomCheetahRewardWrapper(env)

#setup the model algorithm
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=200_000,        
    batch_size=128,             
    train_freq=(64, "step"),   
    gradient_steps=64,         
    learning_starts=5_000,
    device="cpu",
    verbose=1,
)

#Start the training

print("Starting training...")
model.learn(total_timesteps=200_000)
print("Training completed.")

#Save the model to see it whenever you want
model.save("SAC_HalfCheetah_CustomReward")

#to see the actual result of the training

test_env = gym.make("HalfCheetah-v5", render_mode="human")
test_env = CustomCheetahRewardWrapper(test_env)

for ep in range(5):
    obs, _ = test_env.reset()
    done = False
    ep_reward = 0.0
    print(f"Starting Episode {ep + 1}...")

    while not done:
        action, _ = model.predict(obs, deterministic=True,)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        ep_reward += reward
        done = terminated or truncated

        time.sleep(0.03)

    print(f"Episode {ep + 1} finished! (Reward: {ep_reward:.1f})")
    time.sleep(0.5)

test_env.close()
