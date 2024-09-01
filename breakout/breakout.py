import gymnasium as gym
import pygame
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C

# Load the trained model
model = A2C.load("breakout/Training/Saved_Models/A2C_Breakout.zip")

# Initialize Pygame for handling inputs
pygame.init()
window = pygame.display.set_mode((640, 480))

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=73)
env = VecFrameStack(env, n_stack=4)
env.metadata['render_fps'] = 60

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

obs = env.reset()

clock = pygame.time.Clock()

episodes = 10
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        score += reward[0]

        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        window.blit(frame_surface, (0, 0))
        pygame.display.update()

        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game {episode}: Score = {score}")



# Clean up
env.close()
pygame.quit()