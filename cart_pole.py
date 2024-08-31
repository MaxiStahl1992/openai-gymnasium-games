import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Load the trained PPO model
model = PPO.load("Training/Saved_Models/PPO_Model_CartPole/best_model.zip")

# Initialize Pygame for handling inputs
pygame.init()
window = pygame.display.set_mode((640, 480))

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()


# Run the environment for a few episodes
episodes = 30
for episode in range(1, episodes+1):
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Pygame rendering
        window.fill((0, 0, 0))  # Clear screen with black
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        window.blit(frame_surface, (0, 0))
        pygame.display.flip()  # Update the display

        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        score += reward

    print("Episode: {} Score: {}".format(episode, score))

# Clean up
env.close()
pygame.quit()

