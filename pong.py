import gymnasium as gym
from ale_py import ALEInterface
from ale_py.roms import Pong
import pygame
import random

# Initialize Pygame for handling inputs
pygame.init()
window = pygame.display.set_mode((640, 480))

# Initialize the ALE interface and load the Pong ROM
ale = ALEInterface()
ale.loadROM(Pong)

# Create the Pong environment with a specific FPS setting
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env.metadata['render_fps'] = 30  # Set render FPS

observation, _ = env.reset()

done = False
clock = pygame.time.Clock()

# Variables to control AI difficulty
ai_speed = 0.1  # Lower speed for a worse AI (lower is slower)
ai_randomness = 0.9  # High randomness makes the AI worse (closer to 1.0 is worse)
ai_error_chance = 0.5  # Chance to make an intentional error

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Get the current state of the keyboard
    keys = pygame.key.get_pressed()

    # Define actions for moving the player's paddle
    if keys[pygame.K_UP]:
        player_action = 2  # Move paddle up
    elif keys[pygame.K_DOWN]:
        player_action = 3  # Move paddle down
    else:
        player_action = 0  # Do nothing

    # AI logic to decide its action
    ball_y = observation[35:194, 16:144, 0].mean(axis=(0, 1)).argmax()  # Approximate ball's y-position
    ai_paddle_y = observation[35:194, 144:, 0].mean(axis=(0, 1)).argmax()  # Approximate AI paddle's y-position

    if random.random() > ai_randomness:
        if ai_paddle_y < ball_y - ai_speed * 240:
            ai_action = 3  # AI paddle moves down
        elif ai_paddle_y > ball_y + ai_speed * 240:
            ai_action = 2  # AI paddle moves up
        else:
            ai_action = 0  # AI paddle does nothing
    else:
        ai_action = 0  # AI paddle does nothing (introduce randomness)

    # Introduce a chance for the AI to make an error
    if random.random() < ai_error_chance:
        ai_action = 0  # AI intentionally does nothing or takes the wrong action

    # Step the environment with the chosen action
    observation, reward, terminated, truncated, info = env.step(player_action)

    # Render the game frame to the Pygame window
    frame = env.render()
    frame = pygame.transform.rotate(pygame.surfarray.make_surface(frame), 90)  # Rotate the frame
    window.blit(pygame.transform.scale(frame, (640, 480)), (0, 0))
    pygame.display.flip()

    # Check if the game is over
    done = terminated or truncated

    # Limit the frame rate
    clock.tick(env.metadata['render_fps'])

# Close the environment and Pygame
env.close()
pygame.quit()
