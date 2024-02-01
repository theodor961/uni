
from gym.envs.registration import register

register(
    id='My-v0',
    entry_point="continuous_mountain_car.envs:Continuous_MountainCarEnv",
     max_episode_steps=2000,
)



# from gym.envs.registration import register

# register(
#     id='Pygame-v0',
#     entry_point='gym_game.envs:CustomEnv',
#     max_episode_steps=2000,
# )
