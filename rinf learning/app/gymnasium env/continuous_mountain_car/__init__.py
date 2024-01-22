
from gym.envs.registration import register

register(
    id = "env/continuous_mountain_car",
    entry_point= "my_cmc.env:continuous_mountain_car"
)