from gym.envs.registration import register

register(
    id='PomdpHallway-v0',
    entry_point='rl.domains.pomdp_hallway_v0:PomdpHallwayV0',
    max_episode_steps=100,    
)

register(
    id='PomdpHallway2-v0',
    entry_point='rl.domains.pomdp_hallway2_v0:PomdpHallway2V0',
    max_episode_steps=100,    
)

register(
    id='PomdpHeavenHell-v0',
    entry_point='rl.domains.pomdp_heavenhell_v0:PomdpHeavenHellV0',
    max_episode_steps=100,    
)
