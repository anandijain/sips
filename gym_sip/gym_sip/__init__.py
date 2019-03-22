from gym.envs.registration import register

register(
    id='Sip-v0',
    entry_point='gym_sip.envs.sip_env:SipEnv',
    kwargs={'fn': '../data/nba2.csv'}
)
