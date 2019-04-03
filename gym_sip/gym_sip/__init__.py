from gym.envs.registration import register

register(
    id='Sip-v0',
    entry_point='gym_sip.envs:SipEnv',
    kwargs={'fn': None}
)

register(
    id='Sip-v1',
    entry_point='gym_sip.envs:SipEnv2',
    kwargs={'fn': './data/bangout.csv'}
)
