from gym.envs.registration import register

register(
    id='Sip-v0',
    entry_point='gym_sip.envs:SipEnv',
    kwargs={'fn': './data/nba2.csv'}
)

register(
    id='Sip-v1',
    entry_point='gym_sip.envs:SipEnv2',
    kwargs={'fn': './data/bangout.csv'}
)

# register(
#     id='Sip-v2',
#     entry_point='gym_sip.envs:SipEnv3',
#     kwargs={}
# )
