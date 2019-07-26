from gym.envs.registration import register

register(
    id='Sip-v0',
    entry_point='gym_sip.envs:SipEnv',
    kwargs={'fn': './data/static/nba2.csv'}
)

register(
    id='Sip-v1',
    entry_point='gym_sip.envs:SipEnv2',
    kwargs={'fn': './data/static/bangout3.csv'}
)

# register(
#     id='Sip-v2',
#     entry_point='gym_sip.envs:SipEnv3',
#     kwargs={}
# )

# register(
#     id='Sip-v3',
#     entry_point='gym_sip.envs:SipEnv4',
#     kwargs={''}
# )
