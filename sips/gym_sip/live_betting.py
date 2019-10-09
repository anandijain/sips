import gym
import gym_sip

import h

import lines as ll
import sipqn


def main():
    env = gym.make('Sip-v0').unwrapped
    sip = ll.Sippy(fn=None, header=0, league='mlb', verbosity=0) # start scraper for nba, not writing to file

    N_STATES = env.observation_space.shape[0]
    N_ACTIONS = env.action_space.n
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

    steps_done = 0
    dqn = sipqn.DQN(N_STATES, N_ACTIONS, ENV_A_SHAPE)

    sip.step()
    try:
        while True:
            sip.step()
            steps_done += 1

            for game in sip.games:
                # print(game)
                # print(cur_state)
                cur_state = game.return_row()

                a = dqn.choose_action(cur_state)  # give deep q network state and return action
                next_state, r, d, odds = env.step(a, cur_state)  # next state, reward, done, odds

                dqn.store_transition(cur_state, a, r, next_state)

                print('step: {}'.format(dqn.memory_counter))
                print('game_id: {}'.format(env.game.id))
                print('state: {}'.format(cur_state))
                print('reward: {}'.format(r))
                print('env.money: {}'.format(env.money))
                print(odds)
                print('\n')

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if not d:
                	print('.')
                else:
                    break
    except KeyboardInterrupt:
        print('stopped')


if __name__ == '__main__':
    main()
