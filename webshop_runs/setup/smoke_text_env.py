"""Non-interactive smoke test for the WebShop gym text environment.

Upstream's run_envs/run_web_agent_text_env.py loops forever with a RandomPolicy
and prints rich markup, which is useless in a batch log. This does the same
three things a real agent does -- reset, search, click -- and asserts on each,
so a broken index or a missing data file fails loudly instead of hanging.
"""
import sys

import gym
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (registers the env)
from web_agent_site.utils import DEBUG_PROD_SIZE


def main():
    env = gym.make('WebAgentTextEnv-v0',
                   observation_mode='text',
                   num_products=DEBUG_PROD_SIZE)
    try:
        env.reset()
        obs = env.observation
        assert obs and 'Instruction' in obs, f'unexpected initial observation: {obs[:200]!r}'
        print('--- goal ---')
        print(obs[:400])

        actions = env.get_available_actions()
        print('--- available actions at root ---')
        print(actions)
        assert actions.get('has_search_bar'), 'root page has no search bar'

        # A query the 1k subset is guaranteed to have *some* lexical overlap with.
        env.step('search[shirt]')
        obs = env.observation
        print('--- after search[shirt] (first 400 chars) ---')
        print(obs[:400])

        actions = env.get_available_actions()
        clickables = [c for c in actions.get('clickables', [])
                      if c not in ('search', 'back to search')]
        print('--- clickables after search ---')
        print(clickables[:12])
        assert clickables, 'search returned no clickable results -- index is empty or mismatched'

        # Click the first product ASIN and confirm the page actually changes.
        asin = clickables[0]
        before = env.observation
        env.step(f'click[{asin}]')
        after = env.observation
        assert after != before, f'click[{asin}] did not change the page'
        print(f'--- after click[{asin}] (first 300 chars) ---')
        print(after[:300])

        print('SMOKE TEST PASSED')
        return 0
    finally:
        env.close()


if __name__ == '__main__':
    sys.exit(main())
