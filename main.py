import gymnasium as gym
import sinergym
import numpy as np
from datetime import datetime, timedelta
from numpy.random import default_rng
from typing import Dict, Any, List

from sinergym.utils.rewards import BaseReward

# 🔹 kendi rule-based controller'ını import et
from rule_based import RuleBasedControllerDiscrete




# =======================
#   MAIN SCRIPT
# =======================
if __name__ == "__main__":

    env_name = 'Eplus-A403mediumfanger-hot-discrete'  # use the same environment name

    print(f"--- Creating environment '{env_name}'... ---")


    env = gym.make(env_name)

    print("--- Environment created successfully! ---")

    # İstersen yorum satırlarını açıp gözlem isimlerini, action space'i vs görebilirsin
    # print("--- Environment's Observation Variables ---")
    # print(env.get_wrapper_attr('observation_variables'))
    # print("--- ACTION SPACE ---")
    # print("Action Space Type:", env.action_space)
    # print("Actuator Names (the 'key' for the action array):", list(env.spec.kwargs['actuators'].keys()))

    action_mapping_function = env.get_wrapper_attr('action_mapping')

    # Action mapping'i bir kere görmek istersen:
    for action_number in range(env.action_space.n):
        try:
            real_action_values = action_mapping_function(action_number)
            print(f"Action {action_number}: {real_action_values}")
        except IndexError:
            print(f"Action {action_number}: [ERROR - This action is not defined in the mapping!]")
    print("-----------------------")

    print("--- Resetting environment... ---")
    observation, info = env.reset()

    # 🔹 Artık CRAZY yerine RULE-BASED controller kullanıyoruz
    controller = RuleBasedControllerDiscrete()

    rewards = []

    print(f"--- Starting simulation loop (100 steps) ---")
    for i in range(100):
        terminated = False
        truncated = False

        # Dilersen info'dan ay bilgisi vs kullanabilirsin
        # current_month = info['month']

        # Controller'dan action al
        action = controller.act(observation)

        # Ortamda bir adım at
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        print(f"Step: {i+1}/100, Reward: {reward:.4f}")

        if terminated or truncated:
            print(f"--- Episode finished at step {i+1}, resetting... ---")
            observation, info = env.reset()

    print("\n--- EPISODE FINISHED ---")
    print(f"Episode Mean reward: {np.mean(rewards):.4f}")
    print(f"Episode Cumulative reward: {sum(rewards):.2f}")
    print("--------------------------\n")

    env.close()
    print("--- Controller Script Finished ---")
