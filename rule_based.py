# rule_based.py
import numpy as np

# ----- ACTION MAPPING (0–39) -----
DISCRETE_ACTIONS = [
    [21, 22, 1.0, 0.0],  # 0
    [21, 22, 1.0, 0.5],  # 1
    [21, 22, 1.0, 0.75], # 2
    [21, 22, 1.0, 1.0],  # 3
    [22, 23, 1.0, 0.0],  # 4
    [22, 23, 1.0, 0.5],  # 5
    [22, 23, 1.0, 0.75], # 6
    [22, 23, 1.0, 1.0],  # 7
    [23, 24, 1.0, 0.0],  # 8
    [23, 24, 1.0, 0.5],  # 9
    [23, 24, 1.0, 0.75], # 10
    [23, 24, 1.0, 1.0],  # 11
    [24, 25, 1.0, 0.0],  # 12
    [24, 25, 1.0, 0.5],  # 13
    [24, 25, 1.0, 0.75], # 14
    [24, 25, 1.0, 1.0],  # 15
    [25, 26, 1.0, 0.0],  # 16
    [25, 26, 1.0, 0.5],  # 17
    [25, 26, 1.0, 0.75], # 18
    [25, 26, 1.0, 1.0],  # 19
    [26, 27, 1.0, 0.0],  # 20
    [26, 27, 1.0, 0.5],  # 21
    [26, 27, 1.0, 0.75], # 22
    [26, 27, 1.0, 1.0],  # 23
    [27, 28, 1.0, 0.0],  # 24
    [27, 28, 1.0, 0.5],  # 25
    [27, 28, 1.0, 0.75], # 26
    [27, 28, 1.0, 1.0],  # 27
    [28, 29, 1.0, 0.0],  # 28
    [28, 29, 1.0, 0.5],  # 29
    [28, 29, 1.0, 0.75], # 30
    [28, 29, 1.0, 1.0],  # 31
    [29, 30, 1.0, 0.0],  # 32
    [29, 30, 1.0, 0.5],  # 33
    [29, 30, 1.0, 0.75], # 34
    [29, 30, 1.0, 1.0],  # 35
    [5, 50, 0.0, 0.0],   # 36 (AC "off-like", WF off)
    [5, 50, 0.0, 0.5],   # 37
    [5, 50, 0.0, 0.75],  # 38
    [5, 50, 0.0, 1.0],   # 39
]

# Separate grids for temp setpoints and CO2 / WF
DISCRETE_ACTIONS_TEMP = [
    [5, 50],   # "off-like"
    [21, 22],
    [22, 23],
    [23, 24],
    [24, 25],
    [25, 26],
    [26, 27],
    [27, 28],
    [28, 29],
    [29, 30],
]

DISCRETE_ACTIONS_CO2 = [
    0.0,
    0.5,
    0.75,
    1.0,
]

TEMP_THRESHOLD = 1  # steps before patience bumps the discrete index

DISCRETE_ACTIONS = np.array(DISCRETE_ACTIONS, dtype=float)
DISCRETE_ACTIONS_TEMP = np.array(DISCRETE_ACTIONS_TEMP, dtype=float)
DISCRETE_ACTIONS_CO2 = np.array(DISCRETE_ACTIONS_CO2, dtype=float)


class RuleBasedControllerDiscrete:
    """
    Simple rule-based controller for the Sinergym smart room environment.
    
    Observation vector:
      [month, day_of_month, hour, outdoor_temperature, outdoor_humidity,
       htg_setpoint, clg_setpoint, air_temperature, air_humidity,
       people_occupant, air_co2, window_fan_energy, pmv, ppd,
       total_electricity_HVAC]

    Returns:
      action_idx (int in [0, 39]),
      temp_patience (int),
      co2_patience (int)
    """

    def __init__(self,
                 winter_months=(11, 12, 1, 2, 3),
                 temp_margin=0.5):
        self.winter_months = set(winter_months)
        self.temp_margin = temp_margin

    def _get_comfort_range(self, month: int):
        """Return (low, high) comfort temperature range depending on month."""
        if month in self.winter_months:
            # Winter comfort
            return 20.0, 23.0
        else:
            # Summer comfort
            return 23.0, 26.0

    def _decide_window_fan_speed(self, co2: float) -> float:
        """Piecewise rule to decide WF speed based on CO₂ level."""
        if co2 < 800.0:
            return 0.0
        elif co2 < 1000.0:
            return 0.5
        elif co2 < 1200.0:
            return 0.75
        else:
            return 1.0

    def _find_best_action_index(self, desired_action):
        """
        Given desired [htg_sp, clg_sp, fan, wf] (floats),
        choose the closest index from DISCRETE_ACTIONS.
        """
        diffs = DISCRETE_ACTIONS - np.array(desired_action, dtype=float)
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))
    
    def _find_best_action_index_temp(self, desired_temp_pair):
        """
        Given desired [htg_sp, clg_sp],
        choose the closest index from DISCRETE_ACTIONS_TEMP.
        """
        diffs = DISCRETE_ACTIONS_TEMP - np.array(desired_temp_pair, dtype=float)
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def _find_best_action_index_co2(self, desired_wf_speed):
        """
        Given desired WF speed (float),
        choose the closest index from DISCRETE_ACTIONS_CO2.
        """
        diffs = np.abs(DISCRETE_ACTIONS_CO2 - float(desired_wf_speed))
        return int(np.argmin(diffs))

    def act(self, obs, temp_patience, co2_patience):
        """
        Main policy function.
        :param obs: numpy array or list of shape (15,) (observation vector)
        :param temp_patience: int, previous temp patience counter
        :param co2_patience: int, previous CO2 patience counter
        :return: (action_idx, temp_patience, co2_patience)
        """
        obs = np.array(obs, dtype=float)

        # Unpack observation
        month    = int(obs[0])
        air_temp = float(obs[7])
        air_co2  = float(obs[10])

        # 1) Determine comfort range
        t_low, t_high = self._get_comfort_range(month)
        target_temp = 0.5 * (t_low + t_high)

        # --- Update patience counters properly ---
        if air_temp > t_high or air_temp < t_low:
            temp_patience += 1
        else:
            temp_patience = 0

        if air_co2 > 800:
            co2_patience += 1
        else:
            co2_patience = 0

        # 2) Decide if AC should be on or off-like
        if air_temp > t_high:
            ac_on = True   # Too hot -> cooling
        elif air_temp < t_low:
            ac_on = True   # Too cold -> heating
        else:
            ac_on = False  # Inside comfort band

        # 3) Base WF speed from CO2
        wf_speed = self._decide_window_fan_speed(air_co2)

        # 4) Base heating/cooling setpoints + fan
        if not ac_on:
            # Use AC off-like behaviour: [5, 50, 0, wf_speed]
            heating_sp = 5.0
            cooling_sp = 50.0
            fan_speed = 0.0
        else:
            # AC on: choose heating/cooling setpoints around target_temp
            heating_sp = float(np.clip(np.floor(target_temp), 21, 29))
            cooling_sp = float(np.clip(heating_sp + 1.0, 22, 30))
            fan_speed  = 1.0  # ON

        # 5) Map to discrete temp / CO2 indices
        base_temp_idx = self._find_best_action_index_temp([heating_sp, cooling_sp])
        base_co2_idx = self._find_best_action_index_co2(wf_speed)

        temp_patience_residual = temp_patience // TEMP_THRESHOLD
        co2_patience_residual = co2_patience // TEMP_THRESHOLD

        # Adjust temp index based on sign of error + patience (your original logic)
        if not ac_on:
            temp_index = 0
        else:

            if air_temp > t_high:
                # "Too hot" branch in your version
                temp_index = max(base_temp_idx - temp_patience_residual,
                                1)
            elif air_temp < t_low:
                # "Too cold" branch in your version
                temp_index = min(base_temp_idx + temp_patience_residual, len(DISCRETE_ACTIONS_TEMP) - 1)
            else:
                temp_index = base_temp_idx

        #  Clamp temp_index to valid range [0, len(DISCRETE_ACTIONS_TEMP)-1]
        temp_index = max(0, min(temp_index, len(DISCRETE_ACTIONS_TEMP) - 1))

        # Adjust CO2 index monotonically upwards with patience
        co2_index = min(base_co2_idx + co2_patience_residual,
                        len(DISCRETE_ACTIONS_CO2) - 1)

        #  Clamp co2_index to valid range [0, len(DISCRETE_ACTIONS_CO2)-1]
        co2_index = max(0, min(co2_index, len(DISCRETE_ACTIONS_CO2) - 1))

        # 6) Read back discrete values
        htg_sp, clg_sp = DISCRETE_ACTIONS_TEMP[temp_index]
        wf_sp = DISCRETE_ACTIONS_CO2[co2_index]

        # Final desired 4D continuous action
        desired_action = [float(htg_sp),
                          float(clg_sp),
                          float(fan_speed),
                          float(wf_sp)]

        # 7) Map to closest discrete action index [0, 39]
        action_idx = self._find_best_action_index(desired_action)

        # Safety: never allow undefined actions
        if action_idx >= len(DISCRETE_ACTIONS):
            action_idx = len(DISCRETE_ACTIONS) - 1

        return action_idx, temp_patience, co2_patience
