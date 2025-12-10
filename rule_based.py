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

DISCRETE_ACTIONS = np.array(DISCRETE_ACTIONS, dtype=float)


class RuleBasedControllerDiscrete:
    """
    Simple rule-based controller for the Sinergym smart room environment.
    
    Observation vector:
      [month, day_of_month, hour, outdoor_temperature, outdoor_humidity,
       htg_setpoint, clg_setpoint, air_temperature, air_humidity,
       people_occupant, air_co2, window_fan_energy, pmv, ppd,
       total_electricity_HVAC]
    Returns a discrete action index in [0, 39].
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
        Given desired [htg_sp, clg_sp, fan, wf] (float),
        choose the closest index from DISCRETE_ACTIONS.
        """
        diffs = DISCRETE_ACTIONS - np.array(desired_action, dtype=float)
        dists = np.linalg.norm(diffs, axis=1)
        return int(np.argmin(dists))

    def act(self, obs):
        """
        Main policy function.
        :param obs: numpy array or list of shape (15,) (observation vector)
        :return: int, discrete action index in [0, 39]
        """
        obs = np.array(obs, dtype=float)

        # Unpack observation
        month    = int(obs[0])
        air_temp = float(obs[7])
        air_co2  = float(obs[10])

        # 1) Determine comfort range
        t_low, t_high = self._get_comfort_range(month)
        target_temp = 0.5 * (t_low + t_high)

        # 2) Decide if AC should be on or off-like
        if air_temp > t_high + self.temp_margin:
            ac_on = True   # Too hot -> cooling
        elif air_temp < t_low - self.temp_margin:
            ac_on = True   # Too cold -> heating
        else:
            ac_on = False  # Inside comfort band

        # 3) Decide WF speed based on CO2
        wf_speed = self._decide_window_fan_speed(air_co2)

        # 4) Construct desired action then map to discrete
        if not ac_on:
            # Use AC off-like actions 36–39: [5, 50, 0.0, wf_speed]
            desired_action = [5.0, 50.0, 0.0, wf_speed]
        else:
            # AC on: choose heating/cooling setpoints around target_temp
            heating_sp = float(np.clip(np.floor(target_temp), 21, 29))
            cooling_sp = float(np.clip(heating_sp + 1.0, 22, 30))
            fan_speed  = 1.0  # ON
            desired_action = [heating_sp, cooling_sp, fan_speed, wf_speed]

        action_idx = self._find_best_action_index(desired_action)

        # Safety: never allow undefined actions (>= 40)
        if action_idx >= 40:
            action_idx = 39

        return action_idx
