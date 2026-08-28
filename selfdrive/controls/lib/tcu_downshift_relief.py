"""Observation-only TCU downshift load relief for apilot v1.6.1.

This module never requests a gear.  It only returns a positive-acceleration
ceiling while the stock TCU remains solely responsible for gear selection.
Negative acceleration and driver overrides are deliberately left untouched.
"""

from common.numpy_fast import interp


class TcuDownshiftReliefState:
  NORMAL = 0
  PREVENT_G5 = 1
  PREVENT_G6 = 2
  TARGET_G5 = 3
  TARGET_G6 = 4
  POST_DOWNSHIFT = 5
  DECEL_ESCAPE = 6


class TcuDownshiftReliefResult:
  def __init__(self, state=0, cap=0.0, active=False,
               suppress_legacy=False, actual_downshift=False,
               target_down_timer=0.0, cooldown=0.0):
    self.state = int(state)
    self.cap = float(cap)
    self.active = bool(active)
    self.suppress_legacy = bool(suppress_legacy)
    self.actual_downshift = bool(actual_downshift)
    self.target_down_timer = float(target_down_timer)
    self.cooldown = float(cooldown)


class TcuDownshiftRelief:
  """Stateful positive-acceleration ceiling around 5th/6th downshifts."""

  TARGET_RAMP_TIME = 0.15
  TARGET_HOLD_TIME = 0.60
  POST_RELIEF_TIME = 1.50
  LEGACY_COOLDOWN_TIME = 5.00
  DECEL_ESCAPE_CONFIRM_TIME = 0.80
  DECEL_ESCAPE_HOLD_TIME = 1.00

  def __init__(self):
    self.reset()

  def reset(self, current_gear=0):
    self.previous_gear = int(current_gear)
    self.downshift_from_gear = 0
    self.target_down_timer = 0.0
    self.target_relief_elapsed = 0.0
    self.target_hold_timer = 0.0
    self.target_entry_output = 0.0
    self.post_relief_timer = 0.0
    self.cooldown = 0.0
    self.decel_timer = 0.0
    self.decel_escape_timer = 0.0

  @staticmethod
  def _valid_gear(gear):
    return 1 <= int(gear) <= 8

  @staticmethod
  def _g5_prevent_cap(cluster_kph):
    return interp(
      cluster_kph,
      [68.0, 70.0, 80.0, 100.0],
      [0.46, 0.42, 0.38, 0.34],
    )

  @staticmethod
  def _g6_prevent_cap(dv_kph, assist_rpm=0.0):
    # v1.6.1: permit a little more 6th-gear load on the observed
    # 1,850-RPM gentle hill, while backing off earlier in the observed
    # 1,680-RPM 6->5 event.  The RPM correction fades out for large speed
    # deficits where the stock TCU should remain free to choose a lower gear.
    base_cap = interp(
      dv_kph,
      [0.5, 1.5, 3.0, 6.0, 10.0, 15.0, 25.0, 35.0],
      [0.06, 0.10, 0.14, 0.20, 0.22, 0.22, 0.29, 0.32],
    )

    if float(assist_rpm) <= 700.0:
      return base_cap

    rpm_adjust = interp(
      assist_rpm,
      [1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2100.0],
      [-0.03, -0.03, -0.02, 0.00, 0.02, 0.03],
    )
    rpm_weight = interp(
      dv_kph,
      [0.5, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0],
      [0.0, 0.0, 0.8, 1.0, 1.0, 0.5, 0.0],
    )
    return max(base_cap + rpm_adjust * rpm_weight, 0.0)

  @staticmethod
  def _g5_target_cap(dv_kph):
    return interp(
      dv_kph,
      [0.5, 5.0, 15.0, 35.0],
      [0.18, 0.22, 0.30, 0.38],
    )

  @staticmethod
  def _g6_target_cap(dv_kph):
    return interp(
      dv_kph,
      [0.5, 1.5, 3.0, 15.0, 35.0],
      [0.03, 0.05, 0.08, 0.18, 0.24],
    )

  def update(self, dt, positive_control, driver_override,
             raw_output_accel, output_accel, cluster_kph, dv_kph,
             current_gear, target_gear, target_gear_valid, a_ego,
             assist_rpm=0.0):
    dt = max(float(dt), 0.0)
    current_gear = int(current_gear)
    target_gear = int(target_gear)
    cluster_kph = max(float(cluster_kph), 0.0)
    dv_kph = max(float(dv_kph), 0.0)

    current_valid = self._valid_gear(current_gear)
    previous_valid = self._valid_gear(self.previous_gear)

    actual_downshift = bool(
      current_valid and
      previous_valid and
      self.previous_gear in (4, 5, 6) and
      current_gear < self.previous_gear
    )

    gear_increased = bool(
      current_valid and
      previous_valid and
      current_gear > self.previous_gear
    )

    if actual_downshift:
      self.downshift_from_gear = self.previous_gear
      self.post_relief_timer = self.POST_RELIEF_TIME
      self.cooldown = self.LEGACY_COOLDOWN_TIME
      self.target_hold_timer = 0.0
      self.target_down_timer = 0.0
      self.target_relief_elapsed = 0.0

    elif gear_increased:
      self.post_relief_timer = 0.0
      self.target_hold_timer = 0.0
      self.target_down_timer = 0.0
      self.target_relief_elapsed = 0.0

    else:
      self.post_relief_timer = max(
        self.post_relief_timer - dt,
        0.0,
      )

    self.cooldown = max(
      self.cooldown - dt,
      0.0,
    )

    target_down_now = bool(
      current_valid and
      target_gear_valid and
      current_gear in (5, 6) and
      target_gear < current_gear
    )

    if target_down_now:
      if self.target_down_timer <= 0.0:
        self.target_entry_output = max(
          float(output_accel),
          0.0,
        )
        self.target_relief_elapsed = 0.0

      self.target_down_timer += dt
      self.target_relief_elapsed += dt
      self.target_hold_timer = self.TARGET_HOLD_TIME

    else:
      self.target_down_timer = 0.0
      self.target_hold_timer = max(
        self.target_hold_timer - dt,
        0.0,
      )

      if self.target_hold_timer > 0.0:
        self.target_relief_elapsed += dt
      else:
        self.target_relief_elapsed = 0.0

    meaningful_decel = bool(
      float(a_ego) < -0.15 and
      dv_kph > 4.0
    )

    if meaningful_decel:
      self.decel_timer = min(
        self.decel_timer + dt,
        self.DECEL_ESCAPE_CONFIRM_TIME,
      )

    else:
      self.decel_timer = max(
        self.decel_timer - 2.0 * dt,
        0.0,
      )

    if self.decel_timer >= self.DECEL_ESCAPE_CONFIRM_TIME:
      self.decel_escape_timer = self.DECEL_ESCAPE_HOLD_TIME
      self.target_hold_timer = 0.0

    else:
      self.decel_escape_timer = max(
        self.decel_escape_timer - dt,
        0.0,
      )

    self.previous_gear = current_gear if current_valid else 0

    # Safety invariants: this manager never alters braking or driver input.
    bypass = bool(
      not positive_control or
      driver_override or
      float(raw_output_accel) <= 0.0 or
      float(output_accel) <= 0.0
    )

    if bypass:
      return TcuDownshiftReliefResult(
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if self.decel_escape_timer > 0.0:
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if self.post_relief_timer > 0.0:
      if self.downshift_from_gear == 6:
        cap = interp(
          dv_kph,
          [0.5, 3.0, 15.0, 35.0],
          [0.08, 0.10, 0.18, 0.26],
        )
      else:
        cap = self._g5_target_cap(dv_kph)

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.POST_DOWNSHIFT,
        cap=cap,
        active=True,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    target_latched = bool(
      target_down_now or
      self.target_hold_timer > 0.0
    )

    if (
      target_latched and
      current_gear == 6 and
      70.0 <= cluster_kph <= 115.0 and
      0.5 <= dv_kph <= 35.0
    ):
      target_cap = self._g6_target_cap(dv_kph)
      progress = min(
        max(self.target_relief_elapsed, dt) /
        self.TARGET_RAMP_TIME,
        1.0,
      )
      cap = self.target_entry_output + (
        target_cap - self.target_entry_output
      ) * progress

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.TARGET_G6,
        cap=cap,
        active=True,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if (
      target_latched and
      current_gear == 5 and
      62.0 <= cluster_kph <= 110.0 and
      0.5 <= dv_kph <= 35.0
    ):
      target_cap = self._g5_target_cap(dv_kph)
      progress = min(
        max(self.target_relief_elapsed, dt) /
        self.TARGET_RAMP_TIME,
        1.0,
      )
      cap = self.target_entry_output + (
        target_cap - self.target_entry_output
      ) * progress

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.TARGET_G5,
        cap=cap,
        active=True,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if (
      current_gear == 6 and
      70.0 <= cluster_kph <= 115.0 and
      0.5 <= dv_kph <= 35.0
    ):
      cap = self._g6_prevent_cap(dv_kph, assist_rpm)
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.PREVENT_G6,
        cap=cap,
        active=float(output_accel) > cap,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if (
      current_gear == 5 and
      68.0 <= cluster_kph <= 110.0 and
      0.5 <= dv_kph <= 35.0
    ):
      cap = self._g5_prevent_cap(cluster_kph)
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.PREVENT_G5,
        cap=cap,
        active=float(output_accel) > cap,
        suppress_legacy=self.cooldown > 0.0,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    return TcuDownshiftReliefResult(
      suppress_legacy=self.cooldown > 0.0,
      actual_downshift=actual_downshift,
      target_down_timer=self.target_down_timer,
      cooldown=self.cooldown,
    )
