"""Observation-only TCU downshift load relief for apilot v1.7.0.

This module never requests a gear.  It only returns a positive-acceleration
ceiling while the stock TCU remains solely responsible for gear selection.
Negative acceleration and driver overrides are deliberately left untouched.
"""

from common.numpy_fast import interp
from selfdrive.swaglog import cloudlog


class TcuDownshiftReliefState:
  NORMAL = 0
  PREVENT_G5 = 1
  PREVENT_G6 = 2
  TARGET_G5 = 3
  TARGET_G6 = 4
  POST_DOWNSHIFT = 5
  DECEL_ESCAPE = 6
  RECOVERY_SETTLE = 7
  RECOVERY_G3 = 8
  RECOVERY_G4 = 9
  RECOVERY_G5 = 10


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
  """Stateful positive-acceleration ceiling and G3->G6 recovery ladder."""

  TARGET_RAMP_TIME = 0.15
  TARGET_HOLD_TIME = 0.60
  DECEL_ESCAPE_CONFIRM_TIME = 0.80
  DECEL_ESCAPE_HOLD_TIME = 1.00

  # v1.7.0 Recovery Ladder
  RECOVERY_DOWNSHIFT_SETTLE = 0.65
  RECOVERY_UPSHIFT_SETTLE = 0.25
  RECOVERY_MAX_AGE = 18.00
  RECOVERY_STATUS_LOG_PERIOD = 0.50
  RECOVERY_DECEL_PAUSE_TIME = 0.30

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

    # v1.7.0 post-downshift recovery ladder.  This state never
    # commands a gear; it only manages a positive SCC accel ceiling.
    self.recovery_active = False
    self.recovery_origin_gear = 0
    self.recovery_stage_gear = 0
    self.recovery_settle_timer = 0.0
    self.recovery_stage_timer = 0.0
    self.recovery_age = 0.0
    self.recovery_status_log_timer = 0.0

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

    # v1.6.3: make the 6th-gear hold strongly RPM-sensitive.  The road
    # video showed G6 at ~1,680 RPM dropping to G5 while accelerating, but
    # another hill case around ~1,880 RPM could usefully remain in G6 with
    # more load.  Keep positive acceleration in both cases, but back off
    # earlier at low RPM so the stock TCU has less reason to request G5.
    rpm_adjust = interp(
      assist_rpm,
      [1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2100.0],
      [-0.06, -0.055, -0.05, -0.01, 0.025, 0.04],
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
    # Once the TCU explicitly asks for a lower gear, keep one continuous
    # positive-torque relief rather than an accelerator off/on pulse.
    return interp(
      dv_kph,
      [0.5, 1.5, 3.0, 15.0, 35.0],
      [0.03, 0.05, 0.08, 0.16, 0.22],
    )

  @staticmethod
  def _log(message):
    try:
      cloudlog.info(message)
    except Exception:
      pass

  @staticmethod
  def _recovery_g3_cap(dv_kph, assist_rpm):
    demand_cap = interp(
      dv_kph,
      [4.0, 10.0, 20.0, 35.0],
      [0.30, 0.34, 0.38, 0.42],
    )
    if assist_rpm <= 700.0:
      return demand_cap

    rpm_cap = interp(
      assist_rpm,
      [1900.0, 2100.0, 2250.0, 2400.0, 2600.0, 2800.0, 3100.0],
      [0.44, 0.42, 0.38, 0.34, 0.30, 0.26, 0.24],
    )
    return min(demand_cap, rpm_cap)

  @staticmethod
  def _recovery_g4_cap(dv_kph, assist_rpm, stage_timer):
    # Keep useful positive acceleration, but progressively unload a high-RPM
    # 4th gear instead of allowing the engine to wind out toward the target.
    cap = interp(
      dv_kph,
      [2.0, 5.0, 10.0, 20.0, 35.0],
      [0.23, 0.26, 0.29, 0.33, 0.36],
    )

    if assist_rpm > 700.0:
      rpm_cap = interp(
        assist_rpm,
        [1900.0, 2100.0, 2250.0, 2400.0, 2550.0, 2700.0, 3000.0],
        [0.36, 0.32, 0.30, 0.27, 0.23, 0.20, 0.18],
      )
      cap = min(cap, rpm_cap)

    if assist_rpm >= 2450.0 and stage_timer > 1.20:
      deep_cap = interp(
        stage_timer,
        [1.20, 2.00, 3.00, 4.00],
        [cap, 0.22, 0.18, 0.16],
      )
      cap = min(cap, deep_cap)

    return max(cap, 0.0)

  @staticmethod
  def _recovery_g5_cap(dv_kph, assist_rpm, stage_timer,
                       target_gear, target_gear_valid):
    # Positive plateau first.  targetGear is observation, not permission:
    # targetGear==5 does not prevent a 5->6 load-relief attempt.
    cap = interp(
      dv_kph,
      [0.5, 5.0, 15.0, 35.0],
      [0.14, 0.16, 0.20, 0.22],
    )

    if assist_rpm > 700.0:
      rpm_cap = interp(
        assist_rpm,
        [1900.0, 2150.0, 2300.0, 2450.0, 2600.0, 2750.0, 3000.0],
        [0.22, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12],
      )
      cap = min(cap, rpm_cap)

    stubborn_g5 = bool(
      target_gear_valid and
      target_gear == 5 and
      assist_rpm >= 2450.0 and
      stage_timer >= 1.50
    )

    if stubborn_g5:
      deep_cap = interp(
        stage_timer,
        [1.50, 2.20, 3.00, 3.80, 4.80],
        [cap, 0.13, 0.10, 0.08, 0.07],
      )
      cap = min(cap, deep_cap)

    return max(cap, 0.0)

  def _start_recovery(self, from_gear, current_gear,
                      cluster_kph, assist_rpm, dv_kph):
    if current_gear not in (3, 4, 5):
      self.recovery_active = False
      return

    self.recovery_active = True
    self.recovery_origin_gear = int(from_gear)
    self.recovery_stage_gear = int(current_gear)
    self.recovery_settle_timer = self.RECOVERY_DOWNSHIFT_SETTLE
    self.recovery_stage_timer = 0.0
    self.recovery_age = 0.0
    self.recovery_status_log_timer = 0.0

    self._log(
      "[GEAR_EVT] RECOVERY_START "
      f"G{int(from_gear)}->{int(current_gear)} "
      f"V={cluster_kph:.1f} RPM={assist_rpm:.0f} DV={dv_kph:.1f}"
    )

  def _finish_recovery(self, reason, current_gear,
                       cluster_kph, assist_rpm, dv_kph):
    if self.recovery_active:
      self._log(
        "[GEAR_EVT] RECOVERY_END "
        f"reason={reason} G={int(current_gear)} "
        f"V={cluster_kph:.1f} RPM={assist_rpm:.0f} DV={dv_kph:.1f} "
        f"AGE={self.recovery_age:.2f}"
      )

    self.recovery_active = False
    self.recovery_origin_gear = 0
    self.recovery_stage_gear = 0
    self.recovery_settle_timer = 0.0
    self.recovery_stage_timer = 0.0
    self.recovery_age = 0.0
    self.recovery_status_log_timer = 0.0

  def _recovery_result(self, dt, positive_control, driver_override,
                       raw_output_accel, output_accel, cluster_kph, dv_kph,
                       current_gear, target_gear, target_gear_valid, a_ego,
                       assist_rpm, actual_downshift):
    if not self.recovery_active:
      return None

    self.recovery_age += dt
    self.recovery_settle_timer = max(
      self.recovery_settle_timer - dt,
      0.0,
    )

    if self.recovery_age >= self.RECOVERY_MAX_AGE:
      self._finish_recovery(
        "TIMEOUT", current_gear, cluster_kph, assist_rpm, dv_kph,
      )
      return None

    if current_gear >= 6:
      self._finish_recovery(
        "G6", current_gear, cluster_kph, assist_rpm, dv_kph,
      )
      return None

    if current_gear < 3:
      self._finish_recovery(
        "LOW_GEAR", current_gear, cluster_kph, assist_rpm, dv_kph,
      )
      return None

    if dv_kph <= 0.5:
      self._finish_recovery(
        "TARGET_REACHED", current_gear, cluster_kph, assist_rpm, dv_kph,
      )
      return None

    # Advance the ladder on an observed real shift.  A G4->G5 event therefore
    # keeps recovery alive; after only 0.25 s it evaluates G5->G6 even when
    # targetGear still reports 5.
    if current_gear != self.recovery_stage_gear:
      old_gear = self.recovery_stage_gear
      self.recovery_stage_gear = int(current_gear)
      self.recovery_stage_timer = 0.0
      self.recovery_settle_timer = self.RECOVERY_UPSHIFT_SETTLE
      self.recovery_status_log_timer = 0.0

      self._log(
        "[GEAR_EVT] RECOVERY_GEAR "
        f"G{int(old_gear)}->{int(current_gear)} "
        f"V={cluster_kph:.1f} RPM={assist_rpm:.0f} DV={dv_kph:.1f}"
      )

    # Never fight braking, a driver accelerator override, or non-positive
    # longitudinal control.  Recovery remains armed and can resume later.
    bypass = bool(
      not positive_control or
      driver_override or
      raw_output_accel <= 0.0 or
      output_accel <= 0.0
    )
    if bypass:
      self.recovery_stage_timer = 0.0
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    # Pause recovery only after sustained real deceleration, not on one noisy
    # aEgo sample.  0.30 s prevents fighting a genuine slowdown while the
    # longer v1.6.3 decel-escape latch still provides hysteresis at 0.80 s.
    if self.decel_timer >= self.RECOVERY_DECEL_PAUSE_TIME:
      self.recovery_stage_timer = 0.0
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    # Use the existing sustained-deceleration escape latch after confirmation.
    if self.decel_escape_timer > 0.0:
      self.recovery_stage_timer = 0.0
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    # An explicit lower-gear target is respected. targetGear==currentGear is
    # not a veto because the load relief itself may be what lets TCU choose up.
    if target_gear_valid and target_gear < current_gear:
      self.recovery_stage_timer = 0.0
      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    if self.recovery_settle_timer > 0.0:
      # During the short mechanical settle window, keep the same conservative
      # positive cap style used by v1.6.3 post-downshift relief.
      if self.recovery_origin_gear == 6:
        cap = interp(
          dv_kph,
          [0.5, 3.0, 15.0, 35.0],
          [0.08, 0.10, 0.18, 0.26],
        )
      else:
        cap = self._g5_target_cap(dv_kph)

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.RECOVERY_SETTLE,
        cap=cap,
        active=output_accel > cap,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.recovery_settle_timer,
      )

    rpm_valid = assist_rpm > 700.0
    state = TcuDownshiftReliefState.NORMAL
    cap = 0.0
    eligible = False

    if current_gear == 3:
      eligible = bool(
        cluster_kph >= 45.0 and
        dv_kph >= 4.0 and
        (not rpm_valid or assist_rpm >= 2100.0)
      )
      if eligible:
        state = TcuDownshiftReliefState.RECOVERY_G3
        cap = self._recovery_g3_cap(dv_kph, assist_rpm)

    elif current_gear == 4:
      eligible = bool(
        cluster_kph >= 54.0 and
        dv_kph >= 2.0 and
        (not rpm_valid or assist_rpm >= 2050.0)
      )
      if eligible:
        state = TcuDownshiftReliefState.RECOVERY_G4
        cap = self._recovery_g4_cap(
          dv_kph,
          assist_rpm,
          self.recovery_stage_timer,
        )

    elif current_gear == 5:
      eligible = bool(
        cluster_kph >= 78.0 and
        dv_kph >= 0.8 and
        (not rpm_valid or assist_rpm >= 1900.0)
      )
      if eligible:
        state = TcuDownshiftReliefState.RECOVERY_G5
        cap = self._recovery_g5_cap(
          dv_kph,
          assist_rpm,
          self.recovery_stage_timer,
          target_gear,
          target_gear_valid,
        )

    if not eligible:
      self.recovery_stage_timer = 0.0
      return None

    self.recovery_stage_timer += dt
    self.recovery_status_log_timer += dt
    active = output_accel > cap

    if self.recovery_status_log_timer >= self.RECOVERY_STATUS_LOG_PERIOD:
      self.recovery_status_log_timer = 0.0
      self._log(
        "[GEAR_CTL] "
        f"S={state} G={current_gear}>"
        f"{target_gear if target_gear_valid else 0} "
        f"V={cluster_kph:.1f} RPM={assist_rpm:.0f} DV={dv_kph:.1f} "
        f"RAW={raw_output_accel:.2f} IN={output_accel:.2f} "
        f"CAP={cap:.2f} AE={a_ego:.2f} "
        f"T={self.recovery_stage_timer:.2f}"
      )

    return TcuDownshiftReliefResult(
      state=state,
      cap=cap,
      active=active,
      suppress_legacy=True,
      actual_downshift=actual_downshift,
      target_down_timer=self.target_down_timer,
      cooldown=0.0,
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
    assist_rpm = max(float(assist_rpm), 0.0)

    current_valid = self._valid_gear(current_gear)
    previous_valid = self._valid_gear(self.previous_gear)
    previous_gear = self.previous_gear

    actual_downshift = bool(
      current_valid and
      previous_valid and
      previous_gear in (4, 5, 6) and
      current_gear < previous_gear
    )

    gear_increased = bool(
      current_valid and
      previous_valid and
      current_gear > previous_gear
    )

    if actual_downshift:
      self.downshift_from_gear = previous_gear
      self.post_relief_timer = self.RECOVERY_DOWNSHIFT_SETTLE
      self.cooldown = self.RECOVERY_DOWNSHIFT_SETTLE
      self.target_hold_timer = 0.0
      self.target_down_timer = 0.0
      self.target_relief_elapsed = 0.0
      self._start_recovery(
        previous_gear,
        current_gear,
        cluster_kph,
        assist_rpm,
        dv_kph,
      )

    elif gear_increased:
      # The Recovery Ladder itself handles G3->G4->G5 progression.  Clear the
      # old one-shot post-downshift latch but do not destroy recovery state.
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

    recovery = self._recovery_result(
      dt,
      positive_control,
      driver_override,
      float(raw_output_accel),
      float(output_accel),
      cluster_kph,
      dv_kph,
      current_gear,
      target_gear,
      target_gear_valid,
      float(a_ego),
      assist_rpm,
      actual_downshift,
    )
    if recovery is not None:
      return recovery

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
        active=float(output_accel) > cap,
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
        active=float(output_accel) > cap,
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
        active=float(output_accel) > cap,
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
