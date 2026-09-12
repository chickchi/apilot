"""Unified observation-only gear assist for APilot v1.8.0.

G1/G2 are telemetry-only.
G3/G4/G5 use per-gear load relief to encourage the stock TCU to upshift.
G6 retains conservative downshift/hold protection.

This module NEVER commands a gear directly.
The stock TCU remains the sole gear selector.

v1.8.0 key changes:
- Remove the shared 18-second recovery timeout.
- Track every gear from G1 through G6.
- Reset stage/eligible/stubborn timers on every real gear transition.
- G1/G2 are observation-only.
- G3/G4/G5 each have independent assist profiles.
- High-RPM "stubborn" timing starts from zero only when the condition starts.
- Positive acceleration is only reduced, never increased.
- Braking and driver override always win.
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

  GEAR_SETTLE = 7
  ASSIST_G3 = 8
  ASSIST_G4 = 9
  ASSIST_G5 = 10

  MONITOR_G1 = 11
  MONITOR_G2 = 12

  # Compatibility with v1.7.0 debug state numbers/names.
  RECOVERY_SETTLE = GEAR_SETTLE
  RECOVERY_G3 = ASSIST_G3
  RECOVERY_G4 = ASSIST_G4
  RECOVERY_G5 = ASSIST_G5


class TcuDownshiftReliefResult:
  def __init__(
    self,
    state=0,
    cap=0.0,
    active=False,
    suppress_legacy=False,
    actual_downshift=False,
    target_down_timer=0.0,
    cooldown=0.0,
  ):
    self.state = int(state)
    self.cap = float(cap)
    self.active = bool(active)
    self.suppress_legacy = bool(suppress_legacy)
    self.actual_downshift = bool(actual_downshift)
    self.target_down_timer = float(target_down_timer)
    self.cooldown = float(cooldown)


class TcuDownshiftRelief:
  """Unified G1-G6 observation and G3/G4/G5 positive-load assist."""

  TARGET_RAMP_TIME = 0.15
  TARGET_HOLD_TIME = 0.60

  DECEL_ESCAPE_CONFIRM_TIME = 0.80
  DECEL_ESCAPE_HOLD_TIME = 1.00

  GEAR_DOWNSHIFT_SETTLE = 0.65
  GEAR_UPSHIFT_SETTLE = 0.25

  GEAR_STATUS_LOG_PERIOD = 0.50
  LOW_GEAR_MONITOR_PERIOD = 1.00

  GEAR_DECEL_PAUSE_TIME = 0.30

  def __init__(self):
    self.reset()

  @staticmethod
  def _valid_gear(gear):
    return 1 <= int(gear) <= 8

  @staticmethod
  def _log(message):
    try:
      cloudlog.info(message)
    except Exception:
      pass

  def reset(self, current_gear=0):
    current_gear = int(current_gear)

    if self._valid_gear(current_gear):
      self.previous_gear = current_gear
      self.gear_stage = current_gear
    else:
      self.previous_gear = 0
      self.gear_stage = 0

    # -------------------------------------------------------------
    # v1.8.0 per-gear timers
    # -------------------------------------------------------------
    self.gear_stage_age = 0.0
    self.gear_eligible_age = 0.0
    self.gear_stubborn_age = 0.0

    self.gear_settle_timer = 0.0
    self.gear_status_log_timer = 0.0
    self.low_gear_monitor_timer = 0.0

    self.gear_transition_down = False

    # Existing lower-gear request protection.
    self.target_down_timer = 0.0
    self.target_relief_elapsed = 0.0
    self.target_hold_timer = 0.0
    self.target_entry_output = 0.0

    # Deceleration escape.
    self.decel_timer = 0.0
    self.decel_escape_timer = 0.0

    # Longcontrol compatibility.
    self.cooldown = 0.0

  # =====================================================================
  # G6 HOLD / DOWNSHIFT PROTECTION
  # =====================================================================

  @staticmethod
  def _g6_prevent_cap(dv_kph, assist_rpm=0.0):
    base_cap = interp(
      dv_kph,
      [
        0.5,
        1.5,
        3.0,
        6.0,
        10.0,
        15.0,
        25.0,
        35.0,
      ],
      [
        0.06,
        0.10,
        0.14,
        0.20,
        0.22,
        0.22,
        0.29,
        0.32,
      ],
    )

    if float(assist_rpm) <= 700.0:
      return base_cap

    rpm_adjust = interp(
      assist_rpm,
      [
        1500.0,
        1600.0,
        1700.0,
        1800.0,
        1900.0,
        2100.0,
      ],
      [
        -0.06,
        -0.055,
        -0.05,
        -0.01,
        0.025,
        0.04,
      ],
    )

    rpm_weight = interp(
      dv_kph,
      [
        0.5,
        3.0,
        6.0,
        10.0,
        15.0,
        20.0,
        25.0,
      ],
      [
        0.0,
        0.0,
        0.8,
        1.0,
        1.0,
        0.5,
        0.0,
      ],
    )

    return max(
      base_cap +
      rpm_adjust *
      rpm_weight,
      0.0,
    )

  @staticmethod
  def _g5_target_cap(dv_kph):
    return interp(
      dv_kph,
      [
        0.5,
        5.0,
        15.0,
        35.0,
      ],
      [
        0.18,
        0.22,
        0.30,
        0.38,
      ],
    )

  @staticmethod
  def _g6_target_cap(dv_kph):
    return interp(
      dv_kph,
      [
        0.5,
        1.5,
        3.0,
        15.0,
        35.0,
      ],
      [
        0.03,
        0.05,
        0.08,
        0.16,
        0.22,
      ],
    )

  # =====================================================================
  # POST-GEAR-CHANGE SETTLE
  # =====================================================================

  @staticmethod
  def _settle_cap(current_gear, dv_kph):
    if current_gear == 3:
      return interp(
        dv_kph,
        [4.0, 15.0, 35.0],
        [0.34, 0.38, 0.42],
      )

    if current_gear == 4:
      return interp(
        dv_kph,
        [2.0, 15.0, 35.0],
        [0.28, 0.32, 0.36],
      )

    if current_gear == 5:
      return interp(
        dv_kph,
        [0.8, 15.0, 35.0],
        [0.18, 0.20, 0.22],
      )

    return 0.0

  # =====================================================================
  # G3 ASSIST
  # =====================================================================

  @staticmethod
  def _gear_g3_cap(
    dv_kph,
    assist_rpm,
    stubborn_age,
  ):
    cap = interp(
      dv_kph,
      [
        4.0,
        10.0,
        20.0,
        35.0,
      ],
      [
        0.30,
        0.34,
        0.38,
        0.42,
      ],
    )

    if assist_rpm > 700.0:
      rpm_cap = interp(
        assist_rpm,
        [
          1900.0,
          2100.0,
          2250.0,
          2400.0,
          2600.0,
          2800.0,
          3100.0,
        ],
        [
          0.44,
          0.42,
          0.38,
          0.34,
          0.30,
          0.26,
          0.24,
        ],
      )

      cap = min(
        cap,
        rpm_cap,
      )

    if stubborn_age > 0.0:
      deep_cap = interp(
        stubborn_age,
        [
          0.0,
          0.8,
          1.6,
          2.6,
        ],
        [
          cap,
          0.31,
          0.27,
          0.24,
        ],
      )

      cap = min(
        cap,
        deep_cap,
      )

    return max(
      cap,
      0.0,
    )

  # =====================================================================
  # G4 ASSIST
  # =====================================================================

  @staticmethod
  def _gear_g4_cap(
    dv_kph,
    assist_rpm,
    stubborn_age,
  ):
    cap = interp(
      dv_kph,
      [
        2.0,
        5.0,
        10.0,
        20.0,
        35.0,
      ],
      [
        0.23,
        0.26,
        0.29,
        0.33,
        0.36,
      ],
    )

    if assist_rpm > 700.0:
      rpm_cap = interp(
        assist_rpm,
        [
          1900.0,
          2100.0,
          2250.0,
          2400.0,
          2550.0,
          2700.0,
          3000.0,
        ],
        [
          0.36,
          0.32,
          0.30,
          0.27,
          0.23,
          0.20,
          0.18,
        ],
      )

      cap = min(
        cap,
        rpm_cap,
      )

    if stubborn_age > 0.0:
      deep_cap = interp(
        stubborn_age,
        [
          0.0,
          0.7,
          1.5,
          2.5,
          3.5,
        ],
        [
          cap,
          0.24,
          0.21,
          0.18,
          0.17,
        ],
      )

      cap = min(
        cap,
        deep_cap,
      )

    return max(
      cap,
      0.0,
    )

  # =====================================================================
  # G5 ASSIST
  # =====================================================================

  @staticmethod
  def _gear_g5_cap(
    dv_kph,
    assist_rpm,
    stubborn_age,
  ):
    # Normal positive plateau.
    cap = interp(
      dv_kph,
      [
        0.5,
        5.0,
        15.0,
        35.0,
      ],
      [
        0.14,
        0.16,
        0.20,
        0.22,
      ],
    )

    if assist_rpm > 700.0:
      rpm_cap = interp(
        assist_rpm,
        [
          1900.0,
          2150.0,
          2300.0,
          2400.0,
          2500.0,
          2600.0,
          2750.0,
          3000.0,
        ],
        [
          0.22,
          0.22,
          0.20,
          0.19,
          0.17,
          0.16,
          0.14,
          0.12,
        ],
      )

      cap = min(
        cap,
        rpm_cap,
      )

    # IMPORTANT:
    # stubborn_age begins at ZERO only when the actual high-RPM G5>5
    # condition becomes true. It never inherits time spent earlier in G5.
    if stubborn_age > 0.0:
      deep_cap = interp(
        stubborn_age,
        [
          0.0,
          0.7,
          1.4,
          2.2,
          3.2,
        ],
        [
          cap,
          0.15,
          0.12,
          0.10,
          0.08,
        ],
      )

      cap = min(
        cap,
        deep_cap,
      )

    return max(
      cap,
      0.0,
    )

  # =====================================================================
  # PER-GEAR STAGE MANAGEMENT
  # =====================================================================

  def _reset_stage(
    self,
    current_gear,
    old_gear,
    cluster_kph,
    assist_rpm,
    dv_kph,
  ):
    self.gear_stage = int(
      current_gear
    )

    self.gear_stage_age = 0.0
    self.gear_eligible_age = 0.0
    self.gear_stubborn_age = 0.0

    self.gear_status_log_timer = 0.0

    self.gear_transition_down = bool(
      old_gear > 0 and
      current_gear <
      old_gear
    )

    if (
      old_gear > 0 and
      current_gear <
      old_gear
    ):
      self.gear_settle_timer = (
        self.GEAR_DOWNSHIFT_SETTLE
      )

    elif (
      old_gear > 0 and
      current_gear >
      old_gear
    ):
      self.gear_settle_timer = (
        self.GEAR_UPSHIFT_SETTLE
      )

    else:
      self.gear_settle_timer = 0.0

    if old_gear > 0:
      self._log(
        "[GEAR_EVT] SHIFT "
        f"G{int(old_gear)}->"
        f"{int(current_gear)} "
        f"V={cluster_kph:.1f} "
        f"RPM={assist_rpm:.0f} "
        f"DV={dv_kph:.1f}"
      )

  # =====================================================================
  # G1-G5 UNIFIED MANAGER
  # =====================================================================

  def _managed_gear_result(
    self,
    dt,
    positive_control,
    driver_override,
    raw_output_accel,
    output_accel,
    cluster_kph,
    dv_kph,
    current_gear,
    target_gear,
    target_gear_valid,
    a_ego,
    assist_rpm,
    actual_downshift,
  ):
    # -------------------------------------------------------------------
    # G1 / G2: OBSERVATION ONLY
    # -------------------------------------------------------------------

    if current_gear in (
      1,
      2,
    ):
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      self.low_gear_monitor_timer += dt

      if (
        self.low_gear_monitor_timer >=
        self.LOW_GEAR_MONITOR_PERIOD and
        (
          positive_control or
          (
            target_gear_valid and
            target_gear !=
            current_gear
          )
        )
      ):
        self.low_gear_monitor_timer = 0.0

        self._log(
          "[GEAR_MON] "
          f"G={current_gear}>"
          f"{target_gear if target_gear_valid else 0} "
          f"V={cluster_kph:.1f} "
          f"RPM={assist_rpm:.0f} "
          f"DV={dv_kph:.1f} "
          f"RAW={raw_output_accel:.2f} "
          f"IN={output_accel:.2f} "
          f"AE={a_ego:.2f} "
          f"SA={self.gear_stage_age:.2f}"
        )

      return TcuDownshiftReliefResult(
        state=(
          TcuDownshiftReliefState.MONITOR_G1
          if current_gear == 1
          else TcuDownshiftReliefState.MONITOR_G2
        ),
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.gear_settle_timer,
      )

    self.low_gear_monitor_timer = 0.0

    # G6 is handled separately below.
    if current_gear >= 6:
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0
      return None

    # Only G3 / G4 / G5 are active assist gears.
    if current_gear not in (
      3,
      4,
      5,
    ):
      return None

    # Driver / braking / non-positive control always wins.
    bypass = bool(
      not positive_control or
      driver_override or
      raw_output_accel <= 0.0 or
      output_accel <= 0.0
    )

    if bypass:
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.gear_settle_timer,
      )

    # Never oppose an explicit LOWER-gear request.
    if (
      target_gear_valid and
      target_gear <
      current_gear
    ):
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0
      return None

    if (
      self.decel_timer >=
      self.GEAR_DECEL_PAUSE_TIME or
      self.decel_escape_timer > 0.0
    ):
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.DECEL_ESCAPE,
        suppress_legacy=False,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.gear_settle_timer,
      )

    rpm_valid = (
      assist_rpm >
      700.0
    )

    # -------------------------------------------------------------------
    # G3
    # -------------------------------------------------------------------

    if current_gear == 3:
      eligible = bool(
        cluster_kph >=
        45.0 and
        dv_kph >=
        4.0 and
        (
          not rpm_valid or
          assist_rpm >=
          2100.0
        )
      )

      stubborn = bool(
        eligible and
        target_gear_valid and
        target_gear ==
        3 and
        rpm_valid and
        assist_rpm >=
        2400.0 and
        cluster_kph >=
        48.0
      )

      state = (
        TcuDownshiftReliefState.ASSIST_G3
      )

    # -------------------------------------------------------------------
    # G4
    # -------------------------------------------------------------------

    elif current_gear == 4:
      eligible = bool(
        cluster_kph >=
        54.0 and
        dv_kph >=
        2.0 and
        (
          not rpm_valid or
          assist_rpm >=
          2100.0
        )
      )

      stubborn = bool(
        eligible and
        target_gear_valid and
        target_gear ==
        4 and
        rpm_valid and
        assist_rpm >=
        2400.0 and
        cluster_kph >=
        58.0
      )

      state = (
        TcuDownshiftReliefState.ASSIST_G4
      )

    # -------------------------------------------------------------------
    # G5
    # -------------------------------------------------------------------

    else:
      eligible = bool(
        cluster_kph >=
        80.0 and
        dv_kph >=
        0.8 and
        (
          not rpm_valid or
          assist_rpm >=
          2050.0
        )
      )

      # Begin BEFORE the old 2450-RPM threshold.
      #
      # The 2026-09-11 run showed that waiting until ~2450 RPM was too late.
      # Start the independent stubborn timer at ~2350 RPM instead.
      stubborn = bool(
        eligible and
        target_gear_valid and
        target_gear ==
        5 and
        rpm_valid and
        assist_rpm >=
        2350.0 and
        cluster_kph >=
        86.0
      )

      state = (
        TcuDownshiftReliefState.ASSIST_G5
      )

    # -------------------------------------------------------------------
    # Mechanical settle
    # -------------------------------------------------------------------

    if self.gear_settle_timer > 0.0:
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      # Only a real downshift gets a settle cap.
      # After a successful upshift we do NOT create an artificial torque hole.
      if self.gear_transition_down:
        cap = self._settle_cap(
          current_gear,
          dv_kph,
        )

        active = bool(
          cap > 0.0 and
          output_accel >
          cap
        )

      else:
        cap = 0.0
        active = False

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.GEAR_SETTLE,
        cap=cap,
        active=active,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.gear_settle_timer,
      )

    # -------------------------------------------------------------------
    # Not yet eligible
    #
    # The new manager still OWNS G3/G4/G5 here.
    # This prevents the old legacy manager racing the v1.8 manager.
    # -------------------------------------------------------------------

    if not eligible:
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.NORMAL,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=0.0,
      )

    # -------------------------------------------------------------------
    # Stage-local timing
    # -------------------------------------------------------------------

    self.gear_eligible_age += dt

    if stubborn:
      self.gear_stubborn_age += dt
    else:
      self.gear_stubborn_age = 0.0

    # -------------------------------------------------------------------
    # Per-gear cap
    # -------------------------------------------------------------------

    if current_gear == 3:
      cap = self._gear_g3_cap(
        dv_kph,
        assist_rpm,
        self.gear_stubborn_age,
      )

    elif current_gear == 4:
      cap = self._gear_g4_cap(
        dv_kph,
        assist_rpm,
        self.gear_stubborn_age,
      )

    else:
      cap = self._gear_g5_cap(
        dv_kph,
        assist_rpm,
        self.gear_stubborn_age,
      )

    active = bool(
      output_accel >
      cap
    )

    # -------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------

    self.gear_status_log_timer += dt

    if (
      self.gear_status_log_timer >=
      self.GEAR_STATUS_LOG_PERIOD
    ):
      self.gear_status_log_timer = 0.0

      self._log(
        "[GEAR_CTL] "
        f"S={state} "
        f"G={current_gear}>"
        f"{target_gear if target_gear_valid else 0} "
        f"V={cluster_kph:.1f} "
        f"RPM={assist_rpm:.0f} "
        f"DV={dv_kph:.1f} "
        f"RAW={raw_output_accel:.2f} "
        f"IN={output_accel:.2f} "
        f"CAP={cap:.2f} "
        f"AE={a_ego:.2f} "
        f"SA={self.gear_stage_age:.2f} "
        f"EA={self.gear_eligible_age:.2f} "
        f"ST={self.gear_stubborn_age:.2f}"
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

  # =====================================================================
  # MAIN UPDATE
  # =====================================================================

  def update(
    self,
    dt,
    positive_control,
    driver_override,
    raw_output_accel,
    output_accel,
    cluster_kph,
    dv_kph,
    current_gear,
    target_gear,
    target_gear_valid,
    a_ego,
    assist_rpm=0.0,
  ):
    dt = max(
      float(dt),
      0.0,
    )

    current_gear = int(
      current_gear
    )

    target_gear = int(
      target_gear
    )

    cluster_kph = max(
      float(cluster_kph),
      0.0,
    )

    dv_kph = max(
      float(dv_kph),
      0.0,
    )

    assist_rpm = max(
      float(assist_rpm),
      0.0,
    )

    raw_output_accel = float(
      raw_output_accel
    )

    output_accel = float(
      output_accel
    )

    a_ego = float(
      a_ego
    )

    current_valid = (
      self._valid_gear(
        current_gear
      )
    )

    previous_valid = (
      self._valid_gear(
        self.previous_gear
      )
    )

    previous_gear = (
      self.previous_gear
    )

    target_gear_valid = bool(
      target_gear_valid and
      self._valid_gear(
        target_gear
      )
    )

    # Any real lower shift from G2-G6 is observable.
    actual_downshift = bool(
      current_valid and
      previous_valid and
      2 <=
      previous_gear <=
      6 and
      current_gear <
      previous_gear
    )

    # -------------------------------------------------------------------
    # Every actual gear transition resets ALL stage-local timers.
    # -------------------------------------------------------------------

    if (
      current_valid and
      current_gear !=
      self.gear_stage
    ):
      self._reset_stage(
        current_gear,
        self.gear_stage,
        cluster_kph,
        assist_rpm,
        dv_kph,
      )

    elif not current_valid:
      self.gear_stage = 0

      self.gear_stage_age = 0.0
      self.gear_eligible_age = 0.0
      self.gear_stubborn_age = 0.0

      self.gear_settle_timer = 0.0

      self.gear_transition_down = False

    if current_valid:
      self.gear_stage_age += dt

    self.gear_settle_timer = max(
      self.gear_settle_timer -
      dt,
      0.0,
    )

    self.cooldown = (
      self.gear_settle_timer
    )

    # -------------------------------------------------------------------
    # Lower-target observation
    # -------------------------------------------------------------------

    target_down_now = bool(
      current_valid and
      target_gear_valid and
      current_gear in (
        5,
        6,
      ) and
      target_gear <
      current_gear
    )

    if target_down_now:
      if self.target_down_timer <= 0.0:
        self.target_entry_output = max(
          output_accel,
          0.0,
        )

        self.target_relief_elapsed = 0.0

      self.target_down_timer += dt
      self.target_relief_elapsed += dt

      self.target_hold_timer = (
        self.TARGET_HOLD_TIME
      )

    else:
      self.target_down_timer = 0.0

      self.target_hold_timer = max(
        self.target_hold_timer -
        dt,
        0.0,
      )

      if self.target_hold_timer > 0.0:
        self.target_relief_elapsed += dt
      else:
        self.target_relief_elapsed = 0.0

    # -------------------------------------------------------------------
    # Decel escape
    # -------------------------------------------------------------------

    meaningful_decel = bool(
      a_ego <
      -0.15 and
      dv_kph >
      4.0
    )

    if meaningful_decel:
      self.decel_timer = min(
        self.decel_timer +
        dt,
        self.DECEL_ESCAPE_CONFIRM_TIME,
      )

    else:
      self.decel_timer = max(
        self.decel_timer -
        2.0 *
        dt,
        0.0,
      )

    if (
      self.decel_timer >=
      self.DECEL_ESCAPE_CONFIRM_TIME
    ):
      self.decel_escape_timer = (
        self.DECEL_ESCAPE_HOLD_TIME
      )

      self.target_hold_timer = 0.0

    else:
      self.decel_escape_timer = max(
        self.decel_escape_timer -
        dt,
        0.0,
      )

    # Update observation history only after comparing old/current.
    self.previous_gear = (
      current_gear
      if current_valid
      else 0
    )

    if not current_valid:
      return TcuDownshiftReliefResult()

    # ===================================================================
    # G1-G5 unified manager
    # ===================================================================

    managed = self._managed_gear_result(
      dt,
      positive_control,
      driver_override,
      raw_output_accel,
      output_accel,
      cluster_kph,
      dv_kph,
      current_gear,
      target_gear,
      target_gear_valid,
      a_ego,
      assist_rpm,
      actual_downshift,
    )

    if managed is not None:
      return managed

    # ===================================================================
    # Remaining G6 / lower-target protection
    # ===================================================================

    bypass = bool(
      not positive_control or
      driver_override or
      raw_output_accel <=
      0.0 or
      output_accel <=
      0.0
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
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    target_latched = bool(
      target_down_now or
      self.target_hold_timer >
      0.0
    )

    # -------------------------------------------------------------------
    # G6 target lower
    # -------------------------------------------------------------------

    if (
      target_latched and
      current_gear ==
      6 and
      70.0 <=
      cluster_kph <=
      115.0 and
      0.5 <=
      dv_kph <=
      35.0
    ):
      target_cap = (
        self._g6_target_cap(
          dv_kph
        )
      )

      progress = min(
        max(
          self.target_relief_elapsed,
          dt,
        ) /
        self.TARGET_RAMP_TIME,
        1.0,
      )

      cap = (
        self.target_entry_output +
        (
          target_cap -
          self.target_entry_output
        ) *
        progress
      )

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.TARGET_G6,
        cap=cap,
        active=output_accel > cap,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    # -------------------------------------------------------------------
    # G5 target lower
    # -------------------------------------------------------------------

    if (
      target_latched and
      current_gear ==
      5 and
      62.0 <=
      cluster_kph <=
      110.0 and
      0.5 <=
      dv_kph <=
      35.0
    ):
      target_cap = (
        self._g5_target_cap(
          dv_kph
        )
      )

      progress = min(
        max(
          self.target_relief_elapsed,
          dt,
        ) /
        self.TARGET_RAMP_TIME,
        1.0,
      )

      cap = (
        self.target_entry_output +
        (
          target_cap -
          self.target_entry_output
        ) *
        progress
      )

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.TARGET_G5,
        cap=cap,
        active=output_accel > cap,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    # -------------------------------------------------------------------
    # G6 hold / downshift prevention
    # -------------------------------------------------------------------

    if (
      current_gear ==
      6 and
      70.0 <=
      cluster_kph <=
      115.0 and
      0.5 <=
      dv_kph <=
      35.0
    ):
      cap = (
        self._g6_prevent_cap(
          dv_kph,
          assist_rpm,
        )
      )

      return TcuDownshiftReliefResult(
        state=TcuDownshiftReliefState.PREVENT_G6,
        cap=cap,
        active=output_accel > cap,
        suppress_legacy=True,
        actual_downshift=actual_downshift,
        target_down_timer=self.target_down_timer,
        cooldown=self.cooldown,
      )

    return TcuDownshiftReliefResult(
      actual_downshift=actual_downshift,
      target_down_timer=self.target_down_timer,
      cooldown=self.cooldown,
    )
