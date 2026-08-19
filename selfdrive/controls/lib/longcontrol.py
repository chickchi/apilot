from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.controls.lib.drive_helpers import CONTROL_N, apply_deadzone
from selfdrive.controls.lib.pid import PIDController
from selfdrive.modeld.constants import T_IDXS
from common.params import Params

LongCtrlState = car.CarControl.Actuators.LongControlState


### apilot
# planned_stop조건인데..... accel은 이미 stoppingAccel보다 낮은상태... 이상태로 stopping으로 진입하면.. 너무 많은 -accel로 정지하게 됨.
# accel이 -값에서 0에 가까와질때까지 기다릴 필요가 있음.... 20230911
def long_control_state_trans(CP, active, long_control_state, v_ego, v_target,
                             v_target_1sec, brake_pressed, cruise_standstill, softHold, a_target_now):
  # Ignore cruise standstill if car has a gas interceptor
  cruise_standstill = cruise_standstill and not CP.enableGasInterceptor
  accelerating = v_target_1sec > (v_target + 0.01)
  planned_stop = (v_target < CP.vEgoStopping and ## apilot: 내리막, 신호정지시 질질 가는 현상... v_target으로 보면.. 급정지, v_ego를 보면 질질감..
                  v_target_1sec < CP.vEgoStopping and
                  not accelerating)
  stay_stopped = (v_ego < CP.vEgoStopping and
                  (brake_pressed or cruise_standstill))
  stopping_condition = planned_stop or stay_stopped

  starting_condition = (v_target_1sec > CP.vEgoStarting and
                        accelerating and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state in (LongCtrlState.off, LongCtrlState.pid):
      long_control_state = LongCtrlState.pid
      if stopping_condition and a_target_now > -1.0:  ### pid출력이 급정지(-accel) 상태에서 stopping으로 들어가면... 차량이 너무 급하게 섬.. 기다려보자.... 시험 230911
        long_control_state = LongCtrlState.stopping

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid

    if softHold:
      long_control_state = LongCtrlState.stopping
  return long_control_state, planned_stop


class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.v_pid = 0.0
    self.last_output_accel = 0.0
    self.debugLoCText = ""
    self.readParamCount = 0
    self.longitudinalTuningKpV = 1.0
    self.longitudinalTuningKiV = 0.0
    self.longitudinalTuningKf = 1.0
    self.startAccelApply = 0.0
    self.stopAccelApply = 0.0

    # v1.3 smooth positive acceleration state/debug
    # This does not force any gear. It only smooths the rise of positive
    # acceleration demand after the longitudinal PID calculation.
    self.raw_output_accel = 0.0
    self.pos_accel_jerk_limit = 0.0
    self.pos_accel_headroom = 0.0
    self.pos_accel_comfort_cap = 0.0
    self.pos_accel_cut = 0.0
    self.pos_accel_limited = False
    self.pos_accel_jerk_limited = False
    self.pos_accel_headroom_limited = False

    # v1.5 adaptive upshift assist
    # 0=idle, 1=soft-release/hold, 2=post-shift low-load, 3=cooldown
    self.upshift_state = 0
    self.upshift_timer = 0.0
    self.upshift_cooldown = 0.0
    self.upshift_recent_decel = 0.0
    self.upshift_entry_output = 0.0
    self.upshift_entry_rpm = 0.0
    self.upshift_peak_rpm = 0.0
    self.upshift_entry_speed = 0.0
    self.upshift_cap = 0.0
    self.upshift_rpm_threshold = 0.0
    self.upshift_shift_detected = False
    self.upshift_limit_active = False

    self.longitudinalActuatorDelayLowerBound = float(int(Params().get("LongitudinalActuatorDelayLowerBound", encoding="utf8"))) * 0.01
    self.longitudinalActuatorDelayUpperBound = float(int(Params().get("LongitudinalActuatorDelayUpperBound", encoding="utf8"))) * 0.01

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

    # Cancel any incomplete upshift-assist cycle when longitudinal control
    # itself is reset (OFF/stopping/starting transition).
    self.upshift_state = 0
    self.upshift_timer = 0.0
    self.upshift_cooldown = 0.0
    self.upshift_recent_decel = 0.0
    self.upshift_cap = 0.0
    self.upshift_shift_detected = False
    self.upshift_limit_active = False

  def update(self, active, CS, long_plan, accel_limits, t_since_plan, CC):
    self.readParamCount += 1
    if self.readParamCount >= 100:
      self.readParamCount = 0
    elif self.readParamCount == 10:
      self.longitudinalTuningKpV = float(int(Params().get("LongitudinalTuningKpV", encoding="utf8"))) * 0.01
      self.longitudinalTuningKiV = float(int(Params().get("LongitudinalTuningKiV", encoding="utf8"))) * 0.001
      self.longitudinalTuningKf = float(int(Params().get("LongitudinalTuningKf", encoding="utf8"))) * 0.01

      ## longcontrolTuning이 한개일때만 적용
      if len(self.CP.longitudinalTuning.kpBP) == 1 and len(self.CP.longitudinalTuning.kiBP)==1:
        self.CP.longitudinalTuning.kpV = [self.longitudinalTuningKpV]
        self.CP.longitudinalTuning.kiV = [self.longitudinalTuningKiV]
        self.pid._k_p = (self.CP.longitudinalTuning.kpBP, self.CP.longitudinalTuning.kpV)
        self.pid._k_i = (self.CP.longitudinalTuning.kiBP, self.CP.longitudinalTuning.kiV)
        self.pid.k_f = self.longitudinalTuningKf
        #self.pid._k_i = ([0, 2.0, 200], [self.longitudinalTuningKiV, 0.0, 0.0]) # 정지때만.... i를 적용해보자... 시험..
    elif self.readParamCount == 30:
      self.longitudinalActuatorDelayLowerBound = float(int(Params().get("LongitudinalActuatorDelayLowerBound", encoding="utf8"))) * 0.01
      self.longitudinalActuatorDelayUpperBound = float(int(Params().get("LongitudinalActuatorDelayUpperBound", encoding="utf8"))) * 0.01
    elif self.readParamCount == 40:
      self.startAccelApply = float(int(Params().get("StartAccelApply", encoding="utf8"))) * 0.01
      self.stopAccelApply = float(int(Params().get("StopAccelApply", encoding="utf8"))) * 0.01
      
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    # Interp control trajectory
    speeds = long_plan.speeds
    a_target_now = 0.0
    if len(speeds) == CONTROL_N:
      v_target_now = interp(t_since_plan, T_IDXS[:CONTROL_N], speeds)
      a_target_now = interp(t_since_plan, T_IDXS[:CONTROL_N], long_plan.accels)
      j_target = long_plan.jerks[0]

      #v_target_lower = interp(self.CP.longitudinalActuatorDelayLowerBound + t_since_plan, T_IDXS[:CONTROL_N], speeds)
      #a_target_lower = 2 * (v_target_lower - v_target_now) / self.CP.longitudinalActuatorDelayLowerBound - a_target_now

      #v_target_upper = interp(self.CP.longitudinalActuatorDelayUpperBound + t_since_plan, T_IDXS[:CONTROL_N], speeds)      
      #a_target_upper = 2 * (v_target_upper - v_target_now) / self.CP.longitudinalActuatorDelayUpperBound - a_target_now

      v_target_lower = interp(self.longitudinalActuatorDelayLowerBound + t_since_plan, T_IDXS[:CONTROL_N], speeds)
      a_target_lower = 2 * (v_target_lower - v_target_now) / self.longitudinalActuatorDelayLowerBound - a_target_now

      v_target_upper = interp(self.longitudinalActuatorDelayUpperBound + t_since_plan, T_IDXS[:CONTROL_N], speeds)
      a_target_upper = 2 * (v_target_upper - v_target_now) / self.longitudinalActuatorDelayUpperBound - a_target_now

      v_target = min(v_target_lower, v_target_upper)
      a_target = min(a_target_lower, a_target_upper)


      #v_target_1sec = interp(self.CP.longitudinalActuatorDelayUpperBound + t_since_plan + 1.0, T_IDXS[:CONTROL_N], speeds)
      #v_target_1sec = interp(self.longitudinalActuatorDelayUpperBound + t_since_plan + 1.0, T_IDXS[:CONTROL_N], speeds)
      v_target_1sec = interp(self.longitudinalActuatorDelayLowerBound + t_since_plan + 1.0, T_IDXS[:CONTROL_N], speeds)
    else:
      v_target = 0.0
      v_target_now = 0.0
      v_target_1sec = 0.0
      a_target = 0.0
      j_target = 0.0
      a_target_lower = a_target_upper = 0.0

    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.CP.startingState = True if self.startAccelApply > 0.0 else False
    self.CP.startAccel = 2.0 * self.startAccelApply
    self.CP.stopAccel = -2.0 * self.stopAccelApply

    output_accel = self.last_output_accel

    self.long_control_state, planned_stop = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       v_target, v_target_1sec, CS.brakePressed,
                                                       CS.cruiseState.standstill, CC.hudControl.softHold, a_target_now)

    if self.long_control_state == LongCtrlState.off:
      self.reset(CS.vEgo)
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
        if CC.hudControl.softHold:
          output_accel = self.CP.stopAccel
      self.reset(CS.vEgo)

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset(CS.vEgo)

    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target_now

      # Toyota starts braking more when it thinks you want to stop
      # Freeze the integrator so we don't accelerate to compensate, and don't allow positive acceleration
      # TODO too complex, needs to be simplified and tested on toyotas
      prevent_overshoot = not self.CP.stoppingControl and CS.vEgo < 1.5 and v_target_1sec < 0.7 and v_target_1sec < self.v_pid
      deadzone = interp(CS.vEgo, self.CP.longitudinalTuning.deadzoneBP, self.CP.longitudinalTuning.deadzoneV)
      freeze_integrator = prevent_overshoot

      error = self.v_pid - CS.vEgo
      error_deadzone = apply_deadzone(error, deadzone)
      output_accel = self.pid.update(error_deadzone, speed=CS.vEgo,
                                     feedforward=a_target,
                                     freeze_integrator=freeze_integrator)

    # ----------------------------------------------------------------------
    # v1.4 Smooth Positive Acceleration + PID Comfort Headroom
    #
    # Why shape the PID output instead of retuning the PID itself?
    #   The PID is still useful for speed-error, grade/load and actuator
    #   compensation.  The vehicle-specific problem seen in the road tests is
    #   that a large positive PID correction can sit on top of the planner
    #   feed-forward and create a torque request large enough to trigger a
    #   harsh 6->5 or 5->4 kickdown.
    #
    # v1.4 therefore keeps the PID calculation intact, but constrains only
    # POSITIVE acceleration after the calculation:
    #
    #   final positive output =
    #       min(raw PID output,
    #           previous positive output + J+ * DT_CTRL,
    #           max(a_target, 0) + speed-dependent headroom)
    #
    # The two limits do different jobs:
    #   J+       : how FAST the accelerator request may increase.
    #   headroom : how FAR above the planner feed-forward PID may push.
    #
    # Falling accel demand and all negative accel/braking are passed through
    # immediately; there is no extra brake slew limit here.
    # ----------------------------------------------------------------------
    self.raw_output_accel = float(output_accel)
    self.pos_accel_jerk_limit = 0.0
    self.pos_accel_headroom = 0.0
    self.pos_accel_comfort_cap = 0.0
    self.pos_accel_cut = 0.0
    self.pos_accel_limited = False
    self.pos_accel_jerk_limited = False
    self.pos_accel_headroom_limited = False

    if self.long_control_state == LongCtrlState.pid and output_accel > 0.0:
      v_ego_kph = CS.vEgo * 3.6

      # v1.3 video-derived positive accel rise-rate limit [m/s^3].
      self.pos_accel_jerk_limit = interp(
        v_ego_kph,
        [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0],
        [0.45, 0.38, 0.30, 0.24, 0.18, 0.15, 0.13, 0.12],
      )

      # v1.4 PID positive headroom above the planner feed-forward [m/s^2].
      # Low speed retains more correction authority for launch/load changes.
      # From ~70 km/h upward, only a small positive correction is allowed
      # so a large speed error cannot create a large kickdown torque request.
      self.pos_accel_headroom = interp(
        v_ego_kph,
        [0.0, 20.0, 40.0, 60.0, 70.0, 80.0, 100.0, 120.0, 140.0],
        [0.20, 0.20, 0.15, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04],
      )

      # Planner-based comfort envelope.  max(a_target, 0) intentionally allows
      # a small PID-only correction (the headroom itself) even when a_target
      # is zero or slightly negative.
      self.pos_accel_comfort_cap = max(a_target, 0.0) + self.pos_accel_headroom

      # J+ ceiling: positive torque builds from the previous positive output.
      # If the previous command was braking, positive torque starts from zero.
      positive_base = max(self.last_output_accel, 0.0)
      positive_rise_max = positive_base + self.pos_accel_jerk_limit * DT_CTRL

      raw_positive = output_accel
      output_accel = min(raw_positive, positive_rise_max, self.pos_accel_comfort_cap)

      eps = 1e-5
      self.pos_accel_jerk_limited = positive_rise_max + eps < raw_positive and positive_rise_max <= self.pos_accel_comfort_cap + eps
      self.pos_accel_headroom_limited = self.pos_accel_comfort_cap + eps < raw_positive and self.pos_accel_comfort_cap <= positive_rise_max + eps
      self.pos_accel_limited = output_accel + eps < raw_positive

    # ----------------------------------------------------------------------
    # v1.5 Adaptive Upshift Assist
    #
    # Reproduces a driver's brief accelerator lift to encourage a natural
    # 5->6 (or 4->5->6) upshift during recovery from a slowdown.
    #
    # No gear is commanded and no fixed RPM is targeted.  Hyundai's TCU keeps
    # full authority over gear selection; this code only creates a short,
    # carefully-gated low-load window.
    # ----------------------------------------------------------------------
    self.upshift_limit_active = False
    self.upshift_cap = 0.0

    if self.upshift_cooldown > 0.0:
      self.upshift_cooldown = max(self.upshift_cooldown - DT_CTRL, 0.0)
      if self.upshift_cooldown <= 0.0 and self.upshift_state == 3:
        self.upshift_state = 0

    # Remember a meaningful slowdown for several seconds so the assist is
    # focused on "decelerate -> recover" events rather than normal cruising.
    if self.long_control_state == LongCtrlState.pid and (CS.aEgo < -0.12 or self.raw_output_accel < -0.08):
      self.upshift_recent_decel = 8.0
    elif self.upshift_recent_decel > 0.0:
      self.upshift_recent_decel = max(self.upshift_recent_decel - DT_CTRL, 0.0)

    v_ego_kph = CS.vEgo * 3.6

    # Use the far end of the planner speed trajectory for the recovery gap.
    # self.v_pid is only the *immediate* target and is often very close to
    # vEgo even when the planner is clearly recovering toward a much higher
    # cruise speed.  The future-plan gap is a much better trigger signal.
    v_plan_future = speeds[-1] if len(speeds) == CONTROL_N else self.v_pid
    dv_kph = (v_plan_future - CS.vEgo) * 3.6
    engine_rpm = float(CS.engineRpm)

    # Speed-dependent high-RPM evidence.  This is not a gear detector.
    self.upshift_rpm_threshold = interp(
      v_ego_kph,
      [78.0, 82.0, 85.0, 88.0, 92.0, 96.0],
      [2400.0, 2450.0, 2500.0, 2580.0, 2680.0, 2820.0],
    )

    driver_override = CS.gasPressed or CS.brakePressed
    rpm_valid = engine_rpm > 700.0
    high_rpm = rpm_valid and engine_rpm >= self.upshift_rpm_threshold

    # Simple grade/load guard: do not request an upshift when the car is not
    # actually accelerating under the current positive command.
    accel_response_ok = CS.aEgo > 0.08

    trigger_window = (
      self.long_control_state == LongCtrlState.pid and
      not driver_override and
      self.upshift_state == 0 and
      self.upshift_cooldown <= 0.0 and
      82.0 <= v_ego_kph <= 92.0 and
      4.0 <= dv_kph <= 22.0 and
      output_accel >= 0.28 and
      accel_response_ok and
      high_rpm and
      (self.upshift_recent_decel > 0.0 or engine_rpm >= self.upshift_rpm_threshold + 250.0)
    )

    if trigger_window:
      self.upshift_state = 1
      self.upshift_timer = 0.0
      self.upshift_entry_output = max(output_accel, 0.0)
      self.upshift_entry_rpm = engine_rpm
      self.upshift_peak_rpm = engine_rpm
      self.upshift_entry_speed = v_ego_kph
      self.upshift_shift_detected = False

    # Abort on driver intervention, braking demand, a new slowdown, invalid
    # RPM telemetry, or leaving the useful recovery region.
    abort_assist = (
      self.upshift_state in (1, 2) and (
        driver_override or
        self.long_control_state != LongCtrlState.pid or
        self.raw_output_accel <= 0.0 or
        not rpm_valid or
        v_ego_kph < 78.0 or
        v_ego_kph > 98.0 or
        dv_kph < 1.5 or
        CS.aEgo < -0.18
      )
    )

    if abort_assist:
      self.upshift_state = 3
      self.upshift_timer = 0.0
      self.upshift_cooldown = 2.0
      self.upshift_cap = 0.0

    elif self.upshift_state == 1:
      self.upshift_timer += DT_CTRL
      self.upshift_peak_rpm = max(self.upshift_peak_rpm, engine_rpm)

      # Smoothly lift toward a low positive request rather than dropping the
      # accel command abruptly.
      relief_target = interp(
        v_ego_kph,
        [80.0, 85.0, 90.0, 95.0],
        [0.22, 0.20, 0.18, 0.16],
      )
      release_ramp_time = 0.35
      release_progress = min(self.upshift_timer / release_ramp_time, 1.0)
      self.upshift_cap = self.upshift_entry_output + (relief_target - self.upshift_entry_output) * release_progress
      output_accel = min(output_accel, self.upshift_cap)
      self.upshift_limit_active = True

      # RPM drop while road speed is held is evidence of an upshift.
      rpm_drop = self.upshift_peak_rpm - engine_rpm
      speed_held = v_ego_kph >= self.upshift_entry_speed - 0.8
      if self.upshift_timer >= 0.20 and rpm_drop >= 300.0 and speed_held:
        self.upshift_shift_detected = True
        self.upshift_state = 2
        self.upshift_timer = 0.0
      elif self.upshift_timer >= 1.10:
        self.upshift_state = 2
        self.upshift_timer = 0.0

    elif self.upshift_state == 2:
      self.upshift_timer += DT_CTRL

      # Briefly keep load moderate after the likely upshift so the TCU is less
      # likely to kick back down immediately.
      post_cap = interp(
        v_ego_kph,
        [80.0, 85.0, 90.0, 95.0, 100.0],
        [0.34, 0.32, 0.30, 0.27, 0.24],
      )
      self.upshift_cap = min(post_cap, self.pos_accel_comfort_cap if self.pos_accel_comfort_cap > 0.0 else post_cap)
      output_accel = min(output_accel, self.upshift_cap)
      self.upshift_limit_active = True

      # If reduced torque no longer produces forward acceleration, exit early
      # instead of lugging the engine on an uphill/load condition.
      weak_response = self.upshift_timer > 0.45 and CS.aEgo < 0.02 and dv_kph > 5.0
      post_duration = 2.20 if self.upshift_shift_detected else 1.35

      if weak_response or self.upshift_timer >= post_duration or dv_kph < 3.0 or v_ego_kph >= 97.0:
        self.upshift_state = 3
        self.upshift_timer = 0.0
        self.upshift_cooldown = 4.0
        self.upshift_cap = 0.0

    self.last_output_accel = clip(output_accel, accel_limits[0], accel_limits[1])
    self.pos_accel_cut = max(self.raw_output_accel - self.last_output_accel, 0.0)

    # Compact v1.5 debug:
    # E=speed error, AT=planner accel, R=raw PID, PC=v1.4 comfort cap,
    # J=v1.3 rise limit, O=final accel, C=total cut,
    # U=upshift state 0/1/2/3, UC=upshift cap,
    # RPM=engine rpm, RT=rpm trigger threshold, DV=target gap km/h,
    # SD=RPM-drop shift detected, AE=measured accel.
    self.debugLoCText = (
      f"LC E={self.v_pid - CS.vEgo:+.2f}"
      f" AT={a_target:.2f} R={self.raw_output_accel:.2f}"
      f" PC={self.pos_accel_comfort_cap:.2f} J={self.pos_accel_jerk_limit:.2f}"
      f" O={self.last_output_accel:.2f} C={self.pos_accel_cut:.2f}"
      f" U={self.upshift_state} UC={self.upshift_cap:.2f}"
      f" RPM={int(CS.engineRpm)} RT={int(self.upshift_rpm_threshold)}"
      f" DV={dv_kph:.1f} SD={int(self.upshift_shift_detected)}"
      f" AE={CS.aEgo:.2f}"
    )

    return self.last_output_accel, -0.5 if planned_stop else j_target
