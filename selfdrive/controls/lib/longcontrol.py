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
                             v_target_1sec, v_target_1p8sec,
                             brake_pressed, cruise_standstill, softHold, a_target_now,
                             lead_status=False, lead_d=0.0, lead_v=0.0, lead_vrel=0.0):
  # Ignore cruise standstill if car has a gas interceptor
  cruise_standstill = cruise_standstill and not CP.enableGasInterceptor

  accelerating_1sec = v_target_1sec > (v_target + 0.01)
  accelerating_1p8sec = v_target_1p8sec > (v_target + 0.01)

  planned_stop = (v_target < CP.vEgoStopping and
                  v_target_1sec < CP.vEgoStopping and
                  not accelerating_1sec)
  stay_stopped = (v_ego < CP.vEgoStopping and
                  (brake_pressed or cruise_standstill))
  stopping_condition = planned_stop or stay_stopped

  normal_start = (v_target_1sec > CP.vEgoStarting and
                  accelerating_1sec and
                  not cruise_standstill and
                  not brake_pressed)

  # v1.5.7 HKG-inspired lead-aware STOPPING release.
  nearby_lead = lead_status and 0.0 < lead_d < 35.0
  stationary_lead_gate = (
    nearby_lead and
    lead_d < 20.0 and
    lead_v < 0.35 and
    abs(lead_vrel) < 0.60
  )
  moving_lead_start = (
    nearby_lead and
    lead_v > max(float(CP.vEgoStarting), 0.50) and
    lead_vrel > 0.05 and
    v_target_1p8sec > CP.vEgoStarting and
    accelerating_1p8sec and
    not cruise_standstill and
    not brake_pressed
  )

  if stationary_lead_gate:
    starting_condition = False
  elif moving_lead_start:
    starting_condition = True
  else:
    starting_condition = normal_start

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

    # v1.5.6 final cruise-speed fail-safe
    self.cruise_guard_cap = 0.0
    self.cruise_overspeed_kph = 0.0
    self.cruise_guard_active = False

    # v1.5.7 lead-aware stop/restart diagnostics
    self.lead_start_status = False
    self.lead_start_moving = False
    self.lead_start_stationary = False
    self.lead_start_v = 0.0
    self.lead_start_d = 0.0
    self.v_target_start_lookahead = 0.0

    # v1.5.3 Adaptive TCU Load Manager
    # 0=NORMAL, 1=PRE_RELIEF, 2=SHIFT, 3=POST_SHIFT, 4=HOLD6, 5=RPM_PROTECT
    self.upshift_state = 0
    self.upshift_timer = 0.0
    self.upshift_cooldown = 0.0
    self.upshift_candidate_timer = 0.0
    self.upshift_entry_output = 0.0
    self.upshift_entry_speed = 0.0
    self.upshift_entry_gear = 0
    self.upshift_post_gear = 0
    self.upshift_cap = 0.0
    self.upshift_soft_rpm = 0.0
    self.upshift_hard_rpm = 0.0
    self.upshift_soft_cap = 0.0
    self.upshift_shift_cap = 0.0
    self.upshift_protect_cap = 0.0
    self.upshift_shift_detected = False
    self.upshift_limit_active = False

    self.longitudinalActuatorDelayLowerBound = float(int(Params().get("LongitudinalActuatorDelayLowerBound", encoding="utf8"))) * 0.01
    self.longitudinalActuatorDelayUpperBound = float(int(Params().get("LongitudinalActuatorDelayUpperBound", encoding="utf8"))) * 0.01

  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

    # Cancel any incomplete v1.5.3 transmission-load cycle.
    self.upshift_state = 0
    self.upshift_timer = 0.0
    self.upshift_cooldown = 0.0
    self.upshift_candidate_timer = 0.0
    self.upshift_entry_output = 0.0
    self.upshift_entry_speed = 0.0
    self.upshift_entry_gear = 0
    self.upshift_post_gear = 0
    self.upshift_cap = 0.0
    self.upshift_soft_rpm = 0.0
    self.upshift_hard_rpm = 0.0
    self.upshift_soft_cap = 0.0
    self.upshift_shift_cap = 0.0
    self.upshift_protect_cap = 0.0
    self.upshift_shift_detected = False
    self.upshift_limit_active = False

  def update(self, active, CS, long_plan, accel_limits, t_since_plan, CC, v_cruise_kph_apply, radar_state=None):
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
      # Only STOPPING release uses this longer forecast.
      v_target_1p8sec = interp(self.longitudinalActuatorDelayLowerBound + t_since_plan + 1.8, T_IDXS[:CONTROL_N], speeds)
    else:
      v_target = 0.0
      v_target_now = 0.0
      v_target_1sec = 0.0
      v_target_1p8sec = 0.0
      a_target = 0.0
      j_target = 0.0
      a_target_lower = a_target_upper = 0.0

    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.CP.startingState = True if self.startAccelApply > 0.0 else False
    self.CP.startAccel = 2.0 * self.startAccelApply
    self.CP.stopAccel = -2.0 * self.stopAccelApply

    output_accel = self.last_output_accel

    # v1.5.7 actual radar lead is used only by the STOPPING release gate.
    self.lead_start_status = False
    self.lead_start_moving = False
    self.lead_start_stationary = False
    self.lead_start_v = 0.0
    self.lead_start_d = 0.0
    self.v_target_start_lookahead = float(v_target_1p8sec)

    lead_vrel = 0.0
    if radar_state is not None:
      lead_one = radar_state.leadOne
      if lead_one.status:
        self.lead_start_status = True
        self.lead_start_v = max(float(lead_one.vLead), 0.0)
        self.lead_start_d = max(float(lead_one.dRel), 0.0)
        lead_vrel = float(lead_one.vRel)
        self.lead_start_stationary = (
          self.lead_start_d < 20.0 and
          self.lead_start_v < 0.35 and
          abs(lead_vrel) < 0.60
        )
        self.lead_start_moving = (
          self.lead_start_d < 35.0 and
          self.lead_start_v > max(float(self.CP.vEgoStarting), 0.50) and
          lead_vrel > 0.05
        )

    self.long_control_state, planned_stop = long_control_state_trans(
      self.CP, active, self.long_control_state, CS.vEgo,
      v_target, v_target_1sec, v_target_1p8sec, CS.brakePressed,
      CS.cruiseState.standstill, CC.hudControl.softHold, a_target_now,
      self.lead_start_status, self.lead_start_d, self.lead_start_v, lead_vrel,
    )

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
    # v1.5.3 Adaptive TCU Load Manager
    #
    # Road-test driven redesign:
    # - No recent_decel hard gate.
    # - Uses G(current gear) + TG(TCU target gear) + TR(TCU RPM) together.
    # - Treats 3->4->5->6 as sequential human-like accelerator eases.
    # - Can preserve 6th if it is already accelerating the car adequately.
    # - Never commands a gear; it only reduces positive acceleration load.
    # - Negative accel/braking is never limited here.
    # ----------------------------------------------------------------------
    self.upshift_limit_active = False
    self.upshift_cap = 0.0

    v_ego_kph = CS.vEgo * 3.6
    v_ego_cluster_kph = float(CS.vEgoCluster * 3.6)
    if v_ego_cluster_kph <= 0.5:
      v_ego_cluster_kph = v_ego_kph

    cruise_target_kph = float(v_cruise_kph_apply)
    if not (1.0 <= cruise_target_kph <= 200.0):
      cruise_target_kph = v_ego_cluster_kph
    # v_cruise_kph_apply is cluster/display-speed based, so DV must use
    # vEgoCluster as well.  The old mixed-domain subtraction made DV stay
    # positive even around the displayed set speed on this vehicle.
    dv_kph = max(cruise_target_kph - v_ego_cluster_kph, 0.0)

    engine_rpm = float(CS.engineRpm)
    tcu_rpm = float(CS.tcuRpm)
    current_gear = int(CS.currentGear)
    target_gear = int(CS.targetGear)
    gear_valid = 1 <= current_gear <= 8
    target_gear_valid = 1 <= target_gear <= 8

    # User's road data: engineRpm is 0, while TCU12.N_TC_RAW follows shift RPM.
    assist_rpm = engine_rpm if engine_rpm > 700.0 else tcu_rpm
    rpm_valid = assist_rpm > 700.0

    driver_override = CS.gasPressed or CS.brakePressed
    positive_control = self.long_control_state == LongCtrlState.pid and output_accel > 0.0
    base_context = positive_control and not driver_override and dv_kph > 2.5

    if self.upshift_cooldown > 0.0:
      self.upshift_cooldown = max(self.upshift_cooldown - DT_CTRL, 0.0)

    # Per-gear comfortable RPM/load envelope derived from the uploaded videos.
    soft_min_v = 0.0
    hard_min_v = 0.0
    soft_rpm_base = 0.0
    hard_rpm_base = 0.0
    soft_cap_base = 0.0
    shift_cap_base = 0.0
    protect_cap_base = 0.0

    if current_gear == 3:
      soft_min_v, hard_min_v = 45.0, 42.0
      soft_rpm_base, hard_rpm_base = 2100.0, 2350.0
      soft_cap_base, shift_cap_base, protect_cap_base = 0.42, 0.34, 0.46
    elif current_gear == 4:
      # v1.5.4: earlier but shallower 4->5 preparation.
      soft_min_v, hard_min_v = 56.0, 50.0
      soft_rpm_base, hard_rpm_base = 2200.0, 2425.0
      soft_cap_base, shift_cap_base, protect_cap_base = 0.40, 0.31, 0.42
    elif current_gear == 5:
      # v1.5.4: earlier 5->6 preparation; M=2 can add a second micro-lift.
      soft_min_v, hard_min_v = 79.0, 65.0
      soft_rpm_base, hard_rpm_base = 2100.0, 2300.0
      soft_cap_base, shift_cap_base, protect_cap_base = 0.30, 0.20, 0.34
    elif not gear_valid:
      # Conservative fallback if numeric gear is unavailable.
      soft_min_v, hard_min_v = 82.0, 78.0
      soft_rpm_base, hard_rpm_base = 2500.0, 2750.0
      soft_cap_base, shift_cap_base, protect_cap_base = 0.30, 0.22, 0.34

    # Large target gaps allow a modestly stronger, but still smooth, recovery.
    demand_rpm_boost_raw = interp(
      dv_kph,
      [0.0, 20.0, 40.0, 60.0],
      [0.0, 0.0, 120.0, 220.0],
    )
    # v1.5.4: large DV must not make upper gears wind out.
    if current_gear == 3:
      demand_rpm_boost = demand_rpm_boost_raw * 0.80
    elif current_gear == 4:
      demand_rpm_boost = demand_rpm_boost_raw * 0.35
    elif current_gear == 5:
      demand_rpm_boost = 0.0
    else:
      demand_rpm_boost = demand_rpm_boost_raw

    demand_cap_boost = interp(
      dv_kph,
      [0.0, 20.0, 40.0, 60.0],
      [0.0, 0.0, 0.03, 0.05],
    )

    self.upshift_soft_rpm = soft_rpm_base + demand_rpm_boost if soft_rpm_base > 0.0 else 0.0
    self.upshift_hard_rpm = hard_rpm_base + demand_rpm_boost if hard_rpm_base > 0.0 else 0.0
    self.upshift_soft_cap = soft_cap_base + demand_cap_boost if soft_cap_base > 0.0 else 0.0
    self.upshift_shift_cap = shift_cap_base + demand_cap_boost if shift_cap_base > 0.0 else 0.0
    self.upshift_protect_cap = protect_cap_base + demand_cap_boost if protect_cap_base > 0.0 else 0.0

    tg_up = target_gear_valid and gear_valid and target_gear > current_gear
    tg_down = target_gear_valid and gear_valid and target_gear < current_gear

    # Do not encourage an upshift on a hill/load condition where the car is
    # barely responding.  Strong response is required for RPM_PROTECT.
    accel_response_ok = CS.aEgo > 0.08
    strong_response = CS.aEgo > 0.15

    gear_managed = (current_gear in (3, 4, 5)) if gear_valid else rpm_valid

    soft_rpm_candidate = (
      base_context and gear_managed and rpm_valid and accel_response_ok and
      v_ego_kph >= soft_min_v and assist_rpm >= self.upshift_soft_rpm
    )
    hard_rpm_candidate = (
      base_context and gear_managed and rpm_valid and accel_response_ok and
      v_ego_kph >= hard_min_v and assist_rpm >= self.upshift_hard_rpm
    )

    # TG>G means the TCU itself wants the higher gear.  A small load reduction
    # is then more appropriate than waiting to wind the current gear out.
    tcu_up_candidate = (
      base_context and gear_valid and current_gear in (3, 4, 5) and tg_up and
      accel_response_ok and v_ego_kph >= max(hard_min_v, soft_min_v - 4.0) and
      (not rpm_valid or assist_rpm >= max(self.upshift_soft_rpm - 150.0, 1200.0))
    )

    # TG<G + already high RPM + strong acceleration: reduce load enough to
    # avoid one more unnecessary kickdown, but do not use the deep SHIFT cap.
    rpm_protect_candidate = (
      base_context and gear_valid and current_gear in (3, 4, 5) and tg_down and
      rpm_valid and strong_response and v_ego_kph >= hard_min_v and
      assist_rpm >= self.upshift_hard_rpm
    )

    # Preserve 6th only while it is clearly doing the job and the TCU has not
    # yet requested a lower gear.  This is deliberately not a gear lock.
    hold6_candidate = (
      base_context and gear_valid and current_gear == 6 and
      78.0 <= v_ego_kph <= 94.0 and 3.0 <= dv_kph <= 25.0 and
      output_accel > 0.38 and CS.aEgo > 0.08 and
      (not target_gear_valid or target_gear >= 6)
    )

    candidate_state = 0
    if hold6_candidate:
      candidate_state = 4
    elif rpm_protect_candidate:
      candidate_state = 5
    elif tcu_up_candidate or hard_rpm_candidate:
      candidate_state = 2
    elif soft_rpm_candidate:
      candidate_state = 1

    # Short debounce: enough to reject a one-frame TG/RPM spike, not enough to
    # reproduce v1.5.2's HD=0.50 but U=0 missed events.
    if self.upshift_state == 0 and self.upshift_cooldown <= 0.0 and candidate_state != 0:
      self.upshift_candidate_timer = min(self.upshift_candidate_timer + DT_CTRL, 0.30)
      debounce_required = 0.10 if candidate_state in (2, 5) else 0.12
      if self.upshift_candidate_timer >= debounce_required:
        self.upshift_state = candidate_state
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(output_accel, 0.0)
        self.upshift_entry_speed = v_ego_kph
        self.upshift_entry_gear = current_gear if gear_valid else 0
        self.upshift_post_gear = 0
        self.upshift_shift_detected = False
        self.upshift_candidate_timer = 0.0
    elif self.upshift_state == 0:
      self.upshift_candidate_timer = 0.0

    # Braking/driver intervention always wins.
    abort_manager = (
      self.upshift_state != 0 and (
        driver_override or
        self.long_control_state != LongCtrlState.pid or
        self.raw_output_accel <= 0.0 or
        dv_kph <= 1.5
      )
    )
    if abort_manager:
      self.upshift_state = 0
      self.upshift_timer = 0.0
      self.upshift_candidate_timer = 0.0
      self.upshift_cooldown = 0.20
      self.upshift_cap = 0.0

    gear_increased = (
      gear_valid and self.upshift_entry_gear > 0 and
      current_gear > self.upshift_entry_gear
    )
    gear_decreased = (
      gear_valid and self.upshift_entry_gear > 0 and
      current_gear < self.upshift_entry_gear
    )

    # M=1 PRE_RELIEF: first, gentle accelerator ease.
    if self.upshift_state == 1:
      self.upshift_timer += DT_CTRL

      if gear_increased:
        self.upshift_state = 3
        self.upshift_timer = 0.0
        self.upshift_post_gear = current_gear
        self.upshift_shift_detected = True
        self.upshift_entry_output = max(output_accel, 0.0)
      elif rpm_protect_candidate:
        self.upshift_state = 5
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(output_accel, 0.0)
      elif tcu_up_candidate or hard_rpm_candidate:
        self.upshift_state = 2
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(output_accel, 0.0)
      else:
        target_cap = self.upshift_soft_cap
        release_progress = min(self.upshift_timer / 0.40, 1.0)
        self.upshift_cap = self.upshift_entry_output + (target_cap - self.upshift_entry_output) * release_progress
        output_accel = min(output_accel, self.upshift_cap)
        self.upshift_limit_active = True

        weak_response = self.upshift_timer > 0.50 and CS.aEgo < 0.02 and dv_kph > 5.0
        rpm_recovered = rpm_valid and self.upshift_timer > 0.30 and assist_rpm < self.upshift_soft_rpm - 140.0
        if weak_response or rpm_recovered or self.upshift_timer >= 2.0:
          self.upshift_state = 0
          self.upshift_timer = 0.0
          self.upshift_cooldown = 0.50

    # M=2 SHIFT: smooth shift relief.  G5 gets a second human-like micro-lift
    # if the first ~.20 request does not complete 5->6.
    elif self.upshift_state == 2:
      self.upshift_timer += DT_CTRL

      if gear_increased:
        self.upshift_state = 3
        self.upshift_timer = 0.0
        self.upshift_post_gear = current_gear
        self.upshift_shift_detected = True
        self.upshift_entry_output = max(output_accel, 0.0)
      else:
        if self.upshift_entry_gear == 5:
          stage_a_cap = self.upshift_shift_cap
          if self.upshift_timer <= 0.65:
            release_progress = min(self.upshift_timer / 0.30, 1.0)
            target_cap = self.upshift_entry_output + (stage_a_cap - self.upshift_entry_output) * release_progress
          else:
            micro_cap = min(0.10 + demand_cap_boost * 0.35, stage_a_cap)
            micro_progress = min((self.upshift_timer - 0.65) / 0.35, 1.0)
            target_cap = stage_a_cap + (micro_cap - stage_a_cap) * micro_progress
          shift_timeout = 2.20
          weak_check_time = 0.95
        elif self.upshift_entry_gear == 4:
          release_progress = min(self.upshift_timer / 0.32, 1.0)
          target_cap = self.upshift_entry_output + (self.upshift_shift_cap - self.upshift_entry_output) * release_progress
          shift_timeout = 1.35
          weak_check_time = 0.50
        else:
          release_progress = min(self.upshift_timer / 0.30, 1.0)
          target_cap = self.upshift_entry_output + (self.upshift_shift_cap - self.upshift_entry_output) * release_progress
          shift_timeout = 1.40
          weak_check_time = 0.50

        self.upshift_cap = target_cap
        output_accel = min(output_accel, self.upshift_cap)
        self.upshift_limit_active = True

        weak_response = self.upshift_timer > weak_check_time and CS.aEgo < 0.01 and dv_kph > 5.0
        if weak_response or self.upshift_timer >= shift_timeout:
          self.upshift_state = 0
          self.upshift_timer = 0.0
          self.upshift_cooldown = 0.45 if self.upshift_entry_gear == 5 else 0.50

    # M=5 RPM_PROTECT: high RPM while TCU asks for another downshift.
    elif self.upshift_state == 5:
      self.upshift_timer += DT_CTRL

      if gear_decreased:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.35
      elif tcu_up_candidate:
        self.upshift_state = 2
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(output_accel, 0.0)
      else:
        target_cap = self.upshift_protect_cap
        release_progress = min(self.upshift_timer / 0.30, 1.0)
        self.upshift_cap = self.upshift_entry_output + (target_cap - self.upshift_entry_output) * release_progress
        output_accel = min(output_accel, self.upshift_cap)
        self.upshift_limit_active = True

        protect_resolved = (
          (not tg_down) or
          (rpm_valid and assist_rpm < self.upshift_hard_rpm - 150.0) or
          CS.aEgo < 0.05
        )
        if (self.upshift_timer > 0.30 and protect_resolved) or self.upshift_timer >= 1.20:
          self.upshift_state = 0
          self.upshift_timer = 0.0
          self.upshift_cooldown = 0.45

    # M=3 POST_SHIFT: brief mechanical settling, no long cooldown.
    elif self.upshift_state == 3:
      self.upshift_timer += DT_CTRL

      if gear_valid and current_gear > self.upshift_post_gear > 0:
        self.upshift_post_gear = current_gear
        self.upshift_timer = 0.0
        self.upshift_shift_detected = True

      if current_gear <= 4:
        post_cap = 0.44 + demand_cap_boost
        post_duration = 0.70
      elif current_gear == 5:
        post_cap = 0.38 + demand_cap_boost
        post_duration = 0.75
      else:
        post_cap = 0.34 + demand_cap_boost
        post_duration = 1.00

      self.upshift_cap = post_cap
      output_accel = min(output_accel, self.upshift_cap)
      self.upshift_limit_active = True

      next_up_ready = (
        self.upshift_timer >= 0.35 and
        gear_valid and current_gear in (3, 4, 5) and
        target_gear_valid and target_gear > current_gear and
        (not rpm_valid or assist_rpm >= max(self.upshift_soft_rpm - 100.0, 1200.0))
      )
      if next_up_ready:
        self.upshift_state = 2
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(output_accel, 0.0)
        self.upshift_entry_gear = current_gear
      elif self.upshift_timer >= post_duration:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.20
        self.upshift_entry_gear = current_gear if gear_valid else 0

    # M=4 HOLD6: human-like "don't press deeper if 6th is already pulling".
    elif self.upshift_state == 4:
      self.upshift_timer += DT_CTRL

      hold6_cap = interp(
        dv_kph,
        [3.0, 15.0, 25.0],
        [0.35, 0.36, 0.37],
      )
      release_progress = min(self.upshift_timer / 0.30, 1.0)
      self.upshift_cap = self.upshift_entry_output + (hold6_cap - self.upshift_entry_output) * release_progress
      output_accel = min(output_accel, self.upshift_cap)
      self.upshift_limit_active = True

      tcu_requests_down = target_gear_valid and target_gear < 6
      weak_sixth = self.upshift_timer > 0.45 and CS.aEgo < 0.06 and dv_kph > 5.0
      sixth_done = v_ego_kph >= 94.0 or dv_kph < 6.0
      sixth_lost = gear_valid and current_gear < 6

      if tcu_requests_down or weak_sixth or sixth_done or sixth_lost or self.upshift_timer >= 5.0:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.35
        self.upshift_entry_gear = current_gear if gear_valid else 0

    # ------------------------------------------------------------------
    # v1.5.6 CRUISE SPEED FAIL-SAFE (LongControl layer)
    #
    # Independent of planner/MPC.  If the displayed vehicle speed exceeds
    # the applied displayed cruise target, positive acceleration is no longer
    # allowed to persist.  Driver accelerator override is intentionally left
    # untouched.
    # ------------------------------------------------------------------
    self.cruise_guard_cap = 0.0
    self.cruise_overspeed_kph = 0.0
    self.cruise_guard_active = False
    cruise_overspeed_kph = v_ego_cluster_kph - cruise_target_kph
    cruise_guard_valid = 1.0 <= cruise_target_kph <= 200.0 and v_ego_cluster_kph > 0.5

    if (self.long_control_state == LongCtrlState.pid and not CS.gasPressed and
        cruise_guard_valid and cruise_overspeed_kph > 0.5):
      cruise_guard_cap = interp(
        cruise_overspeed_kph,
        [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
        [0.05, 0.00, -0.05, -0.10, -0.20, -0.35, -0.60],
      )
      output_accel = min(output_accel, cruise_guard_cap)
      self.cruise_guard_cap = float(cruise_guard_cap)
      self.cruise_overspeed_kph = float(cruise_overspeed_kph)
      self.cruise_guard_active = True

    self.last_output_accel = clip(output_accel, accel_limits[0], accel_limits[1])
    self.pos_accel_cut = max(self.raw_output_accel - self.last_output_accel, 0.0)

    # v1.5.3 debug:
    # M: 0 normal / 1 pre / 2 shift / 3 post / 4 hold6 / 5 protect
    self.debugLoCText = (
      f"LC R={self.raw_output_accel:.2f} O={self.last_output_accel:.2f}"
      f" M={self.upshift_state} G={current_gear}>{target_gear if target_gear_valid else 0}"
      f" TR={int(assist_rpm)} S/H={int(self.upshift_soft_rpm)}/{int(self.upshift_hard_rpm)}"
      f" CT={cruise_target_kph:.0f} DV={dv_kph:.1f}"
      f" CG={self.cruise_guard_cap:.2f} OV={self.cruise_overspeed_kph:.1f}"
      f" C={self.upshift_cap:.2f} T={self.upshift_timer:.2f}"
      f" ML={int(self.upshift_state == 2 and self.upshift_entry_gear == 5 and self.upshift_timer > 0.65)}"
      f" SD={int(self.upshift_shift_detected)} AE={CS.aEgo:.2f}"
      f" LS={int(self.lead_start_status)}/{int(self.lead_start_moving)}/{int(self.lead_start_stationary)}"
      f" LV={self.lead_start_v:.1f} LD={self.lead_start_d:.1f}"
      f" V18={self.v_target_start_lookahead:.2f}"
    )

    return self.last_output_accel, -0.5 if planned_stop else j_target
