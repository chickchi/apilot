from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.controls.lib.drive_helpers import CONTROL_N, apply_deadzone
from selfdrive.controls.lib.pid import PIDController
from selfdrive.controls.lib.tcu_downshift_relief import TcuDownshiftRelief
from selfdrive.modeld.constants import T_IDXS
from common.params import Params
from selfdrive.swaglog import cloudlog




LongCtrlState = car.CarControl.Actuators.LongControlState




### apilot
def long_control_state_trans(
  CP,
  active,
  long_control_state,
  v_ego,
  v_target,
  v_target_1sec,
  v_target_1p8sec,
  brake_pressed,
  cruise_standstill,
  softHold,
  a_target_now,
  lead_status=False,
  lead_d=0.0,
  lead_v=0.0,
  lead_vrel=0.0,
):
  cruise_standstill = (
    cruise_standstill and
    not CP.enableGasInterceptor
  )


  accelerating_1sec = (
    v_target_1sec >
    (v_target + 0.01)
  )


  accelerating_1p8sec = (
    v_target_1p8sec >
    (v_target + 0.01)
  )


  planned_stop = (
    v_target < CP.vEgoStopping and
    v_target_1sec < CP.vEgoStopping and
    not accelerating_1sec
  )


  stay_stopped = (
    v_ego < CP.vEgoStopping and
    (
      brake_pressed or
      cruise_standstill
    )
  )


  stopping_condition = (
    planned_stop or
    stay_stopped
  )


  normal_start = (
    v_target_1sec > CP.vEgoStarting and
    accelerating_1sec and
    not cruise_standstill and
    not brake_pressed
  )


  nearby_lead = (
    lead_status and
    0.0 < lead_d < 35.0
  )


  # A stopped lead must remain a stop gate even while ego is still approaching.
  # Do not use abs(vRel) here: a stopped lead naturally has a large negative
  # vRel while ego is still moving.
  stationary_lead_gate = (
    nearby_lead and
    lead_d < 25.0 and
    lead_v < 0.50 and
    v_ego < 5.0
  )


  moving_lead_start = (
    nearby_lead and
    lead_v >
    max(
      float(CP.vEgoStarting),
      0.65,
    ) and
    lead_vrel > 0.08 and
    v_target_1p8sec >
    CP.vEgoStarting and
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


  started_condition = (
    v_ego >
    CP.vEgoStarting
  )


  if not active:
    long_control_state = LongCtrlState.off


  else:
    if long_control_state in (
      LongCtrlState.off,
      LongCtrlState.pid,
    ):
      long_control_state = LongCtrlState.pid


      if (
        stopping_condition and
        a_target_now > -1.0
      ):
        long_control_state = LongCtrlState.stopping


    elif long_control_state == LongCtrlState.stopping:
      if (
        starting_condition and
        CP.startingState
      ):
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


  return (
    long_control_state,
    planned_stop,
  )




class LongControl:
  def __init__(self, CP):
    self.CP = CP


    self.long_control_state = LongCtrlState.off


    self.pid = PIDController(
      (
        CP.longitudinalTuning.kpBP,
        CP.longitudinalTuning.kpV,
      ),
      (
        CP.longitudinalTuning.kiBP,
        CP.longitudinalTuning.kiV,
      ),
      k_f=CP.longitudinalTuning.kf,
      rate=1 / DT_CTRL,
    )


    self.v_pid = 0.0
    self.last_output_accel = 0.0


    self.debugLoCText = ""


    self.readParamCount = 0


    self.longitudinalTuningKpV = 1.0
    self.longitudinalTuningKiV = 0.0
    self.longitudinalTuningKf = 1.0


    self.startAccelApply = 0.0
    self.stopAccelApply = 0.0


    # v1.3 smooth positive acceleration
    self.raw_output_accel = 0.0
    self.pos_accel_jerk_limit = 0.0
    self.pos_accel_headroom = 0.0
    self.pos_accel_comfort_cap = 0.0
    self.pos_accel_cut = 0.0
    self.pos_accel_limited = False
    self.pos_accel_jerk_limited = False
    self.pos_accel_headroom_limited = False


    # v1.5.6 cruise-speed fail-safe
    self.cruise_guard_cap = 0.0
    self.cruise_overspeed_kph = 0.0
    self.cruise_guard_active = False


    # v1.5.7 stop/restart diagnostics
    self.lead_start_status = False
    self.lead_start_moving = False
    self.lead_start_stationary = False
    self.lead_start_v = 0.0
    self.lead_start_d = 0.0
    self.v_target_start_lookahead = 0.0


    # Stop-release safety/diagnostics.
    # reset() is called every frame during STOPPING, therefore these must not
    # be cleared in reset().
    self.stop_lead_latch_timer = 0.0
    self.stop_depart_confirm_timer = 0.0
    self.stop_depart_confirmed = False
    self.stop_guard_active = False
    self.prev_stop_guard_active = False
    self.prev_lead_source = None
    self.lead_source = "-"
    self.lead_source_changed = False


    # v1.5.8 clear-road / transmission state
    self.prev_lane_change_active = False
    self.prev_close_lead = False


    self.clear_lead_confirm_timer = 0.0
    self.clear_road_recovery_timer = 0.0
    self.clear_road_recovery = False


    self.hold6_down_request_timer = 0.0
    self.load_pre_shift_dbg = False


    # v1.6.1 observation-only downshift load relief
    self.downshift_relief = TcuDownshiftRelief()
    self.downshift_relief_state = 0
    self.downshift_relief_cap = 0.0
    self.downshift_relief_active = False
    self.downshift_relief_suppress_legacy = False
    self.downshift_relief_target_timer = 0.0
    self.downshift_relief_cooldown = 0.0


    # Adaptive TCU Load Manager
    # 0 NORMAL
    # 1 PRE_RELIEF
    # 2 SHIFT
    # 3 POST_SHIFT
    # 4 HOLD6
    # 5 RPM_PROTECT
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


    self.g5_upshift_nudge_active = False


    self.longitudinalActuatorDelayLowerBound = (
      float(
        int(
          Params().get(
            "LongitudinalActuatorDelayLowerBound",
            encoding="utf8",
          )
        )
      ) * 0.01
    )


    self.longitudinalActuatorDelayUpperBound = (
      float(
        int(
          Params().get(
            "LongitudinalActuatorDelayUpperBound",
            encoding="utf8",
          )
        )
      ) * 0.01
    )


  def reset(self, v_pid):
    self.pid.reset()
    self.v_pid = v_pid


    self.upshift_state = 0
    self.upshift_timer = 0.0
    self.upshift_cooldown = 0.0
    self.upshift_candidate_timer = 0.0


    self.upshift_entry_output = 0.0
    self.upshift_entry_speed = 0.0
    self.g5_upshift_nudge_active = False
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


    self.hold6_down_request_timer = 0.0
    self.load_pre_shift_dbg = False


    self.downshift_relief.reset()
    self.downshift_relief_state = 0
    self.downshift_relief_cap = 0.0
    self.downshift_relief_active = False
    self.downshift_relief_suppress_legacy = False
    self.downshift_relief_target_timer = 0.0
    self.downshift_relief_cooldown = 0.0


    self.prev_close_lead = False


    self.clear_lead_confirm_timer = 0.0
    self.clear_road_recovery_timer = 0.0
    self.clear_road_recovery = False


  def update(
    self,
    active,
    CS,
    long_plan,
    accel_limits,
    t_since_plan,
    CC,
    v_cruise_kph_apply,
    radar_state=None,
    lane_change_active=False,
  ):
    self.readParamCount += 1


    if self.readParamCount >= 100:
      self.readParamCount = 0


    elif self.readParamCount == 10:
      self.longitudinalTuningKpV = (
        float(
          int(
            Params().get(
              "LongitudinalTuningKpV",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


      self.longitudinalTuningKiV = (
        float(
          int(
            Params().get(
              "LongitudinalTuningKiV",
              encoding="utf8",
            )
          )
        ) * 0.001
      )


      self.longitudinalTuningKf = (
        float(
          int(
            Params().get(
              "LongitudinalTuningKf",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


      if (
        len(self.CP.longitudinalTuning.kpBP) == 1 and
        len(self.CP.longitudinalTuning.kiBP) == 1
      ):
        self.CP.longitudinalTuning.kpV = [
          self.longitudinalTuningKpV
        ]


        self.CP.longitudinalTuning.kiV = [
          self.longitudinalTuningKiV
        ]


        self.pid._k_p = (
          self.CP.longitudinalTuning.kpBP,
          self.CP.longitudinalTuning.kpV,
        )


        self.pid._k_i = (
          self.CP.longitudinalTuning.kiBP,
          self.CP.longitudinalTuning.kiV,
        )


        self.pid.k_f = self.longitudinalTuningKf


    elif self.readParamCount == 30:
      self.longitudinalActuatorDelayLowerBound = (
        float(
          int(
            Params().get(
              "LongitudinalActuatorDelayLowerBound",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


      self.longitudinalActuatorDelayUpperBound = (
        float(
          int(
            Params().get(
              "LongitudinalActuatorDelayUpperBound",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


    elif self.readParamCount == 40:
      self.startAccelApply = (
        float(
          int(
            Params().get(
              "StartAccelApply",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


      self.stopAccelApply = (
        float(
          int(
            Params().get(
              "StopAccelApply",
              encoding="utf8",
            )
          )
        ) * 0.01
      )


    speeds = long_plan.speeds
    a_target_now = 0.0


    if len(speeds) == CONTROL_N:
      v_target_now = interp(
        t_since_plan,
        T_IDXS[:CONTROL_N],
        speeds,
      )


      a_target_now = interp(
        t_since_plan,
        T_IDXS[:CONTROL_N],
        long_plan.accels,
      )


      j_target = long_plan.jerks[0]


      v_target_lower = interp(
        self.longitudinalActuatorDelayLowerBound +
        t_since_plan,
        T_IDXS[:CONTROL_N],
        speeds,
      )


      a_target_lower = (
        2 *
        (v_target_lower - v_target_now) /
        self.longitudinalActuatorDelayLowerBound -
        a_target_now
      )


      v_target_upper = interp(
        self.longitudinalActuatorDelayUpperBound +
        t_since_plan,
        T_IDXS[:CONTROL_N],
        speeds,
      )


      a_target_upper = (
        2 *
        (v_target_upper - v_target_now) /
        self.longitudinalActuatorDelayUpperBound -
        a_target_now
      )


      v_target = min(
        v_target_lower,
        v_target_upper,
      )


      a_target = min(
        a_target_lower,
        a_target_upper,
      )


      v_target_1sec = interp(
        self.longitudinalActuatorDelayLowerBound +
        t_since_plan +
        1.0,
        T_IDXS[:CONTROL_N],
        speeds,
      )


      v_target_1p8sec = interp(
        self.longitudinalActuatorDelayLowerBound +
        t_since_plan +
        1.8,
        T_IDXS[:CONTROL_N],
        speeds,
      )


    else:
      v_target = 0.0
      v_target_now = 0.0
      v_target_1sec = 0.0
      v_target_1p8sec = 0.0


      a_target = 0.0
      a_target_now = 0.0
      j_target = 0.0


      a_target_lower = 0.0
      a_target_upper = 0.0


    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]


    self.CP.startingState = (
      True
      if self.startAccelApply > 0.0
      else False
    )


    self.CP.startAccel = (
      2.0 *
      self.startAccelApply
    )


    self.CP.stopAccel = (
      -2.0 *
      self.stopAccelApply
    )


    output_accel = self.last_output_accel


    # ---------------------------------------------------------------------
    # Lead-aware stopping release
    # ---------------------------------------------------------------------
    self.lead_start_status = False
    self.lead_start_moving = False
    self.lead_start_stationary = False


    self.lead_start_v = 0.0
    self.lead_start_d = 0.0


    self.v_target_start_lookahead = float(
      v_target_1p8sec
    )


    lead_vrel = 0.0
    current_lead_source = "-"
    self.lead_source_changed = False


    if radar_state is not None:
      lead_one = radar_state.leadOne


      if lead_one.status:
        self.lead_start_status = True
        self.lead_start_v = max(
          float(lead_one.vLead),
          0.0,
        )
        self.lead_start_d = max(
          float(lead_one.dRel),
          0.0,
        )


        lead_vrel = float(lead_one.vRel)


        current_lead_source = (
          "R"
          if bool(lead_one.radar)
          else "V"
        )


        self.lead_start_stationary = (
          self.lead_start_d < 25.0 and
          self.lead_start_v < 0.50 and
          CS.vEgo < 5.0
        )


        self.lead_start_moving = (
          self.lead_start_d < 35.0 and
          self.lead_start_v >
          max(
            float(self.CP.vEgoStarting),
            0.65,
          ) and
          lead_vrel > 0.08
        )


    self.lead_source_changed = (
      active and
      self.prev_lead_source in ("R", "V") and
      current_lead_source in ("R", "V") and
      current_lead_source != self.prev_lead_source
    )


    if self.lead_source_changed:
      cloudlog.warning(
        "[LEAD_SRC] {}->{} ego={:.2f} d={:.2f} vLead={:.2f} vRel={:.2f}"
        .format(
          self.prev_lead_source,
          current_lead_source,
          CS.vEgo,
          self.lead_start_d,
          self.lead_start_v,
          lead_vrel,
        )
      )


    self.lead_source = current_lead_source


    stopped_lead_observed = (
      self.lead_start_status and
      0.0 < self.lead_start_d < 25.0 and
      self.lead_start_v < 0.50 and
      CS.vEgo < 5.0 and
      not CS.gasPressed
    )


    # Driver GAS override:
    # 운전자가 직접 가속페달을 밟으면 STOP latch/출발확인 상태를 즉시 해제한다.
    if CS.gasPressed:
      self.stop_lead_latch_timer = 0.0
      self.stop_depart_confirm_timer = 0.0
      self.stop_depart_confirmed = False


    elif stopped_lead_observed:
      self.stop_lead_latch_timer = 0.75


    else:
      self.stop_lead_latch_timer = max(
        self.stop_lead_latch_timer -
        DT_CTRL,
        0.0,
      )


    depart_candidate = (
      self.lead_start_status and
      self.lead_start_d < 35.0 and
      self.lead_start_v >
      max(
        float(self.CP.vEgoStarting),
        0.65,
      ) and
      lead_vrel > 0.08 and
      v_target_1p8sec >
      self.CP.vEgoStarting and
      not CS.brakePressed and
      not CS.gasPressed and
      not self.lead_source_changed
    )


    if depart_candidate:
      self.stop_depart_confirm_timer = min(
        self.stop_depart_confirm_timer +
        DT_CTRL,
        1.0,
      )
    else:
      self.stop_depart_confirm_timer = 0.0


    self.stop_depart_confirmed = (
      self.stop_depart_confirm_timer >= 0.30
    )


    if self.stop_depart_confirmed:
      self.stop_lead_latch_timer = 0.0


    if not active:
      self.stop_lead_latch_timer = 0.0
      self.stop_depart_confirm_timer = 0.0
      self.stop_depart_confirmed = False
      self.prev_lead_source = None
    else:
      self.prev_lead_source = (
        current_lead_source
        if current_lead_source in ("R", "V")
        else None
      )


    # ---------------------------------------------------------------------
    # v1.5.8 clear-road recovery
    # ---------------------------------------------------------------------
    if lane_change_active:
      self.clear_road_recovery_timer = 0.0


    elif self.prev_lane_change_active:
      self.clear_road_recovery_timer = 3.0


    else:
      self.clear_road_recovery_timer = max(
        self.clear_road_recovery_timer -
        DT_CTRL,
        0.0,
      )


    self.prev_lane_change_active = bool(
      lane_change_active
    )


    cluster_kph_for_clear = float(
      CS.vEgoCluster * 3.6
    )


    if cluster_kph_for_clear <= 0.5:
      cluster_kph_for_clear = float(
        CS.vEgo * 3.6
      )


    cruise_gap_for_clear = (
      float(v_cruise_kph_apply) -
      cluster_kph_for_clear
    )


    clear_lead_for_recovery = (
      not self.lead_start_status or
      self.lead_start_d > 70.0 or
      (
        self.lead_start_d > 50.0 and
        lead_vrel > -0.30
      )
    )


    if clear_lead_for_recovery:
      self.clear_lead_confirm_timer = min(
        self.clear_lead_confirm_timer +
        DT_CTRL,
        1.0,
      )


    else:
      self.clear_lead_confirm_timer = 0.0


    clear_lead_confirmed = (
      self.clear_lead_confirm_timer >= 0.30
    )


    close_lead_now = (
      self.lead_start_status and
      self.lead_start_d < 45.0
    )


    if (
      self.prev_close_lead and
      not close_lead_now
    ):
      self.clear_road_recovery_timer = max(
        self.clear_road_recovery_timer,
        2.0,
      )


    self.prev_close_lead = bool(
      close_lead_now
    )


    self.clear_road_recovery = bool(
      self.clear_road_recovery_timer > 0.0 and
      clear_lead_confirmed and
      cruise_gap_for_clear > 4.0 and
      CS.vEgo > 8.0 and
      not CS.gasPressed and
      not CS.brakePressed
    )


    prev_long_control_state = (
      self.long_control_state
    )


    next_long_control_state, planned_stop = (
      long_control_state_trans(
        self.CP,
        active,
        self.long_control_state,
        CS.vEgo,
        v_target,
        v_target_1sec,
        v_target_1p8sec,
        CS.brakePressed,
        CS.cruiseState.standstill,
        CC.hudControl.softHold,
        a_target_now,
        self.lead_start_status,
        self.lead_start_d,
        self.lead_start_v,
        lead_vrel,
      )
    )


    # This guard never creates a new STOPPING state.
    # It only prevents an already-stopping car from releasing braking because
    # of a short lead/source/planner discontinuity.
    self.stop_guard_active = bool(
      active and
      prev_long_control_state ==
      LongCtrlState.stopping and
      next_long_control_state !=
      LongCtrlState.stopping and
      self.stop_lead_latch_timer > 0.0 and
      not self.stop_depart_confirmed and
      not CS.gasPressed and
      not CC.hudControl.softHold
    )


    if self.stop_guard_active:
      next_long_control_state = (
        LongCtrlState.stopping
      )


    self.long_control_state = (
      next_long_control_state
    )


    if (
      self.stop_guard_active !=
      self.prev_stop_guard_active
    ):
      cloudlog.warning(
        "[STOP_GUARD] {} ego={:.2f} d={:.2f} vLead={:.2f} vRel={:.2f} "
        "src={} latch={:.2f} depart={:.2f} v0={:.2f} v1={:.2f} v18={:.2f}"
        .format(
          "ON"
          if self.stop_guard_active
          else "OFF",
          CS.vEgo,
          self.lead_start_d,
          self.lead_start_v,
          lead_vrel,
          self.lead_source,
          self.stop_lead_latch_timer,
          self.stop_depart_confirm_timer,
          v_target_now,
          v_target_1sec,
          v_target_1p8sec,
        )
      )


    if (
      self.long_control_state !=
      prev_long_control_state
    ):
      cloudlog.warning(
        "[STOP_STATE] {}->{} ego={:.2f} d={:.2f} vLead={:.2f} vRel={:.2f} "
        "src={} guard={} v0={:.2f} v1={:.2f} v18={:.2f} a0={:.2f}"
        .format(
          int(prev_long_control_state),
          int(self.long_control_state),
          CS.vEgo,
          self.lead_start_d,
          self.lead_start_v,
          lead_vrel,
          self.lead_source,
          int(self.stop_guard_active),
          v_target_now,
          v_target_1sec,
          v_target_1p8sec,
          a_target_now,
        )
      )


    self.prev_stop_guard_active = bool(
      self.stop_guard_active
    )


    if self.long_control_state == LongCtrlState.off:
      self.reset(CS.vEgo)
      output_accel = 0.0


    elif self.long_control_state == LongCtrlState.stopping:
      if output_accel > self.CP.stopAccel:
        output_accel = min(
          output_accel,
          0.0,
        )


        output_accel -= (
          self.CP.stoppingDecelRate *
          DT_CTRL
        )


        if CC.hudControl.softHold:
          output_accel = self.CP.stopAccel


      self.reset(CS.vEgo)


    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset(CS.vEgo)


    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target_now


      prevent_overshoot = (
        not self.CP.stoppingControl and
        CS.vEgo < 1.5 and
        v_target_1sec < 0.7 and
        v_target_1sec < self.v_pid
      )


      deadzone = interp(
        CS.vEgo,
        self.CP.longitudinalTuning.deadzoneBP,
        self.CP.longitudinalTuning.deadzoneV,
      )


      freeze_integrator = prevent_overshoot


      error = self.v_pid - CS.vEgo


      error_deadzone = apply_deadzone(
        error,
        deadzone,
      )


      output_accel = self.pid.update(
        error_deadzone,
        speed=CS.vEgo,
        feedforward=a_target,
        freeze_integrator=freeze_integrator,
      )


    # ---------------------------------------------------------------------
    # Existing positive acceleration / transmission logic
    # ---------------------------------------------------------------------
    self.raw_output_accel = float(
      output_accel
    )


    self.pos_accel_jerk_limit = 0.0
    self.pos_accel_headroom = 0.0
    self.pos_accel_comfort_cap = 0.0
    self.pos_accel_cut = 0.0


    self.pos_accel_limited = False
    self.pos_accel_jerk_limited = False
    self.pos_accel_headroom_limited = False


    if (
      self.long_control_state ==
      LongCtrlState.pid and
      output_accel > 0.0
    ):
      v_ego_kph = CS.vEgo * 3.6


      self.pos_accel_jerk_limit = interp(
        v_ego_kph,
        [
          0.0,
          20.0,
          40.0,
          60.0,
          80.0,
          100.0,
          120.0,
          140.0,
        ],
        [
          0.45,
          0.38,
          0.30,
          0.24,
          0.18,
          0.15,
          0.13,
          0.12,
        ],
      )


      if self.clear_road_recovery:
        self.pos_accel_jerk_limit *= 1.50


      self.pos_accel_headroom = interp(
        v_ego_kph,
        [
          0.0,
          20.0,
          40.0,
          60.0,
          70.0,
          80.0,
          100.0,
          120.0,
          140.0,
        ],
        [
          0.20,
          0.20,
          0.15,
          0.08,
          0.05,
          0.05,
          0.04,
          0.04,
          0.04,
        ],
      )


      self.pos_accel_comfort_cap = (
        max(
          a_target,
          0.0,
        ) +
        self.pos_accel_headroom
      )


      positive_base = max(
        self.last_output_accel,
        0.0,
      )


      positive_rise_max = (
        positive_base +
        self.pos_accel_jerk_limit *
        DT_CTRL
      )


      raw_positive = output_accel


      output_accel = min(
        raw_positive,
        positive_rise_max,
        self.pos_accel_comfort_cap,
      )


      eps = 1e-5


      self.pos_accel_jerk_limited = (
        positive_rise_max + eps <
        raw_positive and
        positive_rise_max <=
        self.pos_accel_comfort_cap +
        eps
      )


      self.pos_accel_headroom_limited = (
        self.pos_accel_comfort_cap +
        eps <
        raw_positive and
        self.pos_accel_comfort_cap <=
        positive_rise_max +
        eps
      )


      self.pos_accel_limited = (
        output_accel + eps <
        raw_positive
      )


    # =====================================================================
    # ADAPTIVE TCU LOAD MANAGER
    # =====================================================================
    self.upshift_limit_active = False
    self.upshift_cap = 0.0


    v_ego_kph = CS.vEgo * 3.6


    v_ego_cluster_kph = float(
      CS.vEgoCluster *
      3.6
    )


    if v_ego_cluster_kph <= 0.5:
      v_ego_cluster_kph = v_ego_kph


    cruise_target_kph = float(
      v_cruise_kph_apply
    )


    if not (
      1.0 <=
      cruise_target_kph <=
      200.0
    ):
      cruise_target_kph = (
        v_ego_cluster_kph
      )


    dv_kph = max(
      cruise_target_kph -
      v_ego_cluster_kph,
      0.0,
    )


    # ---------------------------------------------------------------------
    # v1.8.2 CLEAR-ROAD CRUISE CATCH-UP
    #
    # v1.8.1 road test: with CT around 100 km/h and ego around 80 km/h,
    # raw PID demand could be ~0.8 m/s^2 while the positive comfort headroom
    # reduced the final request to ~0.2 m/s^2. On a clear road this can be
    # too small to overcome drag/grade, so the car may never reach the
    # selected cruise speed.
    #
    # Re-open only the comfort/headroom cap. The normal positive jerk-rise
    # limit, raw PID demand, planner limits, TCU manager, downshift relief,
    # cruise overspeed guard, and driver overrides remain authoritative.
    # ---------------------------------------------------------------------
    self.cruise_recovery_active = False
    self.cruise_recovery_cap = 0.0


    clear_cruise_recovery = (
      self.long_control_state == LongCtrlState.pid and
      self.raw_output_accel > 0.0 and
      dv_kph > 6.0 and
      not CS.gasPressed and
      not CS.brakePressed and
      (
        not self.lead_start_status or
        self.lead_start_d > 50.0
      ) and
      CS.aEgo < 0.15
    )


    if clear_cruise_recovery:
      recovery_cap = interp(
        dv_kph,
        [6.0, 10.0, 15.0, 25.0],
        [0.28, 0.34, 0.42, 0.50],
      )


      recovered_accel = min(
        self.raw_output_accel,
        positive_rise_max,
        recovery_cap,
      )


      if recovered_accel > output_accel:
        output_accel = recovered_accel
        self.cruise_recovery_active = True


      self.cruise_recovery_cap = float(
        recovery_cap
      )


    engine_rpm = float(
      CS.engineRpm
    )


    tcu_rpm = float(
      CS.tcuRpm
    )


    current_gear = int(
      CS.currentGear
    )


    target_gear = int(
      CS.targetGear
    )


    gear_valid = (
      1 <=
      current_gear <=
      8
    )


    target_gear_valid = (
      1 <=
      target_gear <=
      8
    )


    assist_rpm = (
      engine_rpm
      if engine_rpm > 700.0
      else tcu_rpm
    )


    rpm_valid = (
      assist_rpm >
      700.0
    )


    driver_override = (
      CS.gasPressed or
      CS.brakePressed
    )


    positive_control = (
      self.long_control_state ==
      LongCtrlState.pid and
      output_accel > 0.0
    )


    base_context = (
      positive_control and
      not driver_override and
      dv_kph > 2.5
    )


    g5_final_approach_context = (
      positive_control and
      not driver_override and
      gear_valid and
      current_gear == 5 and
      v_ego_kph >= 80.0 and
      dv_kph > 0.5
    )


    upshift_context = (
      base_context or
      g5_final_approach_context
    )


    if self.upshift_cooldown > 0.0:
      self.upshift_cooldown = max(
        self.upshift_cooldown -
        DT_CTRL,
        0.0,
      )


    soft_min_v = 0.0
    hard_min_v = 0.0


    soft_rpm_base = 0.0
    hard_rpm_base = 0.0


    soft_cap_base = 0.0
    shift_cap_base = 0.0
    protect_cap_base = 0.0


    if current_gear == 3:
      soft_min_v = 45.0
      hard_min_v = 42.0
      soft_rpm_base = 2100.0
      hard_rpm_base = 2350.0
      soft_cap_base = 0.42
      shift_cap_base = 0.34
      protect_cap_base = 0.46


    elif current_gear == 4:
      soft_min_v = 54.0
      hard_min_v = 50.0
      soft_rpm_base = 2150.0
      hard_rpm_base = 2375.0
      soft_cap_base = 0.38
      shift_cap_base = 0.29
      protect_cap_base = 0.40


    elif current_gear == 5:
      soft_min_v = 80.0
      hard_min_v = 76.0
      soft_rpm_base = 2050.0
      hard_rpm_base = 2250.0
      soft_cap_base = 0.27
      shift_cap_base = 0.18
      protect_cap_base = 0.32


    elif not gear_valid:
      soft_min_v = 82.0
      hard_min_v = 78.0
      soft_rpm_base = 2500.0
      hard_rpm_base = 2750.0
      soft_cap_base = 0.30
      shift_cap_base = 0.22
      protect_cap_base = 0.34


    demand_rpm_boost_raw = interp(
      dv_kph,
      [0.0, 20.0, 40.0, 60.0],
      [0.0, 0.0, 120.0, 220.0],
    )


    if current_gear == 3:
      demand_rpm_boost = demand_rpm_boost_raw * 0.80
    elif current_gear in (4, 5):
      demand_rpm_boost = 0.0
    else:
      demand_rpm_boost = demand_rpm_boost_raw


    demand_cap_boost = interp(
      dv_kph,
      [0.0, 20.0, 40.0, 60.0],
      [0.0, 0.0, 0.03, 0.05],
    )


    self.upshift_soft_rpm = (
      soft_rpm_base +
      demand_rpm_boost
      if soft_rpm_base > 0.0
      else 0.0
    )


    self.upshift_hard_rpm = (
      hard_rpm_base +
      demand_rpm_boost
      if hard_rpm_base > 0.0
      else 0.0
    )


    self.upshift_soft_cap = (
      soft_cap_base +
      demand_cap_boost
      if soft_cap_base > 0.0
      else 0.0
    )


    self.upshift_shift_cap = (
      shift_cap_base +
      demand_cap_boost
      if shift_cap_base > 0.0
      else 0.0
    )


    self.upshift_protect_cap = (
      protect_cap_base +
      demand_cap_boost
      if protect_cap_base > 0.0
      else 0.0
    )


    tg_up = (
      target_gear_valid and
      gear_valid and
      target_gear >
      current_gear
    )


    tg_down = (
      target_gear_valid and
      gear_valid and
      target_gear <
      current_gear
    )


    downshift_relief = self.downshift_relief.update(
      DT_CTRL,
      positive_control,
      driver_override,
      self.raw_output_accel,
      output_accel,
      v_ego_cluster_kph,
      dv_kph,
      current_gear,
      target_gear,
      target_gear_valid,
      CS.aEgo,
      assist_rpm,
    )


    self.downshift_relief_state = int(
      downshift_relief.state
    )


    self.downshift_relief_cap = float(
      downshift_relief.cap
    )


    self.downshift_relief_active = bool(
      downshift_relief.active
    )


    self.downshift_relief_suppress_legacy = bool(
      downshift_relief.suppress_legacy
    )


    self.downshift_relief_target_timer = float(
      downshift_relief.target_down_timer
    )


    self.downshift_relief_cooldown = float(
      downshift_relief.cooldown
    )


    if downshift_relief.actual_downshift:
      self.upshift_state = 0
      self.upshift_timer = 0.0
      self.upshift_candidate_timer = 0.0


      downshift_retry_cooldown = 0.65


      self.upshift_cooldown = max(
        self.upshift_cooldown,
        downshift_retry_cooldown,
      )


    if current_gear == 5:
      accel_response_ok = (
        CS.aEgo >
        -0.03
      )


    elif current_gear == 4:
      accel_response_ok = (
        CS.aEgo >
        0.00
      )


    else:
      accel_response_ok = (
        CS.aEgo >
        0.08
      )


    strong_response = (
      CS.aEgo >
      0.15
    )


    gear_managed = (
      current_gear in (
        3,
        4,
        5,
      )
      if gear_valid
      else rpm_valid
    )


    soft_rpm_candidate = (
      upshift_context and
      gear_managed and
      rpm_valid and
      accel_response_ok and
      v_ego_kph >= soft_min_v and
      assist_rpm >= self.upshift_soft_rpm
    )


    hard_rpm_candidate = (
      upshift_context and
      gear_managed and
      rpm_valid and
      accel_response_ok and
      v_ego_kph >= hard_min_v and
      assist_rpm >= self.upshift_hard_rpm
    )


    tcu_up_candidate = (
      upshift_context and
      gear_valid and
      current_gear in (3, 4, 5) and
      tg_up and
      accel_response_ok and
      v_ego_kph >= max(
        hard_min_v,
        soft_min_v - 4.0,
      ) and
      (
        not rpm_valid or
        assist_rpm >= max(
          self.upshift_soft_rpm - 150.0,
          1200.0,
        )
      )
    )


    rpm_protect_candidate = (
      base_context and
      gear_valid and
      current_gear in (3, 4, 5) and
      tg_down and
      rpm_valid and
      strong_response and
      v_ego_kph >= hard_min_v and
      assist_rpm >= self.upshift_hard_rpm
    )


    load_pre_candidate = (
      upshift_context and
      gear_valid and
      accel_response_ok and
      (
        (
          current_gear == 4 and
          v_ego_kph >= 54.0 and
          dv_kph >= 8.0 and
          self.raw_output_accel >= 0.36
        ) or
        (
          current_gear == 5 and
          v_ego_kph >= 80.0 and
          dv_kph >= 0.8 and
          self.raw_output_accel >= 0.26 and
          (
            not rpm_valid or
            assist_rpm >= 1950.0
          )
        )
      )
    )


    self.load_pre_shift_dbg = bool(
      load_pre_candidate
    )


    if (
      gear_valid and
      current_gear == 6 and
      target_gear_valid and
      target_gear < 6
    ):
      self.hold6_down_request_timer = min(
        self.hold6_down_request_timer +
        DT_CTRL,
        1.0,
      )


    elif (
      gear_valid and
      current_gear == 6
    ):
      self.hold6_down_request_timer = max(
        self.hold6_down_request_timer -
        2.0 * DT_CTRL,
        0.0,
      )


    else:
      self.hold6_down_request_timer = 0.0


    tcu_down_persistent = (
      self.hold6_down_request_timer >= 0.50
    )


    hold6_context = (
      positive_control and
      not driver_override and
      dv_kph > 0.5
    )


    hold6_candidate = (
      hold6_context and
      gear_valid and
      current_gear == 6 and
      78.0 <= v_ego_kph <= 100.5 and
      0.5 <= dv_kph <= 25.0 and
      output_accel > 0.28 and
      CS.aEgo > 0.05 and
      not tcu_down_persistent
    )


    candidate_state = 0


    if hold6_candidate:
      candidate_state = 4
    elif rpm_protect_candidate:
      candidate_state = 5
    elif tcu_up_candidate or hard_rpm_candidate:
      candidate_state = 2
    elif soft_rpm_candidate or load_pre_candidate:
      candidate_state = 1


    if self.downshift_relief_suppress_legacy:
      candidate_state = 0


      if self.upshift_state != 0:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_candidate_timer = 0.0
        self.upshift_cooldown = max(
          self.upshift_cooldown,
          self.downshift_relief_cooldown,
        )


    if (
      self.upshift_state == 0 and
      self.upshift_cooldown <= 0.0 and
      candidate_state != 0
    ):
      self.upshift_candidate_timer = min(
        self.upshift_candidate_timer +
        DT_CTRL,
        0.30,
      )


      debounce_required = (
        0.10
        if candidate_state in (
          2,
          5,
        )
        else 0.12
      )


      if (
        self.upshift_candidate_timer >=
        debounce_required
      ):
        self.upshift_state = candidate_state
        self.upshift_timer = 0.0


        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


        self.upshift_entry_speed = v_ego_kph


        self.upshift_entry_gear = (
          current_gear
          if gear_valid
          else 0
        )


        self.upshift_post_gear = 0
        self.upshift_shift_detected = False
        self.upshift_candidate_timer = 0.0


    elif self.upshift_state == 0:
      self.upshift_candidate_timer = 0.0


    if (
      self.upshift_state == 4 or
      (
        self.upshift_state in (
          1,
          2,
        ) and
        self.upshift_entry_gear == 5
      )
    ):
      manager_dv_abort = 0.5
    else:
      manager_dv_abort = 1.5


    abort_manager = (
      self.upshift_state != 0 and
      (
        driver_override or
        self.long_control_state !=
        LongCtrlState.pid or
        self.raw_output_accel <= 0.0 or
        dv_kph <= manager_dv_abort
      )
    )


    if abort_manager:
      self.upshift_state = 0
      self.upshift_timer = 0.0
      self.upshift_candidate_timer = 0.0
      self.upshift_cooldown = 0.20
      self.upshift_cap = 0.0


    gear_increased = (
      gear_valid and
      self.upshift_entry_gear > 0 and
      current_gear >
      self.upshift_entry_gear
    )


    gear_decreased = (
      gear_valid and
      self.upshift_entry_gear > 0 and
      current_gear <
      self.upshift_entry_gear
    )


    if not (
      self.upshift_state == 2 and
      self.upshift_entry_gear == 5
    ):
      self.g5_upshift_nudge_active = False


    # M1 PRE_RELIEF
    if self.upshift_state == 1:
      self.upshift_timer += DT_CTRL


      if gear_increased:
        self.upshift_state = 3
        self.upshift_timer = 0.0
        self.upshift_post_gear = current_gear
        self.upshift_shift_detected = True
        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


      elif rpm_protect_candidate:
        self.upshift_state = 5
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


      elif (
        tcu_up_candidate or
        hard_rpm_candidate or
        (
          load_pre_candidate and
          (
            (
              self.upshift_entry_gear == 4 and
              self.upshift_timer >= 0.40
            ) or
            (
              self.upshift_entry_gear == 5 and
              self.upshift_timer >= 0.35
            )
          )
        )
      ):
        self.upshift_state = 2
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


      else:
        target_cap = self.upshift_soft_cap


        release_progress = min(
          self.upshift_timer / 0.40,
          1.0,
        )


        self.upshift_cap = (
          self.upshift_entry_output +
          (
            target_cap -
            self.upshift_entry_output
          ) *
          release_progress
        )


        output_accel = min(
          output_accel,
          self.upshift_cap,
        )


        self.upshift_limit_active = True


        if self.upshift_entry_gear == 5:
          weak_response = (
            self.upshift_timer > 0.70 and
            CS.aEgo < -0.08 and
            dv_kph > 5.0
          )
        else:
          weak_response = (
            self.upshift_timer > 0.50 and
            CS.aEgo < 0.02 and
            dv_kph > 5.0
          )


        rpm_recovered = (
          rpm_valid and
          not load_pre_candidate and
          self.upshift_timer > 0.30 and
          assist_rpm <
          self.upshift_soft_rpm - 140.0
        )


        if (
          weak_response or
          rpm_recovered or
          self.upshift_timer >= 2.0
        ):
          failed_gear = self.upshift_entry_gear
          self.upshift_state = 0
          self.upshift_timer = 0.0


          self.upshift_cooldown = (
            0.90
            if failed_gear == 5
            else 0.50
          )


    # M2 SHIFT
    elif self.upshift_state == 2:
      self.upshift_timer += DT_CTRL


      if gear_increased:
        self.upshift_state = 3
        self.upshift_timer = 0.0
        self.upshift_post_gear = current_gear
        self.upshift_shift_detected = True
        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


      else:
        if self.upshift_entry_gear == 5:
          positive_plateau_cap = interp(
            dv_kph,
            [0.5, 5.0, 15.0, 35.0],
            [0.14, 0.16, 0.20, 0.22],
          )


          if rpm_valid:
            rpm_plateau_cap = interp(
              assist_rpm,
              [
                1950.0,
                2150.0,
                2300.0,
                2450.0,
                2600.0,
                2750.0,
              ],
              [
                0.22,
                0.22,
                0.20,
                0.18,
                0.16,
                0.14,
              ],
            )
            positive_plateau_cap = min(
              positive_plateau_cap,
              rpm_plateau_cap,
            )


          stubborn_g5 = (
            gear_valid and
            current_gear == 5 and
            target_gear_valid and
            target_gear == 5 and
            rpm_valid and
            assist_rpm >= 2450.0 and
            v_ego_kph >= 88.0 and
            dv_kph >= 2.0 and
            CS.aEgo > -0.05
          )


          if (
            stubborn_g5 and
            self.upshift_timer >= 1.50
          ):
            self.g5_upshift_nudge_active = True


          target_plateau_cap = positive_plateau_cap


          if self.g5_upshift_nudge_active:
            deep_lift_cap = interp(
              self.upshift_timer,
              [1.50, 2.20, 3.00, 3.80],
              [
                positive_plateau_cap,
                0.13,
                0.10,
                0.08,
              ],
            )


            target_plateau_cap = min(
              target_plateau_cap,
              deep_lift_cap,
            )


          release_progress = min(
            self.upshift_timer /
            0.45,
            1.0,
          )


          target_cap = (
            self.upshift_entry_output +
            (
              target_plateau_cap -
              self.upshift_entry_output
            ) *
            release_progress
          )


          shift_timeout = 4.20
          weak_check_time = 1.50


        elif self.upshift_entry_gear == 4:
          release_progress = min(
            self.upshift_timer / 0.32,
            1.0,
          )


          target_cap = (
            self.upshift_entry_output +
            (
              self.upshift_shift_cap -
              self.upshift_entry_output
            ) *
            release_progress
          )


          shift_timeout = 1.35
          weak_check_time = 0.50


        else:
          release_progress = min(
            self.upshift_timer / 0.30,
            1.0,
          )


          target_cap = (
            self.upshift_entry_output +
            (
              self.upshift_shift_cap -
              self.upshift_entry_output
            ) *
            release_progress
          )


          shift_timeout = 1.40
          weak_check_time = 0.50


        self.upshift_cap = target_cap


        output_accel = min(
          output_accel,
          self.upshift_cap,
        )


        self.upshift_limit_active = True


        if self.upshift_entry_gear == 5:
          weak_response = (
            self.upshift_timer >
            weak_check_time and
            CS.aEgo <
            -0.12 and
            dv_kph >
            5.0
          )
        else:
          weak_response = (
            self.upshift_timer >
            weak_check_time and
            CS.aEgo <
            0.01 and
            dv_kph >
            5.0
          )


        if (
          weak_response or
          self.upshift_timer >=
          shift_timeout
        ):
          failed_gear = self.upshift_entry_gear


          self.upshift_state = 0
          self.upshift_timer = 0.0


          if failed_gear == 5:
            self.upshift_cooldown = (
              1.00
              if self.g5_upshift_nudge_active
              else 0.60
            )
            self.g5_upshift_nudge_active = False
          else:
            self.upshift_cooldown = 0.50


    # M5 RPM_PROTECT
    elif self.upshift_state == 5:
      self.upshift_timer += DT_CTRL


      if gear_decreased:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.35


      elif tcu_up_candidate:
        self.upshift_state = 2
        self.upshift_timer = 0.0
        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


      else:
        target_cap = self.upshift_protect_cap


        release_progress = min(
          self.upshift_timer / 0.30,
          1.0,
        )


        self.upshift_cap = (
          self.upshift_entry_output +
          (
            target_cap -
            self.upshift_entry_output
          ) *
          release_progress
        )


        output_accel = min(
          output_accel,
          self.upshift_cap,
        )


        self.upshift_limit_active = True


        protect_resolved = (
          not tg_down or
          (
            rpm_valid and
            assist_rpm <
            self.upshift_hard_rpm -
            150.0
          ) or
          CS.aEgo < 0.05
        )


        if (
          (
            self.upshift_timer > 0.30 and
            protect_resolved
          ) or
          self.upshift_timer >= 1.20
        ):
          self.upshift_state = 0
          self.upshift_timer = 0.0
          self.upshift_cooldown = 0.45


    # M3 POST_SHIFT
    elif self.upshift_state == 3:
      self.upshift_timer += DT_CTRL


      if (
        gear_valid and
        current_gear >
        self.upshift_post_gear >
        0
      ):
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


      output_accel = min(
        output_accel,
        self.upshift_cap,
      )


      self.upshift_limit_active = True


      next_up_ready = (
        self.upshift_timer >= 0.35 and
        gear_valid and
        current_gear in (3, 4, 5) and
        target_gear_valid and
        target_gear > current_gear and
        (
          not rpm_valid or
          assist_rpm >=
          max(
            self.upshift_soft_rpm -
            100.0,
            1200.0,
          )
        )
      )


      if next_up_ready:
        self.upshift_state = 2
        self.upshift_timer = 0.0


        self.upshift_entry_output = max(
          output_accel,
          0.0,
        )


        self.upshift_entry_gear = current_gear


      elif self.upshift_timer >= post_duration:
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.20


        self.upshift_entry_gear = (
          current_gear
          if gear_valid
          else 0
        )


    # M4 HOLD6
    elif self.upshift_state == 4:
      self.upshift_timer += DT_CTRL


      hold6_cap = interp(
        dv_kph,
        [
          0.5,
          1.5,
          3.0,
          15.0,
          25.0,
        ],
        [
          0.10,
          0.20,
          0.30,
          0.36,
          0.37,
        ],
      )


      release_progress = min(
        self.upshift_timer / 0.30,
        1.0,
      )


      self.upshift_cap = (
        self.upshift_entry_output +
        (
          hold6_cap -
          self.upshift_entry_output
        ) *
        release_progress
      )


      output_accel = min(
        output_accel,
        self.upshift_cap,
      )


      self.upshift_limit_active = True


      tcu_requests_down = (
        self.hold6_down_request_timer >=
        0.50
      )


      weak_sixth = (
        self.upshift_timer > 0.60 and
        CS.aEgo < 0.05 and
        dv_kph > 4.0
      )


      sixth_done = (
        dv_kph < 0.5
      )


      sixth_lost = (
        gear_valid and
        current_gear < 6
      )


      if (
        tcu_requests_down or
        weak_sixth or
        sixth_done or
        sixth_lost or
        self.upshift_timer >= 7.0
      ):
        self.upshift_state = 0
        self.upshift_timer = 0.0
        self.upshift_cooldown = 0.35


        self.upshift_entry_gear = (
          current_gear
          if gear_valid
          else 0
        )


    if (
      self.downshift_relief_active and
      output_accel > 0.0
    ):
      output_accel = min(
        output_accel,
        self.downshift_relief_cap,
      )


      self.upshift_limit_active = True


      if (
        self.upshift_cap <= 0.0 or
        self.downshift_relief_cap <
        self.upshift_cap
      ):
        self.upshift_cap = self.downshift_relief_cap


    # Cruise speed fail-safe
    self.cruise_guard_cap = 0.0
    self.cruise_overspeed_kph = 0.0
    self.cruise_guard_active = False


    cruise_overspeed_kph = (
      v_ego_cluster_kph -
      cruise_target_kph
    )


    cruise_guard_valid = (
      1.0 <=
      cruise_target_kph <=
      200.0 and
      v_ego_cluster_kph > 0.5
    )


    if (
      self.long_control_state ==
      LongCtrlState.pid and
      not CS.gasPressed and
      cruise_guard_valid and
      cruise_overspeed_kph > 0.5
    ):
      cruise_guard_cap = interp(
        cruise_overspeed_kph,
        [
          0.5,
          1.0,
          2.0,
          3.0,
          5.0,
          10.0,
          20.0,
        ],
        [
          0.05,
          0.00,
          -0.05,
          -0.10,
          -0.20,
          -0.35,
          -0.60,
        ],
      )


      output_accel = min(
        output_accel,
        cruise_guard_cap,
      )


      self.cruise_guard_cap = float(
        cruise_guard_cap
      )


      self.cruise_overspeed_kph = float(
        cruise_overspeed_kph
      )


      self.cruise_guard_active = True


    self.last_output_accel = clip(
      output_accel,
      accel_limits[0],
      accel_limits[1],
    )


    self.pos_accel_cut = max(
      self.raw_output_accel -
      self.last_output_accel,
      0.0,
    )


    self.debugLoCText = (
      f"LC R{self.raw_output_accel:.2f} "
      f"O{self.last_output_accel:.2f} "
      f"M{self.upshift_state} "
      f"G{current_gear}>"
      f"{target_gear if target_gear_valid else 0} "
      f"TR{int(assist_rpm)} "
      f"CT{cruise_target_kph:.0f} "
      f"D{dv_kph:.1f}"
      f"|C{self.upshift_cap:.2f} "
      f"T{self.upshift_timer:.2f} "
      f"UC{self.upshift_cooldown:.2f} "
      f"J{self.pos_accel_jerk_limit:.2f} "
      f"RC{int(self.cruise_recovery_active)}/{self.cruise_recovery_cap:.2f} "
      f"CR{int(self.clear_road_recovery)} "
      f"LP{int(self.load_pre_shift_dbg)} "
      f"H6D{self.hold6_down_request_timer:.2f}"
      f"|DR{self.downshift_relief_state} "
      f"DC{self.downshift_relief_cap:.2f} "
      f"DT{self.downshift_relief_target_timer:.2f} "
      f"DX{self.downshift_relief_cooldown:.2f}"
      f"|CG{self.cruise_guard_cap:.2f}/"
      f"{self.cruise_overspeed_kph:.1f} "
      f"ML{int(self.upshift_state == 2 and self.upshift_entry_gear == 5 and self.upshift_timer > 0.45)} "
      f"GP{int(self.upshift_state == 2 and self.upshift_entry_gear == 5 and not self.g5_upshift_nudge_active)} "
      f"GN{int(self.g5_upshift_nudge_active)} "
      f"AE{CS.aEgo:.2f} "
      f"LS{int(self.lead_start_status)}/"
      f"{int(self.lead_start_moving)}/"
      f"{int(self.lead_start_stationary)} "
      f"LD{self.lead_start_d:.1f} "
      f"V18{self.v_target_start_lookahead:.2f}"
      f"|ST{int(self.long_control_state)} "
      f"SG{int(self.stop_guard_active)} "
      f"SL{self.stop_lead_latch_timer:.2f} "
      f"SD{self.stop_depart_confirm_timer:.2f} "
      f"SRC{self.lead_source} "
      f"VL{self.lead_start_v:.2f} "
      f"VR{lead_vrel:+.2f} "
      f"V0{v_target_now:.2f} "
      f"V1{v_target_1sec:.2f} "
      f"V18{v_target_1p8sec:.2f}"
    )


    return (
      self.last_output_accel,
      -0.5
      if planned_stop
      else j_target,
    )
