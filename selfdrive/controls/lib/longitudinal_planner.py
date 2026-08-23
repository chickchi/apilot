#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import clip, interp

import cereal.messaging as messaging
from cereal import log
from common.conversions import Conversions as CV
from common.filter_simple import FirstOrderFilter
from common.realtime import DT_MDL
from selfdrive.modeld.constants import T_IDXS
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, MIN_ACCEL, MAX_ACCEL, N
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N, get_speed_error
from selfdrive.swaglog import cloudlog
from common.params import Params

LON_MPC_STEP = 0.2  # first step is 0.2s
A_CRUISE_MIN = -1.2

A_CRUISE_MAX_VALS = [1.2, 1.1, 1.0, 0.85, 0.75]
A_CRUISE_MAX_BP = [0., 40 * CV.KPH_TO_MS, 60 * CV.KPH_TO_MS, 80 * CV.KPH_TO_MS, 110 * CV.KPH_TO_MS, 140 * CV.KPH_TO_MS]

# Lookup table for turns
_A_TOTAL_MAX_V = [2.0, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0):
    self.CP = CP
    self.mpc = LongitudinalMpc()
    self.fcw = False

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, DT_MDL)
    self.v_model_error = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0
    self.params = Params()
    self.param_read_counter = 0
    self.read_param()

    self.vCluRatio = 1.0

    self.myEcoModeFactor = 1.0
    self.params_count = 0
    self.cruiseMaxVals1 = float(int(Params().get("CruiseMaxVals1", encoding="utf8"))) / 100.
    self.cruiseMaxVals2 = float(int(Params().get("CruiseMaxVals2", encoding="utf8"))) / 100.
    self.cruiseMaxVals3 = float(int(Params().get("CruiseMaxVals3", encoding="utf8"))) / 100.
    self.cruiseMaxVals4 = float(int(Params().get("CruiseMaxVals4", encoding="utf8"))) / 100.
    self.cruiseMaxVals5 = float(int(Params().get("CruiseMaxVals5", encoding="utf8"))) / 100.
    self.cruiseMaxVals6 = float(int(Params().get("CruiseMaxVals6", encoding="utf8"))) / 100.
    self.autoTurnControl = int(Params().get("AutoTurnControl", encoding="utf8"))

    self.mpc.openpilotLongitudinalControl = CP.openpilotLongitudinalControl
    
    self.prev_lead_d = 0.0
    self.depart_cnt = 0  
    self.lead_dep_score = 0
    self.ld_dbg = 0
    self.drate_dbg = 0.0
    self.cap_dbg = 0.0
    # v1.5.5 traffic-follow diagnostics
    self.away_speed_dbg = 0.0
    self.depart_conf_dbg = 0.0
    self.base_cap_dbg = 0.0
    # v1.5.6 cruise-speed fail-safe diagnostics
    self.cruise_guard_cap_dbg = 0.0
    self.cruise_overspeed_dbg = 0.0
    self.cruise_guard_active_dbg = False

    # v1.5.7 cut-in-aware braking authority
    self.lead_brake_candidate_timer = 0.0
    self.brake_authority_dbg = 0.0
    self.required_decel_dbg = 0.0
    self.lead_ttc_dbg = 99.0
    self.lead_speed_dbg = 0.0
    self.lead_brake_confirmed_dbg = False

    # v1.5.8 clear-road recovery state/debug
    self.prev_lane_change_active = False
    self.prev_close_lead = False
    self.clear_lead_confirm_timer = 0.0
    self.clear_road_recovery_timer = 0.0
    self.clear_road_recovery_dbg = False
    self.driving_mode_dbg = 0
    self.mode_max_accel_dbg = 0.0

    self.gear_hold_cap_dbg = 0.0
    self.final_accel_max_dbg = 0.0

  def read_param(self):
    #try:
    #  self.personality = int(self.params.get('LongitudinalPersonality'))
    #except (ValueError, TypeError):
    #  self.personality = log.LongitudinalPersonality.standard

    self.myEcoModeFactor = float(int(Params().get("MyEcoModeFactor", encoding="utf8"))) / 100.
    self.cruiseMaxVals1 = float(int(Params().get("CruiseMaxVals1", encoding="utf8"))) / 100.
    self.cruiseMaxVals2 = float(int(Params().get("CruiseMaxVals2", encoding="utf8"))) / 100.
    self.cruiseMaxVals3 = float(int(Params().get("CruiseMaxVals3", encoding="utf8"))) / 100.
    self.cruiseMaxVals4 = float(int(Params().get("CruiseMaxVals4", encoding="utf8"))) / 100.
    self.cruiseMaxVals5 = float(int(Params().get("CruiseMaxVals5", encoding="utf8"))) / 100.
    self.cruiseMaxVals6 = float(int(Params().get("CruiseMaxVals6", encoding="utf8"))) / 100.
    self.autoTurnControl = int(Params().get("AutoTurnControl", encoding="utf8"))

    
  def get_max_accel(self, v_ego):
    cruiseMaxVals = [self.cruiseMaxVals1, self.cruiseMaxVals2, self.cruiseMaxVals3, self.cruiseMaxVals4, self.cruiseMaxVals5, self.cruiseMaxVals6]
    return interp(v_ego, A_CRUISE_MAX_BP, cruiseMaxVals)
  @staticmethod
  def parse_model(model_msg, model_error, v_ego, autoTurnControl):
    if (len(model_msg.position.x) == 33 and
       len(model_msg.velocity.x) == 33 and
       len(model_msg.acceleration.x) == 33):
      x = np.interp(T_IDXS_MPC, T_IDXS, model_msg.position.x) - model_error * T_IDXS_MPC
      v = np.interp(T_IDXS_MPC, T_IDXS, model_msg.velocity.x) - model_error
      a = np.interp(T_IDXS_MPC, T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
      y = np.interp(T_IDXS_MPC, T_IDXS, model_msg.position.y)
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
      y = np.zeros(len(T_IDXS_MPC))
      
    if False: #autoTurnControl == 2: # 속도를 줄이자~
      max_lat_accel = interp(v_ego, [5, 10, 20], [1.5, 2.0, 3.0])
      curvatures = np.interp(T_IDXS_MPC, T_IDXS, model_msg.orientationRate.z) / np.clip(v, 0.3, 100.0)
      max_v = np.sqrt(max_lat_accel / (np.abs(curvatures) + 1e-3)) - 2.0
      v = np.minimum(max_v, v)
    
    return x, v, a, j, y

  def update(self, sm):
    if self.param_read_counter % 50 == 0:
      self.read_param()
    self.param_read_counter += 1
    #self.mpc.mode = 'blended' if sm['controlsState'].experimentalMode else 'acc'
    self.mpc.experimentalMode = sm['controlsState'].experimentalMode

    v_ego = sm['carState'].vEgo
    v_cruise_kph = sm['controlsState'].vCruise
    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    # controlsState.vCruise and vEgoCluster are both cluster/display-speed based.
    # Keep this unscaled pair for the independent cruise overspeed fail-safe.
    v_cruise_cluster_kph = float(v_cruise_kph)
    v_ego_cluster_kph = float(sm['carState'].vEgoCluster * CV.MS_TO_KPH)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    # neokii
    vCluRatio = sm['carState'].vCluRatio
    if vCluRatio > 0.5:
      self.vCluRatio = vCluRatio
      v_cruise *= vCluRatio
      #v_cruise = int(v_cruise * CV.MS_TO_KPH + 0.25) * CV.KPH_TO_MS
    mySafeModeFactor = sm['controlsState'].mySafeModeFactor
    myDrivingMode = sm['controlsState'].myDrivingMode

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['controlsState'].enabled

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if self.mpc.mode == 'acc':
      # 1) 먼저 기본 myMaxAccel 계산
      if myDrivingMode in [1]:  # 연비
        myMaxAccel = clip(self.get_max_accel(v_ego) * self.myEcoModeFactor, 0, MAX_ACCEL)
      elif myDrivingMode in [2]:  # 안전
        myMaxAccel = clip(self.get_max_accel(v_ego) * self.myEcoModeFactor * mySafeModeFactor, 0, MAX_ACCEL)
      elif myDrivingMode in [3, 4]:  # 일반, 고속
        myMaxAccel = clip(self.get_max_accel(v_ego), 0, MAX_ACCEL)
      else:
        myMaxAccel = self.get_max_accel(v_ego)

      accel_limits = [A_CRUISE_MIN, myMaxAccel]
      self.driving_mode_dbg = int(myDrivingMode)
      self.mode_max_accel_dbg = float(myMaxAccel)
    
    else:
      # blended 등: 감속 여유는 열어두고(안전), 기본 상한은 MAX_ACCEL
      # accel_limits = [MIN_ACCEL, MAX_ACCEL]
      accel_limits = [A_CRUISE_MIN, MAX_ACCEL]
      self.driving_mode_dbg = int(myDrivingMode)
      self.mode_max_accel_dbg = float(MAX_ACCEL)

    # lead는 여기서 딱 1번만 읽기
    lead = sm['radarState'].leadOne

    # ------------------------------------------------------------------
    # v1.5.8 CLEAR-ROAD RECOVERY
    # Start a short recovery window after lane-change completion or release of
    # a previously close lead. Activate it only if the new path is actually
    # clear and the selected cruise speed remains >4 km/h above ego speed.
    # ------------------------------------------------------------------
    lane_change_active = (
      sm['lateralPlan'].laneChangeState != log.LateralPlan.LaneChangeState.off
    )
    if lane_change_active:
      self.clear_road_recovery_timer = 0.0
    elif self.prev_lane_change_active:
      self.clear_road_recovery_timer = 3.0
    else:
      self.clear_road_recovery_timer = max(
        self.clear_road_recovery_timer - DT_MDL, 0.0)

    clear_lead = (
      (not lead.status) or
      lead.dRel > 70.0 or
      (lead.dRel > 50.0 and lead.vRel > -0.30)
    )
    if clear_lead:
      self.clear_lead_confirm_timer = min(
        self.clear_lead_confirm_timer + DT_MDL, 1.0)
    else:
      self.clear_lead_confirm_timer = 0.0
    clear_lead_confirmed = self.clear_lead_confirm_timer >= 0.30

    close_lead_now = lead.status and lead.dRel < 45.0
    # Arm recovery on the actual close->released transition. The independent
    # clear_lead_confirmed gate below still requires 0.30 s of genuinely clear
    # road before the recovery becomes active.
    if self.prev_close_lead and not close_lead_now:
      self.clear_road_recovery_timer = max(
        self.clear_road_recovery_timer, 2.0)

    cruise_gap_kph = v_cruise_cluster_kph - v_ego_cluster_kph
    self.clear_road_recovery_dbg = bool(
      self.clear_road_recovery_timer > 0.0 and
      clear_lead_confirmed and
      cruise_gap_kph > 4.0 and
      v_ego > 8.0 and
      not reset_state and
      not sm['carState'].gasPressed and
      not sm['carState'].brakePressed
    )
    self.prev_lane_change_active = bool(lane_change_active)
    self.prev_close_lead = bool(close_lead_now)

    # v1.5.5 Traffic Comfort:
    # Detect a pulling-away lead early, but do not convert it into a binary
    # "wait -> full boost" event.  Keep a short confidence score and a
    # continuous away-speed estimate that the close-cap logic can use.
    lead_departing = False
    d_rate = 0.0
    d_rate_mps = 0.0
    away_speed = 0.0

    if lead.status:
      d = float(lead.dRel)
      vr = float(lead.vRel)  # lead - ego, positive = lead pulling away

      if self.prev_lead_d > 0.1:
        d_rate = d - self.prev_lead_d
        # d_rate is metres per model frame.  Convert to m/s, but down-weight
        # it because radar distance derivative is noisier than vRel.
        d_rate_mps = clip(d_rate / max(DT_MDL, 1e-3), -3.0, 3.0)
      self.prev_lead_d = d

      away_speed = clip(max(vr, max(d_rate_mps, 0.0) * 0.55), 0.0, 3.0)
      lead_departing = (
        v_ego < 8.5 and       # ~30 km/h: traffic/jam domain
        d < 35.0 and
        (vr > 0.08 or d_rate_mps > 0.15)
      )

      if lead_departing:
        self.lead_dep_score = min(self.lead_dep_score + 1, 5)
      else:
        self.lead_dep_score = max(self.lead_dep_score - 1, 0)

      # Keep only a short memory for planner acceleration-rise smoothing.
      # The old 2 s memory + 1.35 cap created a noticeable late surge.
      if self.lead_dep_score >= 2:
        self.depart_cnt = max(self.depart_cnt, int(0.8 / DT_MDL))

      self.ld_dbg = int(lead_departing)
      self.drate_dbg = float(d_rate)
      self.away_speed_dbg = float(away_speed)
      self.depart_conf_dbg = float(clip(self.lead_dep_score / 3.0, 0.0, 1.0))

    else:
      self.prev_lead_d = 0.0
      self.depart_cnt = 0
      self.lead_dep_score = 0
      self.ld_dbg = 0
      self.drate_dbg = 0.0
      self.cap_dbg = 0.0
      self.away_speed_dbg = 0.0
      self.depart_conf_dbg = 0.0
      self.base_cap_dbg = 0.0

    # 🔧 72~115km/h 저토크 재가속: 6단 유지 / 불필요한 5단 킥다운 억제
    # - lead 유무/거리와 무관하게 적용: 앞차 감속 후 20~30m 거리에서 재가속할 때도 빠지지 않음
    # - 72km/h부터 시작: 80km/h에 도달하기 전에 이미 5단으로 내려가는 현상 방지
    # - 위험 접근/close_cap/turn 제한은 아래에서 더 낮출 수 있으므로 감속 안전 로직은 그대로 우선
    self.gear_hold_cap_dbg = 0.0
    if 20.0 <= v_ego <= 32.0:
      gear_hold_cap = interp(
        v_ego,
        [20.0, 22.0, 25.0, 28.0, 30.0, 32.0],
        [0.50, 0.48, 0.45, 0.42, 0.40, 0.38],
      )
      accel_limits[1] = min(accel_limits[1], gear_hold_cap)
      self.gear_hold_cap_dbg = float(gear_hold_cap)
    
    # ------------------------------------------------------------------
    # v1.5.7 CUT-IN-AWARE DYNAMIC BRAKING AUTHORITY
    #
    # Replaces the blanket dRel<8m -> MIN_ACCEL(-4.0) rule.
    # Benign cut-ins keep normal braking authority. Risky moving cut-ins
    # are confirmed for ~0.20 s, while very slow/stationary obstacles and
    # extreme TTC bypass the debounce. This only opens the MPC lower limit;
    # it does NOT directly command that amount of braking.
    # ------------------------------------------------------------------
    self.brake_authority_dbg = abs(float(accel_limits[0]))
    self.required_decel_dbg = 0.0
    self.lead_ttc_dbg = 99.0
    self.lead_speed_dbg = 0.0
    self.lead_brake_confirmed_dbg = False

    if lead.status:
      risk_d = max(float(lead.dRel), 0.1)
      risk_vr = float(lead.vRel)
      risk_vlead = max(float(lead.vLead), 0.0)
      closing_speed = max(-risk_vr, 0.0)

      self.lead_speed_dbg = risk_vlead
      if closing_speed > 0.10:
        self.lead_ttc_dbg = min(risk_d / closing_speed, 99.0)

      # Physical collision buffer, intentionally separate from comfort gap.
      collision_buffer = 4.0
      usable_distance = max(risk_d - collision_buffer, 1.0)
      required_decel = ((closing_speed * closing_speed) /
                        (2.0 * usable_distance)) if closing_speed > 0.0 else 0.0
      self.required_decel_dbg = required_decel

      near_stationary_or_slow = (
        risk_vlead < 5.0 and       # <18 km/h lead
        v_ego > 5.0 and           # don't alter normal crawling traffic
        closing_speed > 3.0 and
        self.lead_ttc_dbg < 6.5
      )
      extreme_risk = (
        self.lead_ttc_dbg < 1.5 or
        required_decel > 2.5
      )
      moving_risk_candidate = (
        closing_speed > 3.0 and
        (self.lead_ttc_dbg < 5.0 or required_decel > 1.2)
      )

      if moving_risk_candidate:
        self.lead_brake_candidate_timer = min(
          self.lead_brake_candidate_timer + DT_MDL, 1.0)
      else:
        self.lead_brake_candidate_timer = max(
          self.lead_brake_candidate_timer - 2.0 * DT_MDL, 0.0)

      risk_confirmed = (
        near_stationary_or_slow or
        extreme_risk or
        self.lead_brake_candidate_timer >= 0.20
      )
      self.lead_brake_confirmed_dbg = bool(risk_confirmed)

      if risk_confirmed and (
          moving_risk_candidate or near_stationary_or_slow or extreme_risk):
        authority_mag = clip(
          required_decel * 1.20 + 0.10,
          abs(A_CRUISE_MIN),
          abs(MIN_ACCEL),
        )

        if near_stationary_or_slow:
          slow_floor = interp(
            self.lead_ttc_dbg,
            [1.0, 2.0, 3.5, 6.5],
            [4.0, 3.0, 2.1, 1.6],
          )
          authority_mag = max(authority_mag, slow_floor)

        accel_limits[0] = min(accel_limits[0], -authority_mag)
        self.brake_authority_dbg = authority_mag
    else:
      self.lead_brake_candidate_timer = 0.0

    # 2) v1.5.5 Traffic Comfort: continuous gap + relative-speed cap.
    #
    # Old behavior had a distance-only low cap, then a binary depart boost
    # to 1.35.  That produced: wait -> gap opens -> surge -> brake.
    # New behavior starts gently as soon as the lead pulls away, and increases
    # allowed acceleration continuously with both gap and away speed.
    if lead.status:
      d = float(lead.dRel)
      vr = float(lead.vRel)

      # Safe base envelope when relative speed is near zero.
      base_close_cap = interp(
        d,
        [4.5, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0],
        [0.06, 0.10, 0.18, 0.28, 0.40, 0.60, 0.80, accel_limits[1]],
      )
      close_cap = base_close_cap
      self.base_cap_dbg = float(base_close_cap)

      # Pull-away response.  Confidence rises over only a few model frames;
      # no fixed 1.35 step remains.
      if v_ego < 8.5 and d < 35.0 and self.away_speed_dbg > 0.0:
        depart_conf = self.depart_conf_dbg
        away_bonus = interp(
          self.away_speed_dbg,
          [0.0, 0.15, 0.35, 0.70, 1.20, 2.00, 3.00],
          [0.0, 0.03, 0.10, 0.22, 0.36, 0.52, 0.62],
        )
        gap_bonus = interp(
          d,
          [4.5, 7.0, 10.0, 14.0, 20.0, 30.0],
          [0.0, 0.0, 0.04, 0.08, 0.12, 0.18],
        )
        pullaway_cap = min(base_close_cap + depart_conf * (away_bonus + gap_bonus), 1.00)
        close_cap = max(close_cap, pullaway_cap)

      # Start tapering positive acceleration before closing speed becomes
      # large.  This reduces the late catch-up / strong-brake sawtooth without
      # touching emergency braking or negative-acceleration safety limits.
      if vr < -0.15 and d < 30.0:
        closing_cap = interp(
          vr,
          [-3.0, -1.5, -0.7, -0.3, 0.0],
          [0.08, 0.16, 0.28, 0.42, close_cap],
        )
        close_cap = min(close_cap, closing_cap)

      self.cap_dbg = float(close_cap)
      accel_limits[1] = min(accel_limits[1], close_cap)

    # ------------------------------------------------------------------
    # v1.5.6 CRUISE SPEED FAIL-SAFE (planner layer)
    #
    # Road-test failure: with CT=90 and no lead (L=0), AD/A0 stayed about
    # +0.4 even after cluster speed passed 90, eventually accelerating well
    # beyond the selected cruise speed.  The MPC cruise obstacle is a soft
    # cost and cannot be the only speed-limit safety mechanism.
    #
    # Once cluster speed is >0.5 km/h above the applied cruise target, cap
    # positive acceleration independently of lead/MPC state.  A larger
    # overspeed requests only a modest deceleration; stronger negative accel
    # from normal safety/lead logic remains untouched.
    # ------------------------------------------------------------------
    self.cruise_guard_cap_dbg = 0.0
    self.cruise_overspeed_dbg = 0.0
    self.cruise_guard_active_dbg = False
    cruise_target_valid = 1.0 <= v_cruise_cluster_kph <= V_CRUISE_MAX
    cluster_speed_valid = v_ego_cluster_kph > 0.5
    cruise_overspeed_kph = v_ego_cluster_kph - v_cruise_cluster_kph

    if (not reset_state) and cruise_target_valid and cluster_speed_valid and cruise_overspeed_kph > 0.5:
      cruise_guard_cap = interp(
        cruise_overspeed_kph,
        [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
        [0.05, 0.00, -0.05, -0.10, -0.20, -0.35, -0.60],
      )
      accel_limits[1] = min(accel_limits[1], cruise_guard_cap)
      self.cruise_guard_cap_dbg = float(cruise_guard_cap)
      self.cruise_overspeed_dbg = float(cruise_overspeed_kph)
      self.cruise_guard_active_dbg = True

    # 3) turns 제한은 마지막에 한 번
    accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)


    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = clip(sm['carState'].aEgo, accel_limits[0], accel_limits[1])
      self.mpc.prev_a = np.full(N+1, self.a_desired) ## mpc에서는 prev_a를 참고하여 constraint작동함.... pid off -> on시에는 현재 constraint가 작동하지 않아서 집어넣어봄...
      accel_limits_turns[0] = 0.0

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    # Compute model v_ego error
    self.v_model_error = get_speed_error(sm['modelV2'], v_ego)

    if force_slow_decel:
      v_cruise = 0.0
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.05)
    if self.cruise_guard_active_dbg:
      # Safety cap must never be reopened by the normal a_desired smoothing.
      accel_limits_turns[1] = min(accel_limits_turns[1], self.cruise_guard_cap_dbg)
    else:
      accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)

    # 실제 MPC에 전달되는 최종 acceleration upper limit 기록
    self.final_accel_max_dbg = float(accel_limits_turns[1])

    #self.mpc.set_weights(prev_accel_constraint)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    x, v, a, j, y = self.parse_model(sm['modelV2'], self.v_model_error, v_ego, self.autoTurnControl)

    self.mpc.update(
      sm['carState'], sm['radarState'], sm['modelV2'], sm['controlsState'],
      v_cruise, x, v, a, j, y, prev_accel_constraint, reset_state,
      self.clear_road_recovery_dbg,
    )

    self.v_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.a_solution)
    self.v_desired_trajectory = self.v_desired_trajectory_full[:CONTROL_N]
    self.a_desired_trajectory = self.a_desired_trajectory_full[:CONTROL_N]
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)

    if self.cruise_guard_active_dbg:
      # Hard safety envelope for the plan sent to LongControl.
      # v_cruise is already converted to the vehicle-speed domain by vCluRatio.
      self.v_desired_trajectory = np.minimum(self.v_desired_trajectory, v_cruise)
      self.a_desired_trajectory = np.minimum(self.a_desired_trajectory, self.cruise_guard_cap_dbg)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill and not reset_state
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0

    # v1.5.5: because the lead pull-away is now detected earlier, a huge
    # restart acceleration jump is unnecessary.  Keep only a short mild
    # response assist; LongControl still applies its own positive J+ shaping.
    restart_boost = (
      self.depart_cnt > 0 and
      v_ego < 8.5 and
      lead.status and
      self.away_speed_dbg > 0.15
    )
    j_pos_limit = interp(v_ego,
                         [0.0, 3.0, 8.0, 20.0, 25.0, 30.0],
                         [0.25, 0.35, 0.55, 0.75, 0.60, 0.50])

    if restart_boost:
      j_pos_limit *= 1.35

    a_inc_max = j_pos_limit * DT_MDL

    if restart_boost:
      a_inc_max = max(a_inc_max, 0.06)

    if self.a_desired > a_prev:
      self.a_desired = min(self.a_desired, a_prev + a_inc_max)

    if lead.status:
      # v1.5.7: keep normal low-speed decel smoothing even when a benign
      # cut-in is physically close. Bypass smoothing only after the new
      # dynamic braking-risk logic has actually confirmed danger.
      if v_ego < 10.0 and lead.dRel < 30.0 and not self.lead_brake_confirmed_dbg:
        j_neg_limit = interp(v_ego, [0.0, 3.0, 8.0, 10.0], [0.35, 0.50, 0.80, 1.20])
        a_dec_max = j_neg_limit * DT_MDL
        if self.a_desired < a_prev:
          self.a_desired = max(self.a_desired, a_prev - a_dec_max)

    if self.depart_cnt > 0:
      self.depart_cnt -= 1

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    #C2#longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    #C2#longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    #C2#longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.debugLongText1 = self.mpc.debugLongText1
    #self.mpc.debugLongText2 = "Vout={:3.2f},{:3.2f},{:3.2f},{:3.2f},{:3.2f}".format(longitudinalPlan.speeds[0]*3.6,longitudinalPlan.speeds[1]*3.6,longitudinalPlan.speeds[2]*3.6,longitudinalPlan.speeds[3]*3.6,longitudinalPlan.speeds[-1]*3.6)
    #self.mpc.debugLongText2 = "VisionTurn:State={},Speed={:.1f}".format(self.vision_turn_controller.state, self.vision_turn_controller.v_turn*3.6)
    #longitudinalPlan.debugLongText2 = self.mpc.debugLongText2
    lead = sm['radarState'].leadOne

    # DEBUG ONLY (v1.5.8). '|' is a UI line-break delimiter.
    if lead.status:
      longitudinalPlan.debugLongText2 = (
        f"PL V{sm['carState'].vEgo*3.6:.1f} DM{self.driving_mode_dbg} "
        f"MX{self.mode_max_accel_dbg:.2f} AM{self.final_accel_max_dbg:.2f} "
        f"AD{self.a_desired:.2f} A0{self.a_desired_trajectory[0]:.2f}"
        f"|L D{lead.dRel:.1f} V{lead.vRel:+.2f} CP{self.cap_dbg:.2f} "
        f"AS{self.away_speed_dbg:.2f} DC{self.depart_conf_dbg:.1f} "
        f"RS{self.mpc.restart_stop_distance:.1f}"
        f"|GH{self.gear_hold_cap_dbg:.2f} CG{self.cruise_guard_cap_dbg:.2f}/"
        f"{self.cruise_overspeed_dbg:.1f} CR{int(self.clear_road_recovery_dbg)} "
        f"BA{self.brake_authority_dbg:.2f} T{self.lead_ttc_dbg:.1f} "
        f"BC{int(self.lead_brake_confirmed_dbg)}"
      )
    else:
      longitudinalPlan.debugLongText2 = (
        f"PL V{sm['carState'].vEgo*3.6:.1f} DM{self.driving_mode_dbg} "
        f"MX{self.mode_max_accel_dbg:.2f} AM{self.final_accel_max_dbg:.2f} "
        f"AD{self.a_desired:.2f} A0{self.a_desired_trajectory[0]:.2f}"
        f"|GH{self.gear_hold_cap_dbg:.2f} CG{self.cruise_guard_cap_dbg:.2f}/"
        f"{self.cruise_overspeed_dbg:.1f} CR{int(self.clear_road_recovery_dbg)} "
        f"L0 RS{self.mpc.restart_stop_distance:.1f}"
      )

    longitudinalPlan.trafficState = self.mpc.trafficState
    longitudinalPlan.xState = self.mpc.xState
    if self.mpc.trafficError:
      longitudinalPlan.trafficState = self.mpc.trafficState + 1000
    longitudinalPlan.xStop = float(self.mpc.stopDist) #float(self.mpc.xStop)
    longitudinalPlan.tFollow = float(self.mpc.t_follow)
    longitudinalPlan.cruiseGap = float(self.mpc.applyCruiseGap)
    longitudinalPlan.xObstacle = float(self.mpc.x_obstacle_min[0])
    longitudinalPlan.mpcEvent = self.mpc.mpcEvent
    longitudinalPlan.mpcMode = 1 if self.mpc.mode == 'blended' else 0

    if self.CP.openpilotLongitudinalControl:
      longitudinalPlan.xCruiseTarget = float(self.mpc.v_cruise / self.vCluRatio)
    else:
      longitudinalPlan.xCruiseTarget = float(longitudinalPlan.speeds[-1] / self.vCluRatio)

    pm.send('longitudinalPlan', plan_send)
