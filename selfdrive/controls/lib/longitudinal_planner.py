#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import clip, interp

import cereal.messaging as messaging
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

A_CRUISE_MAX_VALS = [1.2, 1.1, 1.0, 0.85, 0.70]
A_CRUISE_MAX_BP = [0., 20*CV.KPH_TO_MS, 40*CV.KPH_TO_MS, 60*CV.KPH_TO_MS, 80*CV.KPH_TO_MS]

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
    
    else:
      # blended 등: 감속 여유는 열어두고(안전), 기본 상한은 MAX_ACCEL
      # accel_limits = [MIN_ACCEL, MAX_ACCEL]
      accel_limits = [A_CRUISE_MIN, MAX_ACCEL]

    # lead는 여기서 딱 1번만 읽기
    lead = sm['radarState'].leadOne

    lead_departing = False
    d_rate = 0.0
    
    if lead.status:
      d = float(lead.dRel)
      
      # d_rate 첫 프레임 튐 방지 (prev_lead_d가 초기값이면 0으로)
      if self.prev_lead_d > 0.1:
        d_rate = d - self.prev_lead_d
      else:
        d_rate = 0.0
      self.prev_lead_d = d

      lead_departing = (v_ego < 3.0) and (d < 35.0) and ((lead.vRel > 0.05) or (d_rate > 0.03))
      self.ld_dbg = int(lead_departing)
      self.drate_dbg = float(d_rate)
      
      # ✅ score 누적/감쇠 (핵심)
      if lead_departing:
        self.lead_dep_score = min(self.lead_dep_score + 1, 5)
      else:
        self.lead_dep_score = max(self.lead_dep_score - 1, 0)
      
      if self.lead_dep_score >= 2:
        self.depart_cnt = max(self.depart_cnt, int(2.0 / DT_MDL)) # 1.8~2.5s 취향

    else:
      self.prev_lead_d = 0.0
      self.depart_cnt = 0
      self.lead_dep_score = 0
      self.ld_dbg = 0
      self.drate_dbg = 0.0
      self.cap_dbg = 0.0

    # 🔧 80~100km/h 구간에서 과한 가속 억제 (킥다운 방지)
    if 22.0 <= v_ego <= 30.0:
      if not lead.status:
        accel_limits[1] = min(accel_limits[1], 0.55)
      elif lead.dRel > 35.0:
        accel_limits[1] = min(accel_limits[1], 0.60)
    
    # 1-추가) 위험 접근이면 감속 하한만 MIN_ACCEL로 "오픈" (모드 무관)
    if lead.status and (lead.dRel < 8.0 or (v_ego < 10.0 and lead.vRel < -3.0)):
      accel_limits[0] = MIN_ACCEL
    
    # 2) (add) lead가 가까울 때만 accel_limits[1] 추가로 낮춤
    if lead.status:
      d = float(lead.dRel)
      vr = float(lead.vRel)  # lead - ego (negative => closing)

      # 가까울수록 가속을 거의 막고, 25m까지 완만히 풀림
      close_cap = interp(d, [6.0, 12.0, 20.0, 30.0], [0.10, 0.30, 0.75, accel_limits[1]])
      #12m: 0.30 → 너무 답답하지 않게 “살짝” 가속 허용
      #20m: 0.75 → gap이 어느 정도면 자연스럽게 따라가기 시작
      #6m는 그대로 0.10이라 가까울 때 급가속은 계속 막힘
      #더 부드럽게(보수적으로) 가고 싶으면 12m를 0.25로, 더 반응성을 원하면 0.35~0.40까지.

      # 정체 재출발 보정: 저속 + gap 작고 + 앞차가 멀어지는 중이면 cap을 조금 빨리 풀어줌
      if v_ego < 10.0 and d < 25.0 and vr > 0.3:
        # 최소 허용 가속(답답함 방지)
        close_cap = max(close_cap, 0.50) #default 0.60
        # 급출발 방지용 상한(차량 취향에 맞게 1.0~1.4 범위 튜닝)
        close_cap = min(close_cap, 1.05) #default 1.20
      
      # 접근(closing)이면 더 보수적으로
      if vr < -1.0:
        close_cap = min(close_cap, interp(v_ego, [0.0, 6.0, 15.0], [0.15, 0.25, 0.40]))
        
      # depart 부스트 구간에는 close_cap을 조금 완화 (초반 반응성)
      if self.depart_cnt > 0 and v_ego < 12.0 and d < 35.0:
        close_cap = max(close_cap, 1.35)   # 1.2~1.5 취향

      self.cap_dbg = float(close_cap)
      accel_limits[1] = min(accel_limits[1], close_cap)

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
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)

    #self.mpc.set_weights(prev_accel_constraint)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    x, v, a, j, y = self.parse_model(sm['modelV2'], self.v_model_error, v_ego, self.autoTurnControl)

    self.mpc.update(sm['carState'], sm['radarState'], sm['modelV2'], sm['controlsState'], v_cruise, x, v, a, j, y, prev_accel_constraint, reset_state)

    self.v_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory_full = np.interp(T_IDXS, T_IDXS_MPC, self.mpc.a_solution)
    self.v_desired_trajectory = self.v_desired_trajectory_full[:CONTROL_N]
    self.a_desired_trajectory = self.a_desired_trajectory_full[:CONTROL_N]
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill and not reset_state
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0

    j_pos_limit = interp(v_ego,
                         [0.0, 3.0, 8.0, 20.0, 25.0, 30.0],
                         [0.25, 0.35, 0.55, 0.75, 0.60, 0.50])

    if restart_boost:
      j_pos_limit *= 1.9

    a_inc_max = j_pos_limit * DT_MDL

    if restart_boost:
      a_inc_max = max(a_inc_max, 0.22)

    if self.a_desired > a_prev:
      self.a_desired = min(self.a_desired, a_prev + a_inc_max)

    if lead.status:
      if v_ego < 10.0 and lead.dRel < 30.0 and not (lead.dRel < 8.0 or lead.vRel < -3.0):
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

    longitudinalPlan.debugLongText2 = (
      f"{self.mpc.debugLongText2}"
      f" | dep={self.depart_cnt} sc={self.lead_dep_score} ld={self.ld_dbg}"
      f" d={lead.dRel:.1f} vr={lead.vRel:.2f} dr={self.drate_dbg:+.2f}"
      f" cap={self.cap_dbg:.2f} v={sm['carState'].vEgo*3.6:.1f}"
    ) if lead.status else (
      f"{self.mpc.debugLongText2} | dep={self.depart_cnt} sc={self.lead_dep_score} lead=0"
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
