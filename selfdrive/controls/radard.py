#!/usr/bin/env python3
import importlib
import math
from collections import defaultdict, deque

#추가
class LeadStabilizer:
  """
  Stabilize lead to reduce harsh braking on sudden cut-in / radar track ID switch.
  Works on lead dicts: expects keys like status, dRel, vRel, yRel (others preserved).
  """

  def __init__(self, dt, hold_s=0.7, alpha=0.12, danger_dist=12.0,
               ok_frames=6):
    self.dt = float(dt)
    self.hold_frames = max(1, int(hold_s / self.dt))
    self.alpha = float(alpha)
    self.danger_dist = float(danger_dist)
    self.ok_frames = int(ok_frames)

    self.stable = None
    self.stable_id = None
    self.hold = 0
    self.ok = 0

  @staticmethod
  def _lead_id(ld: dict):
    if not isinstance(ld, dict):
      return None
    # 포크별로 키 이름이 다를 수 있어 후보를 넓게 잡음
    for k in ("trackId", "radarId", "track_id", "id"):
      if k in ld:
        return ld[k]
    return None

  @staticmethod
  def _f(ld: dict, key: str, default=0.0):
    try:
      return float(ld.get(key, default))
    except Exception:
      return float(default)

  def _copy(self, ld: dict):
    # 원본 dict 유지하면서 필드만 바꿀거라 얕은복사면 충분
    return dict(ld) if isinstance(ld, dict) else None

  def update(self, raw: dict, v_ego: float) -> dict:
    """
    raw: lead dict from get_lead()
    v_ego: m/s
    returns: stabilized lead dict
    """
    if raw is None or not isinstance(raw, dict) or not raw.get("status", False):
      # lead가 사라졌을 때: 잠깐 hold 후 드랍
      if self.hold > 0 and self.stable is not None and self.stable.get("status", False):
        self.hold -= 1
        out = self._copy(self.stable)
        out["status"] = True
        return out

      self.stable = None
      self.stable_id = None
      self.hold = 0
      self.ok = 0
      return raw if raw is not None else {"status": False}

    # raw 값
    rid = self._lead_id(raw)
    d = self._f(raw, "dRel", 0.0)
    vr = self._f(raw, "vRel", 0.0)
    y = self._f(raw, "yRel", 0.0)

    danger = d < self.danger_dist

    # 첫 유효 lead면 초기화
    if self.stable is None:
      self.stable = self._copy(raw)
      self.stable_id = rid
      self.hold = 0
      self.ok = 0
      return raw

    sd = self._f(self.stable, "dRel", d)
    svr = self._f(self.stable, "vRel", vr)

    # -------------------------
    # 게이팅(튀는 값 판단)
    # -------------------------
    # “너무 가까워지는 점프”만 강하게 막는 게 급브레이크 완화에 효과적
    # 프레임당 허용 거리 감소량(대략):
    #  - 기본 6m + 속도/상대속도 고려해서 조금 가변
    closing = max(0.0, -(vr))  # vr이 음수면 접근 중
    jump_allow = max(6.0, (v_ego * 0.15) + (closing * self.dt * 6.0))  # meters

    too_close_jump = d < (sd - jump_allow)

    # ID 스위치 힌트(있을 때만)
    id_switch = (rid is not None and self.stable_id is not None and rid != self.stable_id)

    # vRel 급변도 감시(단, 단위/노이즈가 있을 수 있어 속도에 비례하게)
    relspd_allow = max(4.0, 0.25 * v_ego)   # m/s-ish
    relspd_jump = abs(vr - svr) > relspd_allow

    spike = (too_close_jump or relspd_jump) and not danger

    if spike:
      # hold 시작/갱신
      self.hold = self.hold_frames
      self.ok = 0
      out = self._copy(self.stable)
      out["status"] = True
      return out

    # 스파이크가 아니면 정상 카운트
    self.ok += 1
    if self.hold > 0 and not danger:
      # hold 중이면 stable 유지(단 danger면 즉시 반영)
      self.hold -= 1
      out = self._copy(self.stable)
      out["status"] = True
      return out

    # -------------------------
    # LPF 업데이트
    # -------------------------
    # ok_frames만큼 연속으로 정상일 때 부드럽게 갱신
    if self.ok >= self.ok_frames or danger or id_switch:
      a = self.alpha if not danger else max(self.alpha, 0.35)  # 위험이면 더 빨리 반영
      new = self._copy(raw)
      new["dRel"] = a * d + (1.0 - a) * sd
      new["vRel"] = a * vr + (1.0 - a) * svr
      new["yRel"] = a * y + (1.0 - a) * self._f(self.stable, "yRel", y)
      new["status"] = True

      self.stable = new
      self.stable_id = rid
      return new

    # 아직 충분히 안정 프레임이 쌓이기 전이면 raw 그대로
    return raw

import cereal.messaging as messaging
from cereal import car
from common.numpy_fast import interp
from common.params import Params
from common.realtime import Ratekeeper, Priority, config_realtime_process
from selfdrive.controls.lib.radar_helpers import Cluster, Track, RADAR_TO_CAMERA
from selfdrive.swaglog import cloudlog
from third_party.cluster.fastcluster_py import cluster_points_centroid
from selfdrive.hardware import TICI
from common.params import Params

from selfdrive.controls.lib.lane_planner import TRAJECTORY_SIZE
import numpy as np

LEAD_PATH_DREL_MIN = 60 # [m] only care about far away leads
MIN_LANE_PROB = 0.6  # Minimum lanes probability to allow use.

#LEAD_PLUS_ONE_MIN_REL_DIST_V = [3.0, 6.0] # [m] min distance between lead+1 and lead at low and high distance
#LEAD_PLUS_ONE_MIN_REL_DIST_BP = [0., 100.] # [m] min distance between lead+1 and lead at low and high distance
#LEAD_PLUS_ONE_MAX_YREL_TO_LEAD = 3.0 # [m]

class KalmanParams():
  def __init__(self, dt):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [dt * 0.01 for dt in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[interp(dt, dts, K0)], [interp(dt, dts, K1)]]


def laplacian_pdf(x, mu, b):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_cluster(v_ego, lead, clusters):
  # match vision point to best statistical cluster match
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, -lead.y[0], lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This is isn't exactly right, but good heuristic
    return prob_d * prob_y * prob_v

  cluster = max(clusters, key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  #dist_sane = abs(cluster.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  dist_sane = abs(cluster.dRel - offset_vision_dist) < max([(offset_vision_dist)*.35, 5.0])
  vel_sane = (abs(cluster.vRel + v_ego - lead.v[0]) < 10) or (v_ego + cluster.vRel > 3)
  if dist_sane and vel_sane:
    return cluster
  else:
    return None

def get_path_adjacent_leads(v_ego, md, lane_width, clusters):
  if len(clusters) == 0:
    return [[],[],[]]
  
  if md is not None and lane_width > 0. and len(md.laneLines) == 4 and len(md.laneLines[1].x) == TRAJECTORY_SIZE:
    # get centerline approximation using one or both lanelines
    ll_x = md.laneLines[1].x  # left and right ll x is the same
    lll_y = np.array(md.laneLines[1].y)
    rll_y = np.array(md.laneLines[2].y)
    l_prob = md.laneLineProbs[1]
    r_prob = md.laneLineProbs[2]

    # Find path from lanes as the average center lane only if min probability on both lanes is above threshold.
    if l_prob > MIN_LANE_PROB and r_prob > MIN_LANE_PROB:
      c_y = (lll_y + rll_y) / 2.
    elif l_prob > MIN_LANE_PROB:
      c_y = lll_y + (lane_width / 2)
    elif r_prob > MIN_LANE_PROB:
      c_y = rll_y - (lane_width / 2)
    else:
      c_y = None
  else:
    c_y = None
    ll_x = None   #add
    
  pos_x = md.position.x if md is not None else []
  pos_y = md.position.y if md is not None else []
  if md is not None and len(pos_x) > 0 and len(pos_y) > 0 and (len(pos_x) == TRAJECTORY_SIZE or pos_x[-1] > LEAD_PATH_DREL_MIN):   
  #if md is not None and (len(md.position.x) == TRAJECTORY_SIZE or md.position.x[-1] > LEAD_PATH_DREL_MIN):
    md_y = pos_y
    md_x = pos_x
  else:
    md_y = None
    md_x = None   #add
    
  leads_left = {}
  leads_center = {}
  leads_right = {}
  half_lane_width = lane_width / 2
  for c in clusters:
    use_model_path = (md_x is not None and md_y is not None and len(md_x) > 0 and len(md_y) > 0)
    use_lane_path  = (c_y is not None and ll_x is not None and len(ll_x) > 0 and len(c_y) > 0)
    
    if use_model_path and (c.dRel <= md_x[-1] or (use_lane_path and md_x[-1] < ll_x[-1])):
      dPath = -c.yRel - interp(c.dRel, md_x, md_y)
      checkSource = 'modelPath'
      
    elif use_lane_path:
      dPath = -c.yRel - interp(c.dRel, ll_x, c_y.tolist())
      checkSource = 'modelLaneLines'
      
    else:
      dPath = -c.yRel
      checkSource = 'lowSpeedOverride'
      
    source = 'vision' if c.dRel > 145. else 'radar'
    
    #ld = c.get_RadarState(source=source, checkSource=checkSource)
    ld = c.get_RadarState()
    ld["dPath"] = dPath
    ld["vLat"] = math.sqrt((10*dPath)**2 + c.dRel**2)
    if abs(dPath) < half_lane_width and ld["vLeadK"] > -1.: # want to still get stopped leads, so put in wiggle-room for radar noise
      leads_center[abs(dPath)] = ld
    elif dPath < 0.:
      leads_left[abs(dPath)] = ld
    else:
      leads_right[abs(dPath)] = ld
  
  ll,lr = [[l[k] for k in sorted(list(l.keys()))] for l in [leads_left,leads_right]]
  lc = sorted(leads_center.values(), key=lambda c:c["dRel"])
  return [ll,lc,lr]

def get_lead(v_ego, ready, clusters, lead_msg, model_v_ego, low_speed_override=True, mixRadarInfo=0):
  # Determine leads, this is where the essential logic happens
  if len(clusters) > 0 and ready and lead_msg.prob > .5:
    cluster = match_vision_to_cluster(v_ego, lead_msg, clusters)
  else:
    cluster = None

  lead_dict = {'status': False}
  if cluster is not None:
    lead_dict = cluster.get_RadarState2(lead_msg.prob, lead_msg, mixRadarInfo)
  elif (cluster is None) and ready and (lead_msg.prob > .5):
    lead_dict = Cluster().get_RadarState_from_vision(lead_msg, v_ego, model_v_ego)

  if low_speed_override:
    low_speed_clusters = [c for c in clusters if c.potential_low_speed_lead(v_ego)]
    if len(low_speed_clusters) > 0:
      closest_cluster = min(low_speed_clusters, key=lambda c: c.dRel)

      # Only choose new cluster if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_cluster.dRel < lead_dict['dRel']):
        lead_dict = closest_cluster.get_RadarState2(lead_msg.prob, lead_msg, mixRadarInfo)

  return lead_dict


class RadarD():
  def __init__(self, radar_ts, delay=0):
    self.current_time = 0

    self.tracks = defaultdict(dict)
    self.kalman_params = KalmanParams(radar_ts)
    
    # (add) lead stabilization to reduce harsh braking on cut-in / id switch
    self.lead_stab_one = LeadStabilizer(radar_ts, hold_s=0.7, alpha=0.12, danger_dist=12.0, ok_frames=6)
    self.lead_stab_two = LeadStabilizer(radar_ts, hold_s=0.5, alpha=0.12, danger_dist=12.0, ok_frames=6)

    
    # v_ego
    self.v_ego = 0.
    self.v_ego_hist = deque([0], maxlen=delay+1)

    self.ready = False
    self.showRadarInfo = False
    self.mixRadarInfo = 0

  def update(self, sm, rr):
    self.showRadarInfo = int(Params().get("ShowRadarInfo"))
    self.mixRadarInfo = int(Params().get("MixRadarInfo"))

    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.updated['carState']:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
    if sm.updated['modelV2']:
      self.ready = True

    ar_pts = {}
    for pt in rr.points:
      ar_pts[pt.trackId] = [pt.dRel, pt.yRel, pt.vRel, pt.measured]

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(v_lead, self.kalman_params, track_id=ids)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    idens = list(sorted(self.tracks.keys()))
    track_pts = [self.tracks[iden].get_key_for_cluster() for iden in idens]

    # If we have multiple points, cluster them
    if len(track_pts) > 1:
      cluster_idxs = cluster_points_centroid(track_pts, 2.5)
      clusters = [None] * (max(cluster_idxs) + 1)

      for idx in range(len(track_pts)):
        cluster_i = cluster_idxs[idx]
        if clusters[cluster_i] is None:
          clusters[cluster_i] = Cluster()
        clusters[cluster_i].add(self.tracks[idens[idx]])
    elif len(track_pts) == 1:
      # FIXME: cluster_point_centroid hangs forever if len(track_pts) == 1
      cluster_idxs = [0]
      clusters = [Cluster()]
      clusters[0].add(self.tracks[idens[0]])
    else:
      clusters = []

    # if a new point, reset accel to the rest of the cluster
    for idx in range(len(track_pts)):
      if self.tracks[idens[idx]].cnt <= 1:
        aLeadK = clusters[cluster_idxs[idx]].aLeadK
        aLeadTau = clusters[cluster_idxs[idx]].aLeadTau
        self.tracks[idens[idx]].reset_a_lead(aLeadK, aLeadTau)

    # *** publish radarState ***
    dat = messaging.new_message('radarState')
    dat.valid = sm.all_checks() and len(rr.errors) == 0
    radarState = dat.radarState
    radarState.mdMonoTime = sm.logMonoTime['modelV2']
    radarState.radarErrors = list(rr.errors)
    radarState.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].temporalPose.trans):
      model_v_ego = sm['modelV2'].temporalPose.trans[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      lead1 = get_lead(self.v_ego, self.ready, clusters, leads_v3[0], model_v_ego, low_speed_override=True, mixRadarInfo=self.mixRadarInfo)
      lead2 = get_lead(self.v_ego, self.ready, clusters, leads_v3[1], model_v_ego, low_speed_override=False, mixRadarInfo=self.mixRadarInfo)
      
      # (add) stabilize leads (cut-in / id switch harsh brake mitigation)
      #lead1 = self.lead_stab_one.update(lead1, self.v_ego)
      #lead2 = self.lead_stab_two.update(lead2, self.v_ego)

      # schema에 trackId 없을 수 있으니 publish 전 제거 (안전)
      lead1.pop("trackId", None)
      lead2.pop("trackId", None)
      
      radarState.leadOne = lead1
      radarState.leadTwo = lead2
      #추가 끝
      
      if self.ready and self.showRadarInfo: #self.extended_radar_enabled and self.ready:
        ll,lc,lr = get_path_adjacent_leads(self.v_ego, sm['modelV2'], sm['lateralPlan'].laneWidth, clusters)
        #try:
        #  if abs(sm['carState'].steeringAngleDeg) < 15 and radarState.leadOne.status and radarState.leadOne.modelProb > 0.5:
        #    check_dist = interp(radarState.leadOne.dRel, LEAD_PLUS_ONE_MIN_REL_DIST_BP, LEAD_PLUS_ONE_MIN_REL_DIST_V)
        #    lc = [l for l in lc if l["dRel"] > radarState.leadOne.dRel + check_dist and abs(l["yRel"] - radarState.leadOne.yRel) <= LEAD_PLUS_ONE_MAX_YREL_TO_LEAD]
        #    if len(lc) > 0: # get the lead+1 car
        #      radarState.leadOnePlus = self.lead_one_plus_lr.update(lc[0], use_v_lat=self.extended_radar_enabled)
        #except AttributeError:
        #  lc = []
        #  self.lead_one_plus_lr.reset()
        for group in (ll, lc, lr):  #add
          for ld in group:   
            if isinstance(ld, dict):
              ld.pop("trackId", None)
        
        radarState.leadsLeft = list(ll)
        radarState.leadsCenter = list(lc)
        radarState.leadsRight = list(lr)

    return dat


# fuses camera and radar data for best lead detection
def radard_thread(sm=None, pm=None, can_sock=None):
  config_realtime_process(5 if TICI else 2, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("radard got CarParams")

  # import the radar from the fingerprint
  cloudlog.info("radard is importing %s", CP.carName)
  RadarInterface = importlib.import_module(f'selfdrive.car.{CP.carName}.radar_interface').RadarInterface

  # *** setup messaging
  if can_sock is None:
    can_sock = messaging.sub_sock('can')
  if sm is None:
    sm = messaging.SubMaster(['modelV2', 'carState', 'lateralPlan'], ignore_avg_freq=['modelV2', 'carState', 'lateralPlan'])  # Can't check average frequency, since radar determines timing
  if pm is None:
    pm = messaging.PubMaster(['radarState', 'liveTracks'])

  RI = RadarInterface(CP)

  rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
  RD = RadarD(CP.radarTimeStep, RI.delay)

  while 1:
    can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    rr = RI.update(can_strings)

    if rr is None:
      continue

    sm.update(0)

    dat = RD.update(sm, rr)
    dat.radarState.cumLagMs = -rk.remaining*1000.

    pm.send('radarState', dat)

    # *** publish tracks for UI debugging (keep last) ***
    tracks = RD.tracks
    dat = messaging.new_message('liveTracks', len(tracks))

    for cnt, ids in enumerate(sorted(tracks.keys())):
      dat.liveTracks[cnt] = {
        "trackId": ids,
        "dRel": float(tracks[ids].dRel),
        "yRel": float(tracks[ids].yRel),
        "vRel": float(tracks[ids].vRel),
      }
    pm.send('liveTracks', dat)

    rk.monitor_time()
    #rk.keep_time()


def main(sm=None, pm=None, can_sock=None):
  radard_thread(sm, pm, can_sock)


if __name__ == "__main__":
  main()
