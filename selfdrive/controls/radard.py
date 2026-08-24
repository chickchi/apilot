#!/usr/bin/env python3
import importlib
import math
from collections import defaultdict, deque


# =============================================================================
# v1.5.9 FAR LEAD SWITCH GUARD
#
# Purpose:
# - Do NOT low-pass/filter a normal continuously tracked lead.
# - Do NOT delay a real close cut-in or genuinely dangerous closing vehicle.
# - Only debounce a physically discontinuous FAR lead switch for a very short
#   time, primarily to suppress curve / adjacent-lane target swaps.
#
# This replaces the old LeadStabilizer behavior that continuously smoothed
# dRel/vRel and could hold a stale lead for 0.5~0.7 s.
# =============================================================================
class FarLeadSwitchGuard:
  def __init__(self, dt, confirm_s=0.20, max_hold_s=0.35):
    self.dt = float(dt)
    self.confirm_s = float(confirm_s)
    self.max_hold_s = float(max_hold_s)

    self.stable = None

    self.pending_id = None
    self.pending_source = None
    self.pending_time = 0.0
    self.hold_time = 0.0

  @staticmethod
  def _copy(ld):
    return dict(ld) if isinstance(ld, dict) else None

  @staticmethod
  def _f(ld, key, default=0.0):
    try:
      return float(ld.get(key, default))
    except Exception:
      return float(default)

  @staticmethod
  def _lead_id(ld):
    if not isinstance(ld, dict):
      return None

    for key in ("trackId", "radarId", "track_id", "id"):
      if key in ld:
        return ld[key]

    return None

  @staticmethod
  def _source(ld):
    if not isinstance(ld, dict):
      return None

    # radar_helpers.py publishes radar=True for radar-backed leads
    # and radar=False for pure vision fallback.
    if "radar" in ld:
      return bool(ld.get("radar", False))

    return None

  def _reset_pending(self):
    self.pending_id = None
    self.pending_source = None
    self.pending_time = 0.0
    self.hold_time = 0.0

  def _accept(self, raw):
    out = self._copy(raw)

    self.stable = self._copy(raw)
    self._reset_pending()

    return out

  def _predict_stable(self):
    if self.stable is None:
      return {"status": False}

    out = self._copy(self.stable)

    d = self._f(out, "dRel", 0.0)
    vr = self._f(out, "vRel", 0.0)

    # Do not freeze the old lead at a fixed distance while confirming a switch.
    # Propagate it with its already-observed relative velocity.
    out["dRel"] = max(0.0, d + vr * self.dt)
    out["status"] = True

    # Keep a private copy. The caller later removes trackId before publishing.
    self.stable = self._copy(out)

    return out

  def update(self, raw, v_ego):
    if raw is None or not isinstance(raw, dict) or not raw.get("status", False):
      # v1.5.9 intentionally does NOT hold a disappeared lead.
      # clear-road recovery elsewhere already has its own confirmation timer.
      self.stable = None
      self._reset_pending()

      return raw if raw is not None else {"status": False}

    d = self._f(raw, "dRel", 0.0)
    vr = self._f(raw, "vRel", 0.0)

    rid = self._lead_id(raw)
    source = self._source(raw)

    # First valid lead: accept immediately.
    if self.stable is None or not self.stable.get("status", False):
      return self._accept(raw)

    sd = self._f(self.stable, "dRel", d)
    svr = self._f(self.stable, "vRel", vr)

    sid = self._lead_id(self.stable)
    stable_source = self._source(self.stable)

    closing = max(-vr, 0.0)
    ttc = d / closing if closing > 0.10 else 99.0

    # ---------------------------------------------------------------------
    # SAFETY BYPASS
    #
    # A real nearby cut-in / dangerous closing vehicle must NEVER wait for
    # far-lead switch confirmation.
    # ---------------------------------------------------------------------
    immediate_risk = (
      d <= 35.0 or
      ttc <= 3.5 or
      (d <= 45.0 and closing >= 5.0) or
      bool(raw.get("fcw", False))
    )

    if immediate_risk:
      return self._accept(raw)

    predicted_d = max(0.0, sd + svr * self.dt)

    distance_allow = max(
      8.0,
      0.10 * max(sd, 1.0),
    )

    relspd_allow = max(
      5.0,
      0.18 * max(float(v_ego), 0.0),
    )

    dist_jump = abs(d - predicted_d) > distance_allow
    relspd_jump = abs(vr - svr) > relspd_allow

    id_switch = (
      rid is not None and
      sid is not None and
      rid != sid
    )

    source_switch = (
      source is not None and
      stable_source is not None and
      source != stable_source
    )

    # Only debounce FAR-target discontinuities.
    #
    # Old lead must already be far, and the new candidate must not be a close
    # vehicle. This is important: a 20~35 m real cut-in bypasses this logic.
    far_context = (
      sd >= 45.0 and
      d >= 40.0
    )

    suspicious_switch = (
      far_context and (
        dist_jump or
        ((id_switch or source_switch) and relspd_jump) or
        (id_switch and abs(d - sd) > 6.0)
      )
    )

    if not suspicious_switch:
      return self._accept(raw)

    same_pending = (
      rid == self.pending_id and
      source == self.pending_source
    )

    if same_pending:
      self.pending_time += self.dt
    else:
      self.pending_id = rid
      self.pending_source = source
      self.pending_time = self.dt
      self.hold_time = 0.0

    self.hold_time += self.dt

    # A new far target that persists for ~0.2 s is real enough to accept.
    # Absolute hold is capped at ~0.35 s even if identity keeps changing.
    if (
      self.pending_time >= self.confirm_s or
      self.hold_time >= self.max_hold_s
    ):
      return self._accept(raw)

    return self._predict_stable()


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


LEAD_PATH_DREL_MIN = 60  # [m] only care about far away leads
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

    K0 = [
      0.12287673, 0.14556536, 0.16522756, 0.18281627,
      0.1988689, 0.21372394, 0.22761098, 0.24069424,
      0.253096, 0.26491023, 0.27621103, 0.28705801,
      0.29750003, 0.30757767, 0.31732515, 0.32677158,
      0.33594201, 0.34485814, 0.35353899, 0.36200124,
    ]

    K1 = [
      0.29666309, 0.29330885, 0.29042818, 0.28787125,
      0.28555364, 0.28342219, 0.28144091, 0.27958406,
      0.27783249, 0.27617149, 0.27458948, 0.27307714,
      0.27162685, 0.27023228, 0.26888809, 0.26758976,
      0.26633338, 0.26511557, 0.26393339, 0.26278425,
    ]

    self.K = [
      [interp(dt, dts, K0)],
      [interp(dt, dts, K1)],
    ]


def laplacian_pdf(x, mu, b):
  b = max(b, 1e-4)
  return math.exp(-abs(x - mu) / b)


def match_vision_to_cluster(v_ego, lead, clusters):
  # match vision point to best statistical cluster match
  offset_vision_dist = lead.x[0] - RADAR_TO_CAMERA

  def prob(c):
    prob_d = laplacian_pdf(
      c.dRel,
      offset_vision_dist,
      lead.xStd[0],
    )

    prob_y = laplacian_pdf(
      c.yRel,
      -lead.y[0],
      lead.yStd[0],
    )

    prob_v = laplacian_pdf(
      c.vRel + v_ego,
      lead.v[0],
      lead.vStd[0],
    )

    return prob_d * prob_y * prob_v

  cluster = max(clusters, key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives

  #dist_sane = abs(cluster.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, 5.0])
  dist_sane = (
    abs(cluster.dRel - offset_vision_dist) <
    max([(offset_vision_dist) * .35, 5.0])
  )

  vel_sane = (
    abs(cluster.vRel + v_ego - lead.v[0]) < 10 or
    (v_ego + cluster.vRel > 3)
  )

  if dist_sane and vel_sane:
    return cluster

  return None


def get_path_adjacent_leads(v_ego, md, lane_width, clusters):
  if len(clusters) == 0:
    return [[], [], []]

  if (
    md is not None and
    lane_width > 0. and
    len(md.laneLines) == 4 and
    len(md.laneLines[1].x) == TRAJECTORY_SIZE
  ):
    # get centerline approximation using one or both lanelines
    ll_x = md.laneLines[1].x

    lll_y = np.array(md.laneLines[1].y)
    rll_y = np.array(md.laneLines[2].y)

    l_prob = md.laneLineProbs[1]
    r_prob = md.laneLineProbs[2]

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
    ll_x = None

  pos_x = md.position.x if md is not None else []
  pos_y = md.position.y if md is not None else []

  if (
    md is not None and
    len(pos_x) > 0 and
    len(pos_y) > 0 and
    (
      len(pos_x) == TRAJECTORY_SIZE or
      pos_x[-1] > LEAD_PATH_DREL_MIN
    )
  ):
    md_y = pos_y
    md_x = pos_x

  else:
    md_y = None
    md_x = None

  leads_left = {}
  leads_center = {}
  leads_right = {}

  half_lane_width = lane_width / 2

  for c in clusters:
    use_model_path = (
      md_x is not None and
      md_y is not None and
      len(md_x) > 0 and
      len(md_y) > 0
    )

    use_lane_path = (
      c_y is not None and
      ll_x is not None and
      len(ll_x) > 0 and
      len(c_y) > 0
    )

    if (
      use_model_path and (
        c.dRel <= md_x[-1] or
        (use_lane_path and md_x[-1] < ll_x[-1])
      )
    ):
      dPath = -c.yRel - interp(
        c.dRel,
        md_x,
        md_y,
      )

      checkSource = 'modelPath'

    elif use_lane_path:
      dPath = -c.yRel - interp(
        c.dRel,
        ll_x,
        c_y.tolist(),
      )

      checkSource = 'modelLaneLines'

    else:
      dPath = -c.yRel
      checkSource = 'lowSpeedOverride'

    source = 'vision' if c.dRel > 145. else 'radar'

    #ld = c.get_RadarState(source=source, checkSource=checkSource)
    ld = c.get_RadarState()

    ld["dPath"] = dPath
    ld["vLat"] = math.sqrt((10 * dPath) ** 2 + c.dRel ** 2)

    if (
      abs(dPath) < half_lane_width and
      ld["vLeadK"] > -1.
    ):
      leads_center[abs(dPath)] = ld

    elif dPath < 0.:
      leads_left[abs(dPath)] = ld

    else:
      leads_right[abs(dPath)] = ld

  ll, lr = [
    [l[k] for k in sorted(list(l.keys()))]
    for l in [leads_left, leads_right]
  ]

  lc = sorted(
    leads_center.values(),
    key=lambda c: c["dRel"],
  )

  return [ll, lc, lr]


def get_lead(
  v_ego,
  ready,
  clusters,
  lead_msg,
  model_v_ego,
  low_speed_override=True,
  mixRadarInfo=0,
):
  # Determine leads, this is where the essential logic happens
  if (
    len(clusters) > 0 and
    ready and
    lead_msg.prob > .5
  ):
    cluster = match_vision_to_cluster(
      v_ego,
      lead_msg,
      clusters,
    )

  else:
    cluster = None

  lead_dict = {
    'status': False,
  }

  if cluster is not None:
    lead_dict = cluster.get_RadarState2(
      lead_msg.prob,
      lead_msg,
      mixRadarInfo,
    )

  elif (
    cluster is None and
    ready and
    lead_msg.prob > .5
  ):
    lead_dict = Cluster().get_RadarState_from_vision(
      lead_msg,
      v_ego,
      model_v_ego,
    )

  if low_speed_override:
    low_speed_clusters = [
      c for c in clusters
      if c.potential_low_speed_lead(v_ego)
    ]

    if len(low_speed_clusters) > 0:
      closest_cluster = min(
        low_speed_clusters,
        key=lambda c: c.dRel,
      )

      if (
        not lead_dict['status'] or
        closest_cluster.dRel < lead_dict['dRel']
      ):
        lead_dict = closest_cluster.get_RadarState2(
          lead_msg.prob,
          lead_msg,
          mixRadarInfo,
        )

  return lead_dict


class RadarD():
  def __init__(self, radar_ts, delay=0):
    self.current_time = 0

    self.tracks = defaultdict(dict)
    self.kalman_params = KalmanParams(radar_ts)

    # ---------------------------------------------------------------------
    # v1.5.9 conservative far-target guards.
    #
    # leadOne:
    #   ~0.20 s confirmation, absolute maximum ~0.35 s
    #
    # leadTwo:
    #   shorter because secondary lead identity can change more frequently.
    # ---------------------------------------------------------------------
    self.lead_guard_one = FarLeadSwitchGuard(
      radar_ts,
      confirm_s=0.20,
      max_hold_s=0.35,
    )

    self.lead_guard_two = FarLeadSwitchGuard(
      radar_ts,
      confirm_s=0.15,
      max_hold_s=0.30,
    )

    self.v_ego = 0.
    self.v_ego_hist = deque(
      [0],
      maxlen=delay + 1,
    )

    self.ready = False
    self.showRadarInfo = False
    self.mixRadarInfo = 0

  def update(self, sm, rr):
    self.showRadarInfo = int(
      Params().get("ShowRadarInfo")
    )

    self.mixRadarInfo = int(
      Params().get("MixRadarInfo")
    )

    self.current_time = (
      1e-9 * max(sm.logMonoTime.values())
    )

    if sm.updated['carState']:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(
        self.v_ego
      )

    if sm.updated['modelV2']:
      self.ready = True

    ar_pts = {}

    for pt in rr.points:
      ar_pts[pt.trackId] = [
        pt.dRel,
        pt.yRel,
        pt.vRel,
        pt.measured,
      ]

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(
          ids,
          None,
        )

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      v_lead = (
        rpt[2] +
        self.v_ego_hist[0]
      )

      if ids not in self.tracks:
        self.tracks[ids] = Track(
          v_lead,
          self.kalman_params,
          track_id=ids,
        )

      self.tracks[ids].update(
        rpt[0],
        rpt[1],
        rpt[2],
        v_lead,
        rpt[3],
      )

    idens = list(
      sorted(self.tracks.keys())
    )

    track_pts = [
      self.tracks[iden].get_key_for_cluster()
      for iden in idens
    ]

    # If we have multiple points, cluster them
    if len(track_pts) > 1:
      cluster_idxs = cluster_points_centroid(
        track_pts,
        2.5,
      )

      clusters = [
        None
      ] * (
        max(cluster_idxs) + 1
      )

      for idx in range(len(track_pts)):
        cluster_i = cluster_idxs[idx]

        if clusters[cluster_i] is None:
          clusters[cluster_i] = Cluster()

        clusters[cluster_i].add(
          self.tracks[idens[idx]]
        )

    elif len(track_pts) == 1:
      cluster_idxs = [0]

      clusters = [
        Cluster()
      ]

      clusters[0].add(
        self.tracks[idens[0]]
      )

    else:
      clusters = []

    # if a new point, reset accel to the rest of the cluster
    for idx in range(len(track_pts)):
      if self.tracks[idens[idx]].cnt <= 1:
        aLeadK = clusters[
          cluster_idxs[idx]
        ].aLeadK

        aLeadTau = clusters[
          cluster_idxs[idx]
        ].aLeadTau

        self.tracks[
          idens[idx]
        ].reset_a_lead(
          aLeadK,
          aLeadTau,
        )

    # *** publish radarState ***
    dat = messaging.new_message(
      'radarState'
    )

    dat.valid = (
      sm.all_checks() and
      len(rr.errors) == 0
    )

    radarState = dat.radarState

    radarState.mdMonoTime = (
      sm.logMonoTime['modelV2']
    )

    radarState.radarErrors = list(
      rr.errors
    )

    radarState.carStateMonoTime = (
      sm.logMonoTime['carState']
    )

    if len(
      sm['modelV2'].temporalPose.trans
    ):
      model_v_ego = (
        sm['modelV2'].temporalPose.trans[0]
      )

    else:
      model_v_ego = self.v_ego

    leads_v3 = sm['modelV2'].leadsV3

    if len(leads_v3) > 1:
      lead1 = get_lead(
        self.v_ego,
        self.ready,
        clusters,
        leads_v3[0],
        model_v_ego,
        low_speed_override=True,
        mixRadarInfo=self.mixRadarInfo,
      )

      lead2 = get_lead(
        self.v_ego,
        self.ready,
        clusters,
        leads_v3[1],
        model_v_ego,
        low_speed_override=False,
        mixRadarInfo=self.mixRadarInfo,
      )

      # v1.5.9:
      # Normal leads pass raw.
      # Only suspicious far-target discontinuities get brief confirmation.
      lead1 = self.lead_guard_one.update(
        lead1,
        self.v_ego,
      )

      lead2 = self.lead_guard_two.update(
        lead2,
        self.v_ego,
      )

      # radarState schema may not contain trackId.
      lead1.pop(
        "trackId",
        None,
      )

      lead2.pop(
        "trackId",
        None,
      )

      radarState.leadOne = lead1
      radarState.leadTwo = lead2

      if (
        self.ready and
        self.showRadarInfo
      ):
        ll, lc, lr = get_path_adjacent_leads(
          self.v_ego,
          sm['modelV2'],
          sm['lateralPlan'].laneWidth,
          clusters,
        )

        for group in (
          ll,
          lc,
          lr,
        ):
          for ld in group:
            if isinstance(ld, dict):
              ld.pop(
                "trackId",
                None,
              )

        radarState.leadsLeft = list(ll)
        radarState.leadsCenter = list(lc)
        radarState.leadsRight = list(lr)

    return dat


# fuses camera and radar data for best lead detection
def radard_thread(
  sm=None,
  pm=None,
  can_sock=None,
):
  config_realtime_process(
    5 if TICI else 2,
    Priority.CTRL_LOW,
  )

  cloudlog.info(
    "radard is waiting for CarParams"
  )

  CP = car.CarParams.from_bytes(
    Params().get(
      "CarParams",
      block=True,
    )
  )

  cloudlog.info(
    "radard got CarParams"
  )

  cloudlog.info(
    "radard is importing %s",
    CP.carName,
  )

  RadarInterface = importlib.import_module(
    f'selfdrive.car.{CP.carName}.radar_interface'
  ).RadarInterface

  if can_sock is None:
    can_sock = messaging.sub_sock(
      'can'
    )

  if sm is None:
    sm = messaging.SubMaster(
      [
        'modelV2',
        'carState',
        'lateralPlan',
      ],
      ignore_avg_freq=[
        'modelV2',
        'carState',
        'lateralPlan',
      ],
    )

  if pm is None:
    pm = messaging.PubMaster(
      [
        'radarState',
        'liveTracks',
      ]
    )

  RI = RadarInterface(CP)

  rk = Ratekeeper(
    1.0 / CP.radarTimeStep,
    print_delay_threshold=None,
  )

  RD = RadarD(
    CP.radarTimeStep,
    RI.delay,
  )

  while 1:
    can_strings = messaging.drain_sock_raw(
      can_sock,
      wait_for_one=True,
    )

    rr = RI.update(
      can_strings
    )

    if rr is None:
      continue

    sm.update(0)

    dat = RD.update(
      sm,
      rr,
    )

    dat.radarState.cumLagMs = (
      -rk.remaining * 1000.
    )

    pm.send(
      'radarState',
      dat,
    )

    # *** publish tracks for UI debugging (keep last) ***
    tracks = RD.tracks

    dat = messaging.new_message(
      'liveTracks',
      len(tracks),
    )

    for cnt, ids in enumerate(
      sorted(tracks.keys())
    ):
      dat.liveTracks[cnt] = {
        "trackId": ids,
        "dRel": float(
          tracks[ids].dRel
        ),
        "yRel": float(
          tracks[ids].yRel
        ),
        "vRel": float(
          tracks[ids].vRel
        ),
      }

    pm.send(
      'liveTracks',
      dat,
    )

    rk.monitor_time()
    #rk.keep_time()


def main(
  sm=None,
  pm=None,
  can_sock=None,
):
  radard_thread(
    sm,
    pm,
    can_sock,
  )


if __name__ == "__main__":
  main()
