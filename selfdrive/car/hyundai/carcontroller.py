from cereal import car
from common.conversions import Conversions as CV
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from selfdrive.car import apply_driver_steer_torque_limits
from selfdrive.car.hyundai import hyundaicanfd, hyundaican
from selfdrive.car.hyundai.values import HyundaiFlags, Buttons, CarControllerParams, CANFD_CAR, CAR, FEATURES
import random
from random import randint
from common.params import Params
from selfdrive.swaglog import cloudlog




VisualAlert = car.CarControl.HUDControl.VisualAlert
LongCtrlState = car.CarControl.Actuators.LongControlState




# EPS faults if you apply torque while the steering angle is above 90 degrees for more than 1 second
# All slightly below EPS thresholds to avoid fault
MAX_ANGLE = 85
MAX_ANGLE_FRAMES = 89
MAX_ANGLE_CONSECUTIVE_FRAMES = 2








def process_hud_alert(enabled, fingerprint, hud_control):
  sys_warning = (hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw))




  # initialize to no line visible
  # TODO: this is not accurate for all cars
  sys_state = 1
  if hud_control.leftLaneVisible and hud_control.rightLaneVisible or sys_warning:  # HUD alert only display when LKAS status is active
    sys_state = 3 if enabled or sys_warning else 4
  elif hud_control.leftLaneVisible:
    sys_state = 5
  elif hud_control.rightLaneVisible:
    sys_state = 6




  # initialize to no warnings
  left_lane_warning = 0
  right_lane_warning = 0
  if hud_control.leftLaneDepart:
    left_lane_warning = 1 if fingerprint in (CAR.GENESIS_G90, CAR.GENESIS_G80) else 2
  if hud_control.rightLaneDepart:
    right_lane_warning = 1 if fingerprint in (CAR.GENESIS_G90, CAR.GENESIS_G80) else 2




  return sys_warning, sys_state, left_lane_warning, right_lane_warning








class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.params = CarControllerParams(CP)
    self.packer = CANPacker(dbc_name)
    self.angle_limit_counter = 0
    self.frame = 0




    self.accel_last = 0
    self.apply_steer_last = 0
    self.car_fingerprint = CP.carFingerprint
    #self.send_lfa_mfa_lkas = True if self.car_fingerprint in FEATURES["send_lfa_mfa"] and self.car_fingerprint not in [CAR.HYUNDAI_GENESIS] else False
    self.send_lfa_mfa_lkas = CP.flags & HyundaiFlags.SEND_LFA.value
    self.last_button_frame = 0
    self.pcmCruiseButtonDelay = 0
    self.jerkStartLimit = 1.0
    self.speedCameraHapticEndFrame = 0
    self.hapticFeedbackWhenSpeedCamera = 0
    self.maxAngleFrames = MAX_ANGLE_FRAMES
    self.softHoldMode = 1
    self.blinking_signal = False #아이콘 깜박이용 1Hz
    self.blinking_frame = int(1.0 / DT_CTRL)
    self.steerDeltaUp = 3
    self.steerDeltaDown = 7
    self.button_wait = 12
    self.jerk_count = 0




    # for Legacy mode car auto resume (DH etc, not tested)
    self.resume_cnt = 0
    self.resume_wait_timer = 0
    self.button_alive = 0
    self.button_alive_frame = 0


    # v1.8.5.1: classic CAN SCC-bus2 OEM MAIN synchronization after CANCEL.
    # Pending synchronization is resolved only while APilot is disabled, no
    # physical cruise button is held, and SCC11 confirms OEM MAIN is still ON.
    self.main_sync_pending = False
    self.main_sync_attempts = 0
    self.main_sync_last_send_frame = -1000

    # v1.8.5.5: MAIN_SYNC is a short, timed MAIN hold rather than a
    # single-frame pulse.  Classic CLU11 runs at about 50 Hz, while the
    # controller loop is 100 Hz, so transmit one synthetic MAIN frame
    # every ~20 ms for ~300 ms, then allow ~200 ms for SCC11 feedback.
    #
    # Two bounded attempts are allowed.  Physical driver input always wins.
    self.main_sync_phase = 0  # 0=ready, 1=MAIN hold, 2=settle
    self.main_sync_phase_start_frame = -1000
    self.main_sync_tx_count = 0
    self.main_sync_hold_frames = max(1, int(round(0.30 / DT_CTRL)))
    self.main_sync_settle_frames = max(1, int(round(0.20 / DT_CTRL)))
    self.main_sync_tx_interval_frames = max(1, int(round(0.02 / DT_CTRL)))




  def update(self, CC, CS):
    actuators = CC.actuators
    hud_control = CC.hudControl




    # steering torque
    new_steer = int(round(actuators.steer * self.params.STEER_MAX))
    self.params.STEER_DELTA_UP = self.steerDeltaUp
    self.params.STEER_DELTA_DOWN = self.steerDeltaDown
    apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last, CS.out.steeringTorque, self.params)




    if not CC.latActive:
      apply_steer = 0




    self.apply_steer_last = apply_steer




    # accel + longitudinal
    accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)




    # v1.5.6 hardware-adjacent cruise-speed fail-safe.
    # hud_control.setSpeed and vEgoCluster are both SI (m/s) here.  This is
    # deliberately redundant with planner/LongControl so a stale/incorrect
    # positive request cannot reach Hyundai SCC merely because an upstream
    # safety layer failed.  Driver accelerator override remains available.
    cruise_target_ms = float(hud_control.setSpeed)
    cluster_speed_ms = float(CS.out.vEgoCluster)
    if (CC.longActive and not CC.cruiseControl.override and not CS.out.gasPressed and
        cruise_target_ms > 1.0 and cluster_speed_ms > 0.5):
      cruise_overspeed_kph = (cluster_speed_ms - cruise_target_ms) * CV.MS_TO_KPH
      if cruise_overspeed_kph > 1.0:
        controller_guard_cap = interp(
          cruise_overspeed_kph,
          [1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
          [0.00, -0.05, -0.10, -0.20, -0.35, -0.60],
        )
        accel = min(accel, controller_guard_cap)




    stopping = actuators.longControlState == LongCtrlState.stopping
    set_speed_in_units = hud_control.setSpeed * (CV.MS_TO_KPH if CS.is_metric else CV.MS_TO_MPH)




    # HUD messages
    sys_warning, sys_state, left_lane_warning, right_lane_warning = process_hud_alert(CC.enabled, self.car_fingerprint,
                                                                                      hud_control)




    if CC.activeHda == 2 and self.speedCameraHapticEndFrame < 0: # 과속카메라 감속시작
      self.speedCameraHapticEndFrame = self.frame + (8.0 / DT_CTRL)  #6초간 켜줌..
    elif CC.activeHda != 2:
      self.speedCameraHapticEndFrame = -1




    if self.frame < self.speedCameraHapticEndFrame and self.hapticFeedbackWhenSpeedCamera>0:
      haptic_stop = (self.speedCameraHapticEndFrame - (5.0/DT_CTRL)) < self.frame < (self.speedCameraHapticEndFrame - (3.0/DT_CTRL))
      if not haptic_stop:
         left_lane_warning = right_lane_warning = self.hapticFeedbackWhenSpeedCamera 
     
    if self.frame % self.blinking_frame == 0:
      self.blinking_signal = True
    elif self.frame % self.blinking_frame == self.blinking_frame / 2:
      self.blinking_signal = False




    jerk = actuators.jerk
    #jerk = accel - self.accel_last
    can_sends = []




    # *** common hyundai stuff ***
    if self.frame % 100 == 0:
      self.jerkStartLimit = float(int(Params().get("JerkStartLimit", encoding="utf8"))) * 0.1
      self.hapticFeedbackWhenSpeedCamera = int(Params().get("HapticFeedbackWhenSpeedCamera", encoding="utf8"))
      self.maxAngleFrames = int(Params().get("MaxAngleFrames", encoding="utf8"))
      self.softHoldMode = int(Params().get("SoftHoldMode", encoding="utf8"))
      self.steerDeltaUp = int(Params().get("SteerDeltaUp", encoding="utf8"))
      self.steerDeltaDown = int(Params().get("SteerDeltaDown", encoding="utf8"))




    # tester present - w/ no response (keeps relevant ECU disabled)
    if self.frame % 100 == 0 and not (self.CP.flags & HyundaiFlags.CANFD_CAMERA_SCC.value) and self.CP.openpilotLongitudinalControl:
      addr, bus = 0x7d0, 0
      if self.CP.flags & HyundaiFlags.CANFD_HDA2.value:
        addr, bus = 0x730, 5
      can_sends.append([addr, 0, b"\x02\x3E\x80\x00\x00\x00\x00\x00", bus])




    # >90 degree steering fault prevention
    # Count up to MAX_ANGLE_FRAMES, at which point we need to cut torque to avoid a steering fault
    if CC.latActive and abs(CS.out.steeringAngleDeg) >= MAX_ANGLE:
      self.angle_limit_counter += 1
    else:
      self.angle_limit_counter = 0




    # Cut steer actuation bit for two frames and hold torque with induced temporary fault
    torque_fault = CC.latActive and self.angle_limit_counter > self.maxAngleFrames
    lat_active = CC.latActive and not torque_fault




    if self.angle_limit_counter >= self.maxAngleFrames + MAX_ANGLE_CONSECUTIVE_FRAMES:
      self.angle_limit_counter = 0




    # CAN-FD platforms
    if self.CP.carFingerprint in CANFD_CAR:
      hda2 = self.CP.flags & HyundaiFlags.CANFD_HDA2
      hda2_long = hda2 and self.CP.openpilotLongitudinalControl




      # steering control
      can_sends.extend(hyundaicanfd.create_steering_messages(self.packer, self.CP, CC.enabled, lat_active, apply_steer))




      # disable LFA on HDA2
      if self.frame % 5 == 0 and hda2:
        can_sends.append(hyundaicanfd.create_cam_0x2a4(self.packer, CS.cam_0x2a4))




      # LFA and HDA icons
      if self.frame % 5 == 0 and (not hda2 or hda2_long):
        can_sends.append(hyundaicanfd.create_lfahda_cluster(self.packer, self.CP, CC.enabled))




      if self.CP.openpilotLongitudinalControl:
        if hda2:
          can_sends.extend(hyundaicanfd.create_adrv_messages(self.packer, self.frame))
        if self.frame % 2 == 0:
          can_sends.append(hyundaicanfd.create_acc_control(self.packer, self.CP, CC.enabled, self.accel_last, accel, stopping, CC.cruiseControl.override,
                                                           set_speed_in_units))
          self.accel_last = accel
      else:
        # button presses
        if (self.frame - self.last_button_frame) * DT_CTRL > 0.25:
          # cruise cancel
          if CC.cruiseControl.cancel:
            if self.CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS:
              can_sends.append(hyundaicanfd.create_acc_cancel(self.packer, self.CP, CS.cruise_info))
              self.last_button_frame = self.frame
            else:
              for _ in range(20):
                can_sends.append(hyundaicanfd.create_buttons(self.packer, self.CP, CS.buttons_counter+1, Buttons.CANCEL))
              self.last_button_frame = self.frame




          # cruise standstill resume
          elif CC.cruiseControl.resume:
            if self.CP.flags & HyundaiFlags.CANFD_ALT_BUTTONS:
              # TODO: resume for alt button cars
              pass
            else:
              for _ in range(20):
                can_sends.append(hyundaicanfd.create_buttons(self.packer, self.CP, CS.buttons_counter+1, Buttons.RES_ACCEL))
              self.last_button_frame = self.frame
    else:
      can_sends.append(hyundaican.create_lkas11(self.packer, self.frame, self.car_fingerprint, self.send_lfa_mfa_lkas, apply_steer, lat_active,
                                                torque_fault, CS.lkas11, sys_warning, sys_state, CC.enabled,
                                                hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                left_lane_warning, right_lane_warning))




      # v1.8.5.5: after a physical CANCEL, synchronize OEM SCC MAIN back
      # to OFF on classic CAN + openpilot longitudinal + SCC bus 2.
      #
      # Why a timed hold instead of the v1.8.5.4 single-frame pulse:
      # real-car logs showed TXBUS=2 was attempted twice, but SCC11
      # MainMode_ACC stayed at 1.  A physical MAIN press is held across
      # multiple CLU11 frames, so reproduce a short 50 Hz MAIN hold.
      #
      # Safety gates:
      # - classic CAN branch only
      # - SCC bus 2 + openpilot longitudinal only
      # - never transmit while APilot is enabled
      # - never transmit while any physical cruise/MAIN button is held
      # - transmit only while received SCC11.MainMode_ACC == 1
      # - at most two bounded hold attempts
      # - any real MAIN/SET/RES/GAP input cancels pending synchronization
      if self.CP.openpilotLongitudinalControl and self.CP.sccBus == 2:
        scc11_main_sync = getattr(CS, "scc11", None)
        main_mode_sync = (
          int(scc11_main_sync.get("MainMode_ACC", -1))
          if scc11_main_sync is not None
          else -1
        )

        physical_cruise_button = (
          int(CS.cruise_buttons[-1])
          if len(CS.cruise_buttons) > 0
          else Buttons.NONE
        )
        physical_main_button = (
          int(CS.main_buttons[-1])
          if len(CS.main_buttons) > 0
          else 0
        )

        # Arm once on CANCEL while OEM MAIN is still ON.  Do not repeatedly
        # reset the state while the driver continues to hold CANCEL.
        if (
          physical_cruise_button == Buttons.CANCEL and
          main_mode_sync == 1 and
          not self.main_sync_pending
        ):
          self.main_sync_pending = True
          self.main_sync_attempts = 0
          self.main_sync_phase = 0
          self.main_sync_phase_start_frame = self.frame
          self.main_sync_last_send_frame = (
            self.frame - self.main_sync_tx_interval_frames
          )
          self.main_sync_tx_count = 0
          cloudlog.info(
            f"[MAIN_SYNC] armed frame={self.frame} S11M={main_mode_sync}"
          )

        # Any new real driver request wins over synthetic synchronization.
        # CANCEL itself is excluded here because it is what arms the sync.
        if self.main_sync_pending and (
          physical_main_button != 0 or
          physical_cruise_button not in (Buttons.NONE, Buttons.CANCEL)
        ):
          cloudlog.info(
            f"[MAIN_SYNC] abort_input frame={self.frame} "
            f"MB={physical_main_button} CB={physical_cruise_button}"
          )
          self.main_sync_pending = False
          self.main_sync_attempts = 0
          self.main_sync_phase = 0
          self.main_sync_tx_count = 0

        if self.main_sync_pending:
          # SCC11 feedback is authoritative.  Stop immediately once MAIN
          # actually becomes OFF; the normal physical CLU11 stream supplies
          # the release (MAIN=0) frames after the synthetic hold stops.
          if main_mode_sync == 0:
            cloudlog.info(
              f"[MAIN_SYNC] done frame={self.frame} "
              f"attempts={self.main_sync_attempts} "
              f"tx={self.main_sync_tx_count}"
            )
            self.main_sync_pending = False
            self.main_sync_attempts = 0
            self.main_sync_phase = 0
            self.main_sync_tx_count = 0

          # If controls somehow re-engaged after CANCEL release, never leave
          # a delayed synthetic MAIN waiting to fire later.
          elif (
            CC.enabled and
            physical_cruise_button == Buttons.NONE and
            physical_main_button == 0
          ):
            cloudlog.info(
              f"[MAIN_SYNC] abort_enabled frame={self.frame} "
              f"S11M={main_mode_sync}"
            )
            self.main_sync_pending = False
            self.main_sync_attempts = 0
            self.main_sync_phase = 0
            self.main_sync_tx_count = 0

          elif (
            not CC.enabled and
            physical_cruise_button == Buttons.NONE and
            physical_main_button == 0 and
            main_mode_sync == 1
          ):
            # Phase 0: start one bounded MAIN-hold attempt.
            if self.main_sync_phase == 0:
              if self.main_sync_attempts < 2:
                self.main_sync_attempts += 1
                self.main_sync_phase = 1
                self.main_sync_phase_start_frame = self.frame
                self.main_sync_last_send_frame = (
                  self.frame - self.main_sync_tx_interval_frames
                )
                self.main_sync_tx_count = 0
                cloudlog.info(
                  f"[MAIN_SYNC] hold_start frame={self.frame} "
                  f"attempt={self.main_sync_attempts} "
                  f"S11M={main_mode_sync} TXBUS=2 "
                  f"hold_frames={self.main_sync_hold_frames}"
                )

            # Phase 1: hold MAIN=1 at roughly the physical CLU11 rate.
            if self.main_sync_phase == 1:
              hold_age = self.frame - self.main_sync_phase_start_frame

              if hold_age < self.main_sync_hold_frames:
                if (
                  self.frame - self.main_sync_last_send_frame >=
                  self.main_sync_tx_interval_frames
                ):
                  # Work on a private CLU11 copy; never mutate live CarState.
                  # create_clu11_button() advances the received alive counter
                  # by one, which keeps each synthetic frame aligned with the
                  # next expected CLU11 counter value.
                  main_sync_clu11 = dict(CS.clu11)
                  main_sync_clu11["CF_Clu_CruiseSwMain"] = 1

                  main_sync_msg = list(
                    hyundaican.create_clu11_button(
                      self.packer,
                      self.frame,
                      main_sync_clu11,
                      Buttons.NONE,
                      self.CP.carFingerprint,
                    )
                  )

                  # v1.8.5.4 finding retained: a host TX on bus 0 does not
                  # traverse Panda's RX forwarding path.  Deliver the
                  # synthetic MAIN directly to the SCC side on bus 2.
                  main_sync_msg[-1] = 2
                  can_sends.append(main_sync_msg)

                  self.main_sync_last_send_frame = self.frame
                  self.main_sync_tx_count += 1

              else:
                self.main_sync_phase = 2
                self.main_sync_phase_start_frame = self.frame
                cloudlog.info(
                  f"[MAIN_SYNC] hold_end frame={self.frame} "
                  f"attempt={self.main_sync_attempts} "
                  f"tx={self.main_sync_tx_count} S11M={main_mode_sync}"
                )

            # Phase 2: stop synthetic MAIN and allow normal MAIN=0 CLU11
            # frames plus SCC11 feedback time before deciding on a retry.
            elif self.main_sync_phase == 2:
              settle_age = self.frame - self.main_sync_phase_start_frame

              if settle_age >= self.main_sync_settle_frames:
                if self.main_sync_attempts < 2:
                  cloudlog.info(
                    f"[MAIN_SYNC] retry frame={self.frame} "
                    f"attempt={self.main_sync_attempts + 1} "
                    f"S11M={main_mode_sync}"
                  )
                  self.main_sync_phase = 0
                  self.main_sync_phase_start_frame = self.frame
                  self.main_sync_tx_count = 0

                else:
                  cloudlog.warning(
                    f"[MAIN_SYNC] failed frame={self.frame} "
                    f"attempts={self.main_sync_attempts} "
                    f"S11M={main_mode_sync}"
                  )
                  self.main_sync_pending = False
                  self.main_sync_attempts = 0
                  self.main_sync_phase = 0
                  self.main_sync_tx_count = 0


      if not self.CP.openpilotLongitudinalControl:
        if CC.cruiseControl.cancel:
          can_sends.append(hyundaican.create_clu11(self.packer, self.frame, CS.clu11, Buttons.CANCEL, self.CP.carFingerprint))
        elif CC.cruiseControl.resume:
          if self.CP.carFingerprint in LEGACY_SAFETY_MODE_CAR:            
            if self.resume_wait_timer > 0:
              self.resume_wait_timer -= 1
            else:
              can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP.carFingerprint))
              self.resume_cnt += 1
              if self.resume_cnt >= int(randint(4, 5) * 2):
                self.resume_cnt = 0
                self.resume_wait_timer = int(randint(20, 25) * 2)
              
          else:
            # send resume at a max freq of 10Hz
            if (self.frame - self.last_button_frame) * DT_CTRL > 0.1:
              # send 25 messages at a time to increases the likelihood of resume being accepted
              #can_sends.extend([hyundaican.create_clu11(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP.carFingerprint)] * 25)
              can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP.carFingerprint))
              self.last_button_frame = self.frame
        else:
          self.resume_wait_timer = 0
          self.resume_cnt = 0
          target = int(set_speed_in_units+0.5)
          current = int(CS.out.cruiseState.speed*CV.MS_TO_KPH + 0.5)




          #CC.debugTextCC = "BTN:00,T:{:.1f},C:{:.1f},{},{}".format(target, current, self.wait_timer, self.alive_timer)
          if (self.frame - self.last_button_frame) > self.button_wait:
            if (self.frame - self.button_alive_frame) > self.button_alive:
              self.button_wait = randint(8,15)
              self.last_button_frame = self.frame
            elif CC.enabled and CS.cruise_buttons[-1] == Buttons.NONE:
              if not CS.out.cruiseState.enabled:
                if CC.longActive and (hud_control.leadVisible or current > 10.0):
                  can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP.carFingerprint))
                  CC.debugTextCC = "BTN:++,T:{:.1f},C:{:.1f}".format(target, current)
                #elif CC.longActive:
                #  can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.SET_DECEL, self.CP.carFingerprint))
                #  CC.debugTextCC = "BTN:--,T:{:.1f},C:{:.1f}".format(target, current)
                #elif CS.out.cruiseGap != hud_control.cruiseGap:
                #  can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.GAP_DIST, self.CP.carFingerprint))
                #  #print("currentGap = {}, target = {}".format(CS.out.cruiseGap, hud_control.cruiseGap))
              elif CS.out.cruiseGap != hud_control.cruiseGap:
                can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.GAP_DIST, self.CP.carFingerprint))
                CC.debugTextCC = "currentGap = {}, target = {}".format(CS.out.cruiseGap, hud_control.cruiseGap)
              elif target < current and current>= 31:
                can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.SET_DECEL, self.CP.carFingerprint))
                CC.debugTextCC = "BTN:--,T:{:.1f},C:{:.1f}".format(target, current)
              elif target > current and current < 160:
                can_sends.append(hyundaican.create_clu11_button(self.packer, self.frame, CS.clu11, Buttons.RES_ACCEL, self.CP.carFingerprint))
                CC.debugTextCC = "BTN:++,T:{:.1f},C:{:.1f}".format(target, current)
          else:
            self.button_alive = randint(4, 8) #randint(12, 18)
            self.button_alive_frame = self.frame




      #CC.debugTextCC = "230206"




      if self.CP.carFingerprint in (CAR.GENESIS_G90_2019, CAR.GENESIS_G90, CAR.K7):
        can_sends.append(hyundaican.create_mdps12(self.packer, self.frame, CS.mdps12))




      if self.frame % 2 == 0 and self.CP.openpilotLongitudinalControl:
        # TODO: unclear if this is needed
        startingJerk = self.jerkStartLimit
        jerkLimit = 5.0
        self.jerk_count += DT_CTRL
        jerk_max = interp(self.jerk_count, [0, 1.5, 2.5], [startingJerk, startingJerk, jerkLimit])
        a_error = accel - CS.out.aEgo
        v_error = actuators.speed - CS.out.vEgo
        cb_upper = cb_lower = 0
        if actuators.longControlState == LongCtrlState.off:
          jerk_u = jerkLimit
          jerk_l = jerkLimit          
          self.jerk_count = 0
        elif actuators.longControlState == LongCtrlState.stopping or hud_control.softHold:
          jerk_u = 0.5
          jerk_l = jerkLimit
          self.jerk_count = 0
        else:
          jerk_u = min(max(0.5, jerk * 2.0), jerk_max)
          jerk_l = min(max(1.0, -jerk * 2.0), jerk_max)
          cb_upper = clip(0.9 + accel * 0.2, 0, 1.2)
          cb_lower = clip(0.8 + accel * 0.2, 0, 1.2)




        # v1.8.5.1: 5 Hz transmit-boundary + Hyundai SCC/button/main-sync trace.
        #
        # AV/CE : CarState cruise available/enabled
        # MB/CB : physical MAIN / RES-SET-GAP-CANCEL button state
        # S11M  : received SCC11.MainMode_ACC
        # S12A  : received SCC12.ACCMode
        # GAS   : physical accelerator state
        #
        # This is diagnostic only; it sends no additional CAN message.
        if self.frame % 20 == 0:
          try:
            scc11_rx = getattr(CS, "scc11", None)
            scc12_rx = getattr(CS, "scc12", None)

            s11_main = (
              int(scc11_rx.get("MainMode_ACC", -1))
              if scc11_rx is not None
              else -1
            )
            s12_acc = (
              int(scc12_rx.get("ACCMode", -1))
              if scc12_rx is not None
              else -1
            )

            main_button = (
              int(CS.main_buttons[-1])
              if len(CS.main_buttons) > 0
              else -1
            )
            cruise_button = (
              int(CS.cruise_buttons[-1])
              if len(CS.cruise_buttons) > 0
              else -1
            )

            cloudlog.info(
              f"[GEAR_TX] G={int(getattr(CS.out, 'currentGear', 0))}>"
              f"{int(getattr(CS.out, 'targetGear', 0))} "
              f"RPM={float(getattr(CS.out, 'tcuRpm', 0.0)):.0f} "
              f"SRC={float(actuators.accel):.2f} TX={float(accel):.2f} "
              f"AE={float(CS.out.aEgo):.2f} "
              f"EN={int(CC.enabled)} LE={int(CC.longEnabled)} "
              f"LA={int(CC.longActive)} OV={int(CC.cruiseControl.override)} "
              f"AV={int(CS.out.cruiseState.available)} "
              f"CE={int(CS.out.cruiseState.enabled)} "
              f"MB={main_button} CB={cruise_button} "
              f"S11M={s11_main} S12A={s12_acc} "
              f"GAS={int(CS.out.gasPressed)} "
              f"MSP={int(self.main_sync_pending)} "
              f"MSA={self.main_sync_attempts} "
              f"MSPH={self.main_sync_phase} "
              f"MSTX={self.main_sync_tx_count}"
            )
          except Exception:
            pass




        # v1.8.3: keep Hyundai SCC MAIN armed whenever APilot is enabled.
        # Actual longitudinal actuation is still gated inside hyundaican by
        # CC.longEnabled / CC.longActive, so MAIN-only sends no acceleration.
        can_sends.extend(hyundaican.create_acc_commands_mix_scc(self.CP, self.packer, CC.enabled, accel, jerk_u, jerk_l, int(self.frame / 2),
                                                      hud_control, set_speed_in_units, stopping, CC, CS, self.softHoldMode, cb_upper, cb_lower))
        self.accel_last = accel




      # 20 Hz LFA MFA message
      if self.frame % 5 == 0 and self.CP.flags & HyundaiFlags.SEND_LFA.value:
        can_sends.append(hyundaican.create_lfahda_mfc(self.packer, CC, self.blinking_signal))




      # 5 Hz ACC options
      if self.frame % 20 == 0 and self.CP.openpilotLongitudinalControl: 
        if self.CP.sccBus == 0:
          can_sends.extend(hyundaican.create_acc_opt(self.CP, CS, self.packer))
        elif CS.scc13 is not None:
          can_sends.append(hyundaican.create_acc_opt_copy(self.CP, CS, self.packer))




      # 2 Hz front radar options
      if self.frame % 50 == 0 and self.CP.openpilotLongitudinalControl  and self.CP.sccBus == 0:
        can_sends.append(hyundaican.create_frt_radar_opt(self.packer))




    new_actuators = actuators.copy()
    new_actuators.steer = apply_steer / self.params.STEER_MAX
    new_actuators.accel = accel




    self.frame += 1
    return new_actuators, can_sends
