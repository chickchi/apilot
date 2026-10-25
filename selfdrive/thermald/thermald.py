#!/usr/bin/env python3
import datetime
import os
import queue
import threading
import time
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil

import cereal.messaging as messaging
from cereal import log
from common.dict_helpers import strip_deprecated_keys
from common.filter_simple import FirstOrderFilter
from common.params import Params
from common.realtime import DT_TRML, sec_since_boot
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.hardware import EON, HARDWARE, PC, TICI
from selfdrive.loggerd.config import get_available_percent
from selfdrive.statsd import statlog
from selfdrive.swaglog import cloudlog
from selfdrive.thermald.power_monitoring import PowerMonitoring
from selfdrive.thermald.eon_battery_async import EonBatteryAsync
from selfdrive.thermald.fan_controller import EonFanController, UnoFanController, TiciFanController
from selfdrive.version import terms_version, training_version
from selfdrive.ntune import ntune_option_enabled, ntune_option_get

ThermalStatus = log.DeviceState.ThermalStatus
NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength
CURRENT_TAU = 15.   # 15s time constant
TEMP_TAU = 5.   # 5s time constant
DISCONNECT_TIMEOUT = 5.  # wait 5 seconds before going offroad after disconnect so you get an alert
PANDA_STATES_TIMEOUT = int(1000 * 2.5 * DT_TRML)  # 2.5x the expected pandaState frequency

ThermalBand = namedtuple("ThermalBand", ['min_temp', 'max_temp'])
HardwareState = namedtuple("HardwareState", ['network_type', 'network_metered', 'network_strength', 'network_info', 'nvme_temps', 'modem_temps', 'wifi_address'])

# List of thermal bands. We will stay within this region as long as we are within the bounds.
# When exiting the bounds, we'll jump to the lower or higher band. Bands are ordered in the dict.
THERMAL_BANDS = OrderedDict({
  ThermalStatus.green: ThermalBand(None, 80.0),
  ThermalStatus.yellow: ThermalBand(75.0, 96.0),
  ThermalStatus.red: ThermalBand(80.0, 107.),
  ThermalStatus.danger: ThermalBand(94.0, None),
})

# Override to highest thermal band when offroad and above this temp
OFFROAD_DANGER_TEMP = 79.5 if TICI else 70.0

prev_offroad_states: Dict[str, Tuple[bool, Optional[str]]] = {}

tz_by_type: Optional[Dict[str, int]] = None
def populate_tz_by_type():
  global tz_by_type
  tz_by_type = {}
  for n in os.listdir("/sys/devices/virtual/thermal"):
    if not n.startswith("thermal_zone"):
      continue
    with open(os.path.join("/sys/devices/virtual/thermal", n, "type")) as f:
      tz_by_type[f.read().strip()] = int(n.lstrip("thermal_zone"))

def read_tz(x):
  if x is None:
    return 0

  if isinstance(x, str):
    if tz_by_type is None:
      populate_tz_by_type()
    x = tz_by_type[x]

  try:
    with open(f"/sys/devices/virtual/thermal/thermal_zone{x}/temp") as f:
      return int(f.read())
  except FileNotFoundError:
    return 0


def read_thermal(thermal_config):
  dat = messaging.new_message('deviceState')
  dat.deviceState.cpuTempC = [read_tz(z) / thermal_config.cpu[1] for z in thermal_config.cpu[0]]
  dat.deviceState.gpuTempC = [read_tz(z) / thermal_config.gpu[1] for z in thermal_config.gpu[0]]
  dat.deviceState.memoryTempC = read_tz(thermal_config.mem[0]) / thermal_config.mem[1]
  dat.deviceState.ambientTempC = read_tz(thermal_config.ambient[0]) / thermal_config.ambient[1]
  dat.deviceState.pmicTempC = [read_tz(z) / thermal_config.pmic[1] for z in thermal_config.pmic[0]]
  return dat


def set_offroad_alert_if_changed(offroad_alert: str, show_alert: bool, extra_text: Optional[str]=None):
  if prev_offroad_states.get(offroad_alert, None) == (show_alert, extra_text):
    return
  prev_offroad_states[offroad_alert] = (show_alert, extra_text)
  set_offroad_alert(offroad_alert, show_alert, extra_text)


def hw_state_thread(end_event, hw_queue):
  """Handles non critical hardware state, and sends over queue"""
  count = 0
  registered_count = 0
  prev_hw_state = None

  modem_version = None
  modem_nv = None
  modem_configured = False

  while not end_event.is_set():
    # these are expensive calls. update every 10s
    if (count % int(10. / DT_TRML)) == 0:
      try:
        network_type = HARDWARE.get_network_type()
        modem_temps = HARDWARE.get_modem_temperatures()
        if len(modem_temps) == 0 and prev_hw_state is not None:
          modem_temps = prev_hw_state.modem_temps

        # Log modem version once
        if TICI and ((modem_version is None) or (modem_nv is None)):
          modem_version = HARDWARE.get_modem_version()  # pylint: disable=assignment-from-none
          modem_nv = HARDWARE.get_modem_nv()  # pylint: disable=assignment-from-none

          if (modem_version is not None) and (modem_nv is not None):
            cloudlog.event("modem version", version=modem_version, nv=modem_nv)

        hw_state = HardwareState(
          network_type=network_type,
          network_metered=HARDWARE.get_network_metered(network_type),
          network_strength=HARDWARE.get_network_strength(network_type),
          network_info=HARDWARE.get_network_info(),
          nvme_temps=HARDWARE.get_nvme_temperatures(),
          modem_temps=modem_temps,
          wifi_address=HARDWARE.get_ip_address(),
        )

        try:
          hw_queue.put_nowait(hw_state)
        except queue.Full:
          pass

        if TICI and (hw_state.network_info is not None) and (hw_state.network_info.get('state', None) == "REGISTERED"):
          registered_count += 1
        else:
          registered_count = 0

        if registered_count > 10:
          cloudlog.warning(f"Modem stuck in registered state {hw_state.network_info}. nmcli conn up lte")
          os.system("nmcli conn up lte")
          registered_count = 0

        # TODO: remove this once the config is in AGNOS
        if not modem_configured and len(HARDWARE.get_sim_info().get('sim_id', '')) > 0:
          cloudlog.warning("configuring modem")
          HARDWARE.configure_modem()
          modem_configured = True

        prev_hw_state = hw_state
      except Exception:
        cloudlog.exception("Error getting hardware state")

    count += 1
    time.sleep(DT_TRML)


# v1.8.13-DIAG3: 1Hz watchdog inherited from DIAG2, plus ultra-light
# stage markers around device_metrics.  No exception swallowing or timeouts
# are added to hardware calls: this is observation only, not a safety bypass.
def thermald_diag2_mark(health, stage):
  now = time.monotonic()
  old_stage = health['stage']
  spent = now - health['stage_ts']
  # One recovery log after a slow *completed* stage.  The watchdog still
  # captures the stage name if a blocking call never returns.
  health['stage'] = stage
  health['stage_ts'] = now
  if old_stage and spent >= 1.25 and now >= health['next_slow_log']:
    health['next_slow_log'] = now + 15.0
    cloudlog.warning(
      f"THERMAL_DIAG3_SLOW stage={old_stage} duration={spent:.2f}s "
      f"next={stage} publishes={health['publishes']}"
    )


def thermald_diag2_watchdog(end_event, health):
  next_log = 0.0
  while not end_event.wait(1.0):
    try:
      last_send = health['last_send']
      if not last_send:
        continue  # Initial startup is excluded until the first deviceState publish.
      now = time.monotonic()
      age = now - last_send
      if age >= 3.0 and now >= next_log:
        stage = health['stage']
        stage_age = max(0.0, now - health['stage_ts'])
        cloudlog.warning(
          f"THERMAL_DIAG2_STALL age={age:.2f}s stage={stage} "
          f"stageAge={stage_age:.2f}s publishes={health['publishes']}"
        )
        next_log = now + 10.0  # Never spam the log while blocked.
    except Exception:
      pass  # Diagnostics must not interfere with device operation.


def thermald_thread(end_event, hw_queue, health):
  pm = messaging.PubMaster(['deviceState'])
  sm = messaging.SubMaster(["peripheralState", "gpsLocationExternal", "controlsState", "pandaStates"], poll=["pandaStates"])

  count = 0

  onroad_conditions: Dict[str, bool] = {
    "ignition": False,
  }
  startup_conditions: Dict[str, bool] = {}
  startup_conditions_prev: Dict[str, bool] = {}

  off_ts = None
  started_ts = None
  started_seen = False
  thermal_status = ThermalStatus.green
  usb_power = True

  last_hw_state = HardwareState(
    network_type=NetworkType.none,
    network_metered=False,
    network_strength=NetworkStrength.unknown,
    network_info=None,
    nvme_temps=[],
    modem_temps=[],
    wifi_address='N/A',
  )

  current_filter = FirstOrderFilter(0., CURRENT_TAU, DT_TRML)
  temp_filter = FirstOrderFilter(0., TEMP_TAU, DT_TRML)
  should_start_prev = False
  in_car = False
  is_uno = False
  engaged_prev = False

  params = Params()
  power_monitor = PowerMonitoring()

  HARDWARE.initialize_hardware()
  thermal_config = HARDWARE.get_thermal_config()

  fan_controller = None

  restart_triggered_ts = 0.
  panda_state_ts = 0.
  is_openpilot_view_enabled = 0
  opkrAutoShutdown = 0
  shutdown_trigger = 1

  #OPKR
  hotspot_run = False

  #hotspot_on_boot = False #ntune_option_enabled('OpkrHotspotOnBoot')
  hotspot_on_boot = params.get_bool("OpkrHotspotOnBoot")
  battery_charging_control = True #ntune_option_enabled('batteryChargingControl')
  battery_charging_min = 70 #int(ntune_option_get('batteryChargingMin'))
  battery_charging_max = 80 #int(ntune_option_get('batteryChargingMax'))

  #if params.get_bool("UseNpilotManager"):
  #  hotspot_on_boot = ntune_option_enabled('OpkrHotspotOnBoot')
  #  battery_charging_control = ntune_option_enabled('batteryChargingControl')
  #  battery_charging_min = int(ntune_option_get('batteryChargingMin'))
  #  battery_charging_max = int(ntune_option_get('batteryChargingMax'))
  #else:
  #  hotspot_on_boot = params.get_bool("OpkrHotspotOnBoot")
  #  battery_charging_control = params.get_bool("OpkrBatteryChargingControl")
  #  battery_charging_min = int(params.get("OpkrBatteryChargingMin", encoding="utf8"))
  #  battery_charging_max = int(params.get("OpkrBatteryChargingMax", encoding="utf8"))

  # EON sysfs battery I/O runs outside the deviceState publisher.
  # Original thermald diagnostics and safeguards remain unchanged.
  battery_async = EonBatteryAsync(end_event, power_monitor, battery_charging_control, battery_charging_max, battery_charging_min) if EON else None

  while not end_event.is_set():
    thermald_diag2_mark(health, 'panda_wait')
    ts = sec_since_boot()
    sm.update(PANDA_STATES_TIMEOUT)

    pandaStates = sm['pandaStates']
    peripheralState = sm['peripheralState']

    thermald_diag2_mark(health, 'read_thermal')
    msg = read_thermal(thermal_config)
    thermald_diag2_mark(health, 'device_metrics')

    # neokii
    thermald_diag2_mark(health, 'restart_params')
    if sec_since_boot() - restart_triggered_ts < 5.:
      onroad_conditions["not_restart_triggered"] = False
    else:
      onroad_conditions["not_restart_triggered"] = True

      if params.get_bool("SoftRestartTriggered"):
        params.put_bool("SoftRestartTriggered", False)
        restart_triggered_ts = sec_since_boot()

    thermald_diag2_mark(health, 'ignition_state')
    if sm.updated['pandaStates'] and len(pandaStates) > 0:

      # Set ignition based on any panda connected
      onroad_conditions["ignition"] = any(ps.ignitionLine or ps.ignitionCan for ps in pandaStates if ps.pandaType != log.PandaState.PandaType.unknown)

      pandaState = pandaStates[0]

      if pandaState.pandaType != log.PandaState.PandaType.unknown:
        panda_state_ts = sec_since_boot()

      in_car = pandaState.harnessStatus != log.PandaState.HarnessStatus.notConnected
      usb_power = peripheralState.usbPowerMode != log.PeripheralState.UsbPowerMode.client

      # Setup fan handler on first connect to panda
      #if fan_controller is None and peripheralState.pandaType != log.PandaState.PandaType.unknown:
      #  is_uno = peripheralState.pandaType == log.PandaState.PandaType.uno

      #  if TICI:
      #    fan_controller = TiciFanController()
      #  elif is_uno or PC:
      #    fan_controller = UnoFanController()
      #  else:
      #    fan_controller = EonFanController()
    #sunny
    #elif params.get_bool("IsOpenpilotViewEnabled") and is_openpilot_view_enabled == 0:
    #  is_openpilot_view_enabled = 1
    #  onroad_conditions["ignition"] = True
    #elif not params.get_bool("IsOpenpilotViewEnabled") and is_openpilot_view_enabled == 1:
    #  is_openpilot_view_enabled = 0
    #  onroad_conditions["ignition"] = False  
    #elif not params.get_bool("IsOpenpilotViewEnabled"):
    #  if sec_since_boot() - panda_state_ts > 3.:
    #    if onroad_conditions["ignition"]:
    #      cloudlog.error("Lost panda connection while onroad")
    #    onroad_conditions["ignition"] = False
    if sec_since_boot() - panda_state_ts > 3.:
      if onroad_conditions["ignition"]:
        cloudlog.error("Lost panda connection while onroad")
      onroad_conditions["ignition"] = False

    thermald_diag2_mark(health, 'hw_state_queue')
    try:
      last_hw_state = hw_queue.get_nowait()
    except queue.Empty:
      pass

    thermald_diag2_mark(health, 'free_space')
    msg.deviceState.freeSpacePercent = get_available_percent(default=100.0)
    thermald_diag2_mark(health, 'memory_usage')
    msg.deviceState.memoryUsagePercent = int(round(psutil.virtual_memory().percent))
    thermald_diag2_mark(health, 'cpu_usage')
    msg.deviceState.cpuUsagePercent = [int(round(n)) for n in psutil.cpu_percent(percpu=True)]
    thermald_diag2_mark(health, 'gpu_usage')
    msg.deviceState.gpuUsagePercent = int(round(HARDWARE.get_gpu_usage_percent()))

    thermald_diag2_mark(health, 'network_copy')
    msg.deviceState.networkType = last_hw_state.network_type
    msg.deviceState.networkMetered = last_hw_state.network_metered
    msg.deviceState.networkStrength = last_hw_state.network_strength
    if last_hw_state.network_info is not None:
      msg.deviceState.networkInfo = last_hw_state.network_info

    msg.deviceState.nvmeTempC = last_hw_state.nvme_temps
    msg.deviceState.modemTempC = last_hw_state.modem_temps
    msg.deviceState.wifiIpAddress = last_hw_state.wifi_address

    thermald_diag2_mark(health, 'screen_brightness')
    msg.deviceState.screenBrightnessPercent = HARDWARE.get_screen_brightness()
    thermald_diag2_mark(health, 'battery_snapshot')
    if battery_async is not None:
      battery_percent, battery_fresh_capacity, battery_fresh_charging, battery_current, usb_online, battery_draw = battery_async.snapshot()
      msg.deviceState.batteryPercent = battery_percent
      msg.deviceState.batteryCurrent = battery_current
      msg.deviceState.usbOnline = usb_online
    else:
      # Non-EON devices preserve their existing hardware read behavior.
      battery_fresh_capacity, battery_fresh_charging, battery_draw = None, None, None
      msg.deviceState.batteryPercent = HARDWARE.get_battery_capacity()
      msg.deviceState.batteryCurrent = HARDWARE.get_battery_current()
      msg.deviceState.usbOnline = HARDWARE.get_usb_present()
    current_filter.update(msg.deviceState.batteryCurrent / 1e6)

    thermald_diag2_mark(health, 'thermal_logic')
    max_comp_temp = temp_filter.update(
      max(max(msg.deviceState.cpuTempC), msg.deviceState.memoryTempC, max(msg.deviceState.gpuTempC))
    )

    if fan_controller is not None:
      msg.deviceState.fanSpeedPercentDesired = fan_controller.update(max_comp_temp, onroad_conditions["ignition"])

    is_offroad_for_5_min = (started_ts is None) and ((not started_seen) or (off_ts is None) or (sec_since_boot() - off_ts > 60 * 5))
    if is_offroad_for_5_min and max_comp_temp > OFFROAD_DANGER_TEMP:
      # If device is offroad we want to cool down before going onroad
      # since going onroad increases load and can make temps go over 107
      thermal_status = ThermalStatus.danger
    else:
      current_band = THERMAL_BANDS[thermal_status]
      band_idx = list(THERMAL_BANDS.keys()).index(thermal_status)
      if current_band.min_temp is not None and max_comp_temp < current_band.min_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx - 1]
      elif current_band.max_temp is not None and max_comp_temp > current_band.max_temp:
        thermal_status = list(THERMAL_BANDS.keys())[band_idx + 1]

    thermald_diag2_mark(health, 'startup_params')
    # **** starting logic ****

    # Ensure date/time are valid
    now = datetime.datetime.utcnow()
    startup_conditions["time_valid"] = True #(now.year > 2020) or (now.year == 2020 and now.month >= 10)
    set_offroad_alert_if_changed("Offroad_InvalidTime", (not startup_conditions["time_valid"]))

    startup_conditions["up_to_date"] = True #params.get("Offroad_ConnectivityNeeded") is None or params.get_bool("DisableUpdates") or params.get_bool("SnoozeUpdate")
    startup_conditions["not_uninstalling"] = not params.get_bool("DoUninstall")
    startup_conditions["accepted_terms"] = params.get("HasAcceptedTerms") == terms_version

    # with 2% left, we killall, otherwise the phone will take a long time to boot
    startup_conditions["free_space"] = msg.deviceState.freeSpacePercent > 2
    startup_conditions["completed_training"] = params.get("CompletedTrainingVersion") == training_version or \
                                               params.get_bool("Passive")
    startup_conditions["not_driver_view"] = not params.get_bool("IsDriverViewEnabled")
    startup_conditions["not_taking_snapshot"] = not params.get_bool("IsTakingSnapshot")
    # if any CPU gets above 107 or the battery gets above 63, kill all processes
    # controls will warn with CPU above 95 or battery above 60
    onroad_conditions["device_temp_good"] = thermal_status < ThermalStatus.danger
    set_offroad_alert_if_changed("Offroad_TemperatureTooHigh", (not onroad_conditions["device_temp_good"]))

    # TODO: this should move to TICI.initialize_hardware, but we currently can't import params there
    if TICI:
      if not os.path.isfile("/persist/comma/living-in-the-moment"):
        if not Path("/data/media").is_mount():
          set_offroad_alert_if_changed("Offroad_StorageMissing", True)
        else:
          # check for bad NVMe
          try:
            with open("/sys/block/nvme0n1/device/model") as f:
              model = f.read().strip()
            if not model.startswith("Samsung SSD 980") and params.get("Offroad_BadNvme") is None:
              set_offroad_alert_if_changed("Offroad_BadNvme", True)
              cloudlog.event("Unsupported NVMe", model=model, error=True)
          except Exception:
            pass

    thermald_diag2_mark(health, 'onroad_transition')
    # Handle offroad/onroad transition
    should_start = all(onroad_conditions.values())
    if started_ts is None:
      should_start = should_start and all(startup_conditions.values())

    if should_start != should_start_prev or (count == 0):
      params.put_bool("IsOnroad", should_start)
      params.put_bool("IsOffroad", not should_start)

      params.put_bool("IsEngaged", False)
      engaged_prev = False
      thermald_diag2_mark(health, 'set_power_save')
      HARDWARE.set_power_save(not should_start)
      thermald_diag2_mark(health, 'engagement_sync')

    thermald_diag2_mark(health, 'engagement_sync')
    if sm.updated['controlsState']:
      engaged = sm['controlsState'].enabled
      if engaged != engaged_prev:
        params.put_bool("IsEngaged", engaged)
        engaged_prev = engaged

      thermald_diag2_mark(health, 'kmsg_write')
      try:
        with open('/dev/kmsg', 'w') as kmsg:
          kmsg.write(f"<3>[thermald] engaged: {engaged}\n")
      except Exception:
        pass

    if should_start:
      off_ts = None
      if started_ts is None:
        started_ts = sec_since_boot()
        started_seen = True
    else:
      if onroad_conditions["ignition"] and (startup_conditions != startup_conditions_prev):
        cloudlog.event("Startup blocked", startup_conditions=startup_conditions, onroad_conditions=onroad_conditions)

      started_ts = None
      if off_ts is None:
        off_ts = sec_since_boot()
        
    thermald_diag2_mark(health, 'hotspot_check')
    # opkr hotspot
    if hotspot_on_boot and not hotspot_run and sec_since_boot() > 80:
      os.system("service call wifi 37 i32 0 i32 1 &")
      hotspot_run = True

    # Offroad power monitoring
    thermald_diag2_mark(health, 'power_monitor')
    power_monitor.calculate(peripheralState, onroad_conditions["ignition"])
    msg.deviceState.offroadPowerUsageUwh = power_monitor.get_power_used()
    msg.deviceState.carBatteryCapacityUwh = max(0, power_monitor.get_car_battery_capacity())
    thermald_diag2_mark(health, 'power_draw')
    current_power_draw = battery_draw if battery_async is not None else HARDWARE.get_current_power_draw()  # pylint: disable=assignment-from-none
    if current_power_draw is not None:
      statlog.sample("power_draw", current_power_draw)
      msg.deviceState.powerDrawW = current_power_draw
    else:
      msg.deviceState.powerDrawW = 0

    # Check if we need to disable charging (handled by boardd)
    thermald_diag2_mark(health, 'charging_status')
    msg.deviceState.chargingDisabled = power_monitor.should_disable_charging(onroad_conditions["ignition"], in_car, off_ts)

    # Check if we need to shut down
    thermald_diag2_mark(health, 'shutdown_check')
    if power_monitor.should_shutdown(peripheralState, onroad_conditions["ignition"], in_car, off_ts, started_seen, battery_fresh_capacity, battery_fresh_charging, battery_async is not None):
      cloudlog.warning(f"shutting device down, offroad since {off_ts}")
      params.put_bool("DoShutdown", True)

    msg.deviceState.chargingError = current_filter.x > 0. and msg.deviceState.batteryPercent < 90  # if current is positive, then battery is being discharged
    msg.deviceState.started = started_ts is not None
    msg.deviceState.startedMonoTime = int(1e9*(started_ts or 0))

    thermald_diag2_mark(health, 'last_athena_ping')
    last_ping = params.get("LastAthenaPingTime")
    if last_ping is not None:
      msg.deviceState.lastAthenaPingTime = int(last_ping)

    msg.deviceState.thermalStatus = thermal_status
    thermald_diag2_mark(health, 'device_publish')
    pm.send("deviceState", msg)
    published_at = time.monotonic()
    previous_send = health['last_send']
    health['last_send'] = published_at
    health['publishes'] += 1
    # Rare recovery log, NOT a log on every successful publish.
    if previous_send and published_at - previous_send >= 3.0:
      cloudlog.warning(f"THERMAL_DIAG2_GAP gap={published_at-previous_send:.2f}s publishes={health['publishes']}")
    thermald_diag2_mark(health, 'post_publish')

    if EON and not is_uno:
      set_offroad_alert_if_changed("Offroad_ChargeDisabled", (not usb_power))

    should_start_prev = should_start
    startup_conditions_prev = startup_conditions.copy()

    # Log to statsd
    statlog.gauge("free_space_percent", msg.deviceState.freeSpacePercent)
    statlog.gauge("gpu_usage_percent", msg.deviceState.gpuUsagePercent)
    statlog.gauge("memory_usage_percent", msg.deviceState.memoryUsagePercent)
    for i, usage in enumerate(msg.deviceState.cpuUsagePercent):
      statlog.gauge(f"cpu{i}_usage_percent", usage)
    for i, temp in enumerate(msg.deviceState.cpuTempC):
      statlog.gauge(f"cpu{i}_temperature", temp)
    for i, temp in enumerate(msg.deviceState.gpuTempC):
      statlog.gauge(f"gpu{i}_temperature", temp)
    statlog.gauge("memory_temperature", msg.deviceState.memoryTempC)
    statlog.gauge("ambient_temperature", msg.deviceState.ambientTempC)
    for i, temp in enumerate(msg.deviceState.pmicTempC):
      statlog.gauge(f"pmic{i}_temperature", temp)
    for i, temp in enumerate(last_hw_state.nvme_temps):
      statlog.gauge(f"nvme_temperature{i}", temp)
    for i, temp in enumerate(last_hw_state.modem_temps):
      statlog.gauge(f"modem_temperature{i}", temp)
    statlog.gauge("fan_speed_percent_desired", msg.deviceState.fanSpeedPercentDesired)
    statlog.gauge("screen_brightness_percent", msg.deviceState.screenBrightnessPercent)

    # atom
    thermald_diag2_mark(health, 'charging_ctrl')
    if battery_async is not None:
      battery_async.set_usb_power(usb_power)
    elif usb_power and battery_charging_control:
      power_monitor.charging_ctrl(msg, ts, battery_charging_max, battery_charging_min)

    # report to server once every 10 minutes
    thermald_diag2_mark(health, 'stats_or_status')
    if (count % int(600. / DT_TRML)) == 0:
      if EON and started_ts is None and msg.deviceState.memoryUsagePercent > 40:
        cloudlog.event("High offroad memory usage", mem=msg.deviceState.memoryUsagePercent)

      cloudlog.event("STATUS_PACKET",
                     count=count,
                     pandaStates=[strip_deprecated_keys(p.to_dict()) for p in pandaStates],
                     peripheralState=strip_deprecated_keys(peripheralState.to_dict()),
                     location=(strip_deprecated_keys(sm["gpsLocationExternal"].to_dict()) if sm.alive["gpsLocationExternal"] else None),
                     deviceState=strip_deprecated_keys(msg.to_dict()))

    thermald_diag2_mark(health, 'loop_end')
    count += 1


def main():
  hw_queue = queue.Queue(maxsize=1)
  end_event = threading.Event()
  health = {'last_send': 0.0, 'publishes': 0, 'stage': 'starting', 'stage_ts': time.monotonic(), 'next_slow_log': 0.0}

  threads = [
    threading.Thread(target=hw_state_thread, args=(end_event, hw_queue)),
    threading.Thread(target=thermald_thread, args=(end_event, hw_queue, health)),
  ]
  # Passive 1Hz watchdog; outside publisher's thread, without CAN or API writes.
  diag_watchdog = threading.Thread(target=thermald_diag2_watchdog, args=(end_event, health), daemon=True)
  diag_watchdog.start()
  for t in threads:
    t.start()

  try:
    while True:
      time.sleep(1)
      if not all(t.is_alive() for t in threads):
        break
  finally:
    end_event.set()

  for t in threads:
    t.join()
  diag_watchdog.join(timeout=1.0)


if __name__ == "__main__":
  main()
