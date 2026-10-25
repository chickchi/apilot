"""v1.8.14 EON battery telemetry isolation.

Only the background worker may perform battery capacity/charge-type sysfs reads
and EON charge-limit I/O. thermald's publisher only reads cached snapshots.

This does not bypass communication failures or any openpilot safety checks.
"""
import threading
import time
from types import SimpleNamespace

from common.realtime import sec_since_boot
from selfdrive.hardware import HARDWARE
from selfdrive.swaglog import cloudlog


class EonBatteryAsync:
  MAX_AGE_S = 5.0
  SAMPLE_PERIOD_S = 1.0

  def __init__(self, stop_event, power_monitor, charging_control, charge_max, charge_min):
    self._stop_event = stop_event
    self._power_monitor = power_monitor
    self._charging_control = bool(charging_control)
    self._charge_max = int(charge_max)
    self._charge_min = int(charge_min)
    self._lock = threading.Lock()
    self._capacity = None
    self._charging = None
    self._current = 0
    self._usb_online = False
    self._power_draw = None
    self._timestamp = None
    self._usb_power = False
    self._stale_logged = None
    self._thread = threading.Thread(target=self._run, name="eon_battery_io", daemon=True)
    self._thread.start()

  def set_usb_power(self, enabled):
    # Updated from thermald; the worker never touches charging when false.
    with self._lock:
      self._usb_power = bool(enabled)

  def snapshot(self):
    """Return display and validated battery telemetry without any sysfs I/O.

    A missing/stale reading must not trigger battery-dependent charge and
    powerdown decisions. The last value is display-only when stale.
    """
    with self._lock:
      capacity, charging, ts = self._capacity, self._charging, self._timestamp
      current, usb, draw = self._current, self._usb_online, self._power_draw
    age = (time.monotonic() - ts) if ts is not None else None
    fresh = age is not None and age <= self.MAX_AGE_S
    if fresh:
      if self._stale_logged is not False:
        if self._stale_logged is True:
          cloudlog.warning("HKG_BATT_ASYNC_RECOVERED age=%.2fs", age)
        self._stale_logged = False
    elif self._stale_logged is not True:
      cloudlog.warning("HKG_BATT_ASYNC_STALE age=%s", "never" if age is None else "%.2fs" % age)
      self._stale_logged = True
    return (capacity if capacity is not None else 0,
            capacity if fresh else None, charging if fresh else None,
            current, usb, draw if fresh else None)

  def _run(self):
    while not self._stop_event.is_set():
      started = time.monotonic()
      try:
        capacity = HARDWARE.get_battery_capacity()
        if type(capacity) is not int or not 0 <= capacity <= 100:
          raise ValueError("out of range EON battery capacity")
        current = int(HARDWARE.get_battery_current())
        usb = bool(HARDWARE.get_usb_present())
        power_draw = HARDWARE.get_current_power_draw()
        charging = bool(HARDWARE.get_battery_charging())
        now = time.monotonic()
        with self._lock:
          self._capacity, self._charging, self._timestamp = capacity, charging, now
          self._current, self._usb_online, self._power_draw = current, usb, power_draw
          charge_allowed = self._usb_power and self._charging_control

        # Preserve the existing 70-80% charging hysteresis and >=30s guard.
        # Battery I/O may block indefinitely here, but never holds the lock
        # and never blocks thermald's deviceState publisher.
        if charge_allowed:
          msg = SimpleNamespace(deviceState=SimpleNamespace(batteryPercent=capacity))
          self._power_monitor.charging_ctrl(msg, sec_since_boot(), self._charge_max, self._charge_min)

      except Exception:
        cloudlog.exception("HKG_BATT_ASYNC_READ_ERROR")
      elapsed = time.monotonic() - started
      if elapsed >= 1.25:
        cloudlog.warning("HKG_BATT_ASYNC_SLOW duration=%.2fs", elapsed)
      self._stop_event.wait(max(0., self.SAMPLE_PERIOD_S - elapsed))
