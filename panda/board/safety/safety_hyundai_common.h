#ifndef SAFETY_HYUNDAI_COMMON_H
#define SAFETY_HYUNDAI_COMMON_H

const int HYUNDAI_PARAM_EV_GAS = 1;
const int HYUNDAI_PARAM_HYBRID_GAS = 2;
const int HYUNDAI_PARAM_LONGITUDINAL = 4;
const int HYUNDAI_PARAM_CAMERA_SCC = 8;
const int HYUNDAI_PARAM_ALT_LIMITS = 64; // TODO: shift this down with the rest of the common flags
const int HYUNDAI_PARAM_AUTO_ENGAGE = 128;
const int HYUNDAI_PARAM_SCC_BUS2 = 256;

const uint8_t HYUNDAI_PREV_BUTTON_SAMPLES = 8;  // roughly 160 ms
const uint32_t HYUNDAI_STANDSTILL_THRSLD = 12;  // 0.375 kph

enum {
  HYUNDAI_BTN_NONE = 0,
  HYUNDAI_BTN_RESUME = 1,
  HYUNDAI_BTN_SET = 2,
  HYUNDAI_BTN_CANCEL = 4,
};

// common state
bool hyundai_ev_gas_signal = false;
bool hyundai_hybrid_gas_signal = false;
bool hyundai_longitudinal = false;
bool hyundai_camera_scc = false;
bool hyundai_alt_limits = false;
bool hyundai_auto_engage = false;
bool hyundai_scc_bus2 = false;
uint8_t hyundai_last_button_interaction;  // button messages since the user pressed an enable button

// v1.8.8-HKG:
// In the classic SCC-bus2 OP-long topology, APilot engagement ownership is:
//   MAIN         -> APilot/Lateral ON only (never OFF)
//   SET/RES      -> LongControl/speed only, after APilot is enabled
//   CANCEL short -> LongControl OFF only
//   CANCEL long  -> full APilot OFF
//
// Legacy safety intentionally clears hyundai_longitudinal after common init,
// so remember the requested LONG+SCC_BUS2 topology here before that happens.
// SCC11 MainMode_ACC is an OEM latch and must not own Panda controls_allowed.
const uint8_t HYUNDAI_HKG_CANCEL_LONG_FRAMES = 21U;  // ~0.4 s at 50 Hz CLU11

bool hyundai_hkg_main_control = false;
bool hyundai_main_button_prev = false;
bool hyundai_main_enable_pending = false;
uint8_t hyundai_cancel_hold_frames = 0U;

// v1.8.9-HKG: post-long-CANCEL synchronized MAIN TX authorization.
// Exactly two MAIN frames are allowed: one bus0 + one bus2.
uint8_t hyundai_hkg_main_sync_tx_budget = 0U;
uint8_t hyundai_hkg_main_sync_window = 0U;
bool hyundai_hkg_scc_main_on = false;

void hyundai_common_init(uint16_t param) {
  hyundai_ev_gas_signal = GET_FLAG(param, HYUNDAI_PARAM_EV_GAS);
  hyundai_hybrid_gas_signal = !hyundai_ev_gas_signal && GET_FLAG(param, HYUNDAI_PARAM_HYBRID_GAS);
  hyundai_camera_scc = GET_FLAG(param, HYUNDAI_PARAM_CAMERA_SCC);
  hyundai_alt_limits = GET_FLAG(param, HYUNDAI_PARAM_ALT_LIMITS);

  hyundai_last_button_interaction = HYUNDAI_PREV_BUTTON_SAMPLES;

#ifdef ALLOW_DEBUG
  hyundai_longitudinal = GET_FLAG(param, HYUNDAI_PARAM_LONGITUDINAL);
#else
  hyundai_longitudinal = false;
#endif
  hyundai_auto_engage = GET_FLAG(param, HYUNDAI_PARAM_AUTO_ENGAGE);
  hyundai_scc_bus2 = GET_FLAG(param, HYUNDAI_PARAM_SCC_BUS2);

  hyundai_hkg_main_control = hyundai_longitudinal && hyundai_scc_bus2;
  hyundai_main_button_prev = false;
  hyundai_main_enable_pending = false;
  hyundai_cancel_hold_frames = 0U;
  hyundai_hkg_main_sync_tx_budget = 0U;
  hyundai_hkg_main_sync_window = 0U;
  hyundai_hkg_scc_main_on = false;

  // A new safety session always starts disengaged in this HKG mode.
  // MAIN is the only physical control allowed to enable Panda actuation.
  if (hyundai_hkg_main_control) {
    controls_allowed = false;
  }
}

void hyundai_common_cruise_state_check(const int cruise_engaged) {
  // v1.8.8-HKG:
  // MainMode_ACC is an OEM SCC latch, not APilot engagement state.
  // Short and long CANCEL may both leave S11M=1, so SCC11 must never force
  // controls_allowed in HKG mode.
  if (hyundai_hkg_main_control) {
    hyundai_hkg_scc_main_on = cruise_engaged != 0;

    // Do not clear a post-button synchronization budget merely because SCC
    // MAIN is currently OFF: MAIN-while-enabled restoration is specifically
    // authorized after the real MAIN press has toggled SCC OFF.  The budget
    // is already bounded by hyundai_hkg_main_sync_window and driver input.
    cruise_engaged_prev = cruise_engaged;
    return;
  }

  // some newer HKG models can re-enable after spamming cancel button,
  // so keep track of user button presses to deny engagement if no interaction

  // enter controls on rising edge of ACC and recent user button press, exit controls when ACC off
  if (!hyundai_longitudinal) {
    if (cruise_engaged && !cruise_engaged_prev && (hyundai_last_button_interaction < HYUNDAI_PREV_BUTTON_SAMPLES)) {
      controls_allowed = true;
    }

    // Preserve the fork's existing behavior for every non-HKG topology.
    // Do not change unrelated Hyundai safety behavior in this patch.
    controls_allowed = true;
    cruise_engaged_prev = cruise_engaged;
  }
}

void hyundai_common_cruise_buttons_check(const int cruise_button, const int main_button) {
  if ((cruise_button == HYUNDAI_BTN_RESUME) || (cruise_button == HYUNDAI_BTN_SET) || (cruise_button == HYUNDAI_BTN_CANCEL) ||
      (main_button != 0)) {
    hyundai_last_button_interaction = 0U;
  } else {
    hyundai_last_button_interaction = MIN(hyundai_last_button_interaction + 1U, HYUNDAI_PREV_BUTTON_SAMPLES);
  }

  if (hyundai_hkg_main_control) {
    const bool main_now = main_button != 0;
    const bool main_press = main_now && !hyundai_main_button_prev;
    const bool main_release = !main_now && hyundai_main_button_prev;

    // Short authorization window (~160 ms at 50 Hz CLU11).  The host normally
    // consumes the two-frame budget immediately after long-CANCEL release.
    if (hyundai_hkg_main_sync_window > 0U) {
      hyundai_hkg_main_sync_window--;
      if (hyundai_hkg_main_sync_window == 0U) {
        hyundai_hkg_main_sync_tx_budget = 0U;
      }
    }

    if (cruise_button == HYUNDAI_BTN_CANCEL) {
      // A CANCEL press immediately belongs to LongControl on the host side,
      // but Panda keeps the actuation gate open while the driver is holding
      // the button.  Duration is classified only on physical release.
      hyundai_main_enable_pending = false;
      if (hyundai_cancel_hold_frames < HYUNDAI_HKG_CANCEL_LONG_FRAMES) {
        hyundai_cancel_hold_frames++;
      }
    } else {
      const bool cancel_release = cruise_button_prev == HYUNDAI_BTN_CANCEL;

      if (cancel_release && (hyundai_cancel_hold_frames >= HYUNDAI_HKG_CANCEL_LONG_FRAMES)) {
        // Long CANCEL: full APilot OFF, then permit exactly one synchronized
        // MAIN press on bus0 and bus2 to normalize both OEM MAIN phases OFF.
        controls_allowed = false;
        hyundai_main_enable_pending = false;
        hyundai_hkg_main_sync_tx_budget = 2U;
        hyundai_hkg_main_sync_window = 8U;
      }
      hyundai_cancel_hold_frames = 0U;

      // MAIN is enable-only.  When already allowed it is a no-op; it must
      // never close the Panda gate.  When disallowed, enable on release so
      // it aligns with interface/controlsd MAIN-release engagement.
      if (!cancel_release) {
        if (main_press) {
          // Any real driver MAIN press cancels stale synthetic-TX permission.
          hyundai_hkg_main_sync_tx_budget = 0U;
          hyundai_hkg_main_sync_window = 0U;
          hyundai_main_enable_pending = !controls_allowed;
        } else if (main_release) {
          if (hyundai_main_enable_pending) {
            // Only open Panda actuation if SCC actually reports MAIN ON.
            // This keeps a failed synchronization / phase-normalization press
            // from enabling actuation before the OEM states are aligned.
            controls_allowed = hyundai_hkg_scc_main_on;
            hyundai_main_enable_pending = false;
          } else if (controls_allowed && !hyundai_hkg_scc_main_on) {
            // MAIN was pressed while APilot was already enabled.  The physical
            // button is visible to bus0 EMS and is now also forwarded to bus2,
            // so both OEM MAIN states have toggled OFF.  Permit exactly one
            // synchronized host pulse to restore both states ON.
            hyundai_hkg_main_sync_tx_budget = 2U;
            hyundai_hkg_main_sync_window = 8U;
          }
        }
      }
    }

    hyundai_main_button_prev = main_now;

    // Keep common previous-button state coherent, but SET/RES must not
    // enable Panda controls in this HKG mode.
    cruise_button_prev = cruise_button;
    return;
  }

  if (hyundai_longitudinal) {
    // enter controls on falling edge of resume or set
    bool set = (cruise_button != HYUNDAI_BTN_SET) && (cruise_button_prev == HYUNDAI_BTN_SET);
    bool res = (cruise_button != HYUNDAI_BTN_RESUME) && (cruise_button_prev == HYUNDAI_BTN_RESUME);
    if (set || res) {
      controls_allowed = true;
    }

    // exit controls on cancel press
    //if (cruise_button == HYUNDAI_BTN_CANCEL) {
    //  controls_allowed = false;
    //}

    cruise_button_prev = cruise_button;
  }
}

#endif
