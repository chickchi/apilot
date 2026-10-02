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

// v1.8.10-HKG-OEM:
// In classic SCC-bus2 OP-long mode, preserve the OEM HKG master-state model:
//   MAIN    -> OEM cruise master ON/OFF and APilot/Lateral ON/OFF
//   SET/RES -> LongControl engage/speed while MAIN is ON
//   CANCEL  -> LongControl disengage only; MAIN stays ON
//
// Legacy safety clears hyundai_longitudinal after common init, so remember the
// requested LONG+SCC_BUS2 topology here. In this mode SCC11 MainMode_ACC is the
// authoritative Panda controls_allowed master state.
bool hyundai_hkg_main_control = false;

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

  // A new safety session starts disengaged. The first valid SCC11 frame then
  // makes controls_allowed follow the real OEM MainMode_ACC state.
  if (hyundai_hkg_main_control) {
    controls_allowed = false;
  }
}

void hyundai_common_cruise_state_check(const int cruise_engaged) {
  // v1.8.10-HKG-OEM: SCC11 MainMode_ACC is the master state. Since the real
  // physical MAIN is forwarded unchanged to SCC, this keeps Panda, bus0 EMS,
  // and bus2 SCC on the same OEM ON/OFF phase without synthetic messages.
  if (hyundai_hkg_main_control) {
    controls_allowed = cruise_engaged != 0;
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
    // v1.8.10-HKG-OEM: do not create a second software button state machine
    // inside Panda. SCC11 MainMode_ACC owns controls_allowed. CANCEL therefore
    // does not close the Panda gate, and SET/RES do not open it when MAIN is off.
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
