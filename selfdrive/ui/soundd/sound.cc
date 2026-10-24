#include "selfdrive/ui/soundd/sound.h"

#include <cmath>

#include <QAudio>
#include <QAudioDeviceInfo>
#include <QDebug>

#include "cereal/messaging/messaging.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/swaglog.h"  // v1.8.13-DIAG3 native sound diagnostics

// TODO: detect when we can't play sounds
// TODO: detect when we can't display the UI

Sound::Sound(QObject *parent) : sm({"carState", "controlsState", "deviceState"}) {
  qInfo() << "default audio device: " << QAudioDeviceInfo::defaultOutputDevice().deviceName();

  for (auto &[alert, fn, loops] : sound_list) {
    QSoundEffect *s = new QSoundEffect(this);
    QObject::connect(s, &QSoundEffect::statusChanged, [=]() {
      // Existing assertion remains unchanged. Log *before* it for diagnosis.
      if (s->status() == QSoundEffect::Error) {
        LOGE("HKG_DIAG3_SOUND_ERROR sound=%d status=%d", static_cast<int>(alert), static_cast<int>(s->status()));
      }
      assert(s->status() != QSoundEffect::Error);
    });
    if (alert == AudibleAlert::WARNING_SOFT || alert == AudibleAlert::WARNING_IMMEDIATE) {
      QObject::connect(s, &QSoundEffect::playingChanged, this, [s, alert]() {
        LOGW("HKG_DIAG3_SOUND_PLAYING sound=%d playing=%d status=%d volume=%.3f",
             static_cast<int>(alert), static_cast<int>(s->isPlaying()),
             static_cast<int>(s->status()), static_cast<double>(s->volume()));
      });
    }
    s->setVolume(Hardware::MIN_VOLUME);
    s->setSource(QUrl::fromLocalFile("../../assets/sounds/" + fn));
    sounds[alert] = {s, loops};
  }

  QTimer *timer = new QTimer(this);
  QObject::connect(timer, &QTimer::timeout, this, &Sound::update);
  timer->start(1000 / UI_FREQ);
};

void Sound::update() {
  const bool started_prev = sm["deviceState"].getDeviceState().getStarted();
  sm.update(0);

  const bool started = sm["deviceState"].getDeviceState().getStarted();
  if (started && !started_prev) {
    started_frame = sm.frame;
  }

  // no sounds while offroad
  // also no sounds if nothing is alive in case thermald crashes while offroad
  const bool crashed = (sm.frame - std::max(sm.rcv_frame("deviceState"), sm.rcv_frame("controlsState"))) > 10*UI_FREQ;
  if (!started || crashed) {
    // Only log when this guard actually interrupts an existing warning.
    if (current_alert.sound == AudibleAlert::WARNING_SOFT ||
        current_alert.sound == AudibleAlert::WARNING_IMMEDIATE) {
      LOGW("HKG_DIAG3_SOUND_CLEAR started=%d crashed=%d devFrames=%llu ctlFrames=%llu",
           static_cast<int>(started), static_cast<int>(crashed),
           static_cast<unsigned long long>(sm.frame - sm.rcv_frame("deviceState")),
           static_cast<unsigned long long>(sm.frame - sm.rcv_frame("controlsState")));
    }
    setAlert({});
    return;
  }

  // scale volume with speed
  if (sm.updated("carState")) {
    float volume = util::map_val(sm["carState"].getCarState().getVEgo(), 11.f, 20.f, 0.f, 1.0f);
    volume = QAudio::convertVolume(volume, QAudio::LogarithmicVolumeScale, QAudio::LinearVolumeScale);
    volume = util::map_val(volume, 0.f, 1.f, Hardware::MIN_VOLUME, Hardware::MAX_VOLUME);
    for (auto &[s, loops] : sounds) {
      s->setVolume(std::round(100 * volume) / 100);
    }
  }

  setAlert(Alert::get(sm, started_frame));
}

void Sound::setAlert(const Alert &alert) {
  if (!current_alert.equal(alert)) {
    current_alert = alert;
    // stop sounds
    for (auto &[s, loops] : sounds) {
      // Only stop repeating sounds
      if (s->loopsRemaining() > 1 || s->loopsRemaining() == QSoundEffect::Infinite) {
        s->stop();
      }
    }

    // play sound. Log only the safety-warning edge (not at UI_FREQ).
    if (alert.sound != AudibleAlert::NONE) {
      auto &[s, loops] = sounds[alert.sound];
      if (alert.sound == AudibleAlert::WARNING_SOFT || alert.sound == AudibleAlert::WARNING_IMMEDIATE) {
        LOGW("HKG_DIAG3_SOUND_REQUEST sound=%d status=%d volume=%.3f devFrames=%llu ctlFrames=%llu",
             static_cast<int>(alert.sound), static_cast<int>(s->status()),
             static_cast<double>(s->volume()),
             static_cast<unsigned long long>(sm.frame - sm.rcv_frame("deviceState")),
             static_cast<unsigned long long>(sm.frame - sm.rcv_frame("controlsState")));
      }
      s->setLoopCount(loops);
      s->play();
      if (alert.sound == AudibleAlert::WARNING_SOFT || alert.sound == AudibleAlert::WARNING_IMMEDIATE) {
        LOGW("HKG_DIAG3_SOUND_PLAY_CALL sound=%d status=%d playing=%d",
             static_cast<int>(alert.sound), static_cast<int>(s->status()), static_cast<int>(s->isPlaying()));
      }
    }
  }
}
