// Copyright (c) 2026 Yuhyeon Lee
// SPDX-License-Identifier: MIT

/*
  analog_stream.ino
  Sensor Studio: an EMG amplifier, a pulse sensor, or any analog signal

  Reads one to four analog pins at a fixed rate and sends each sample as one
  line of text over the USB cable, in the same shape imu_stream.ino uses, so
  that Sensor Studio, the Week 2 scripts and a spreadsheet all read it without
  being told anything:

      #analog_stream 1.0 mode=emg hz=1000 bits=12 vref=3.300
      #COLUMNS n,micros,emg1_mv
      417,4185922,0.483
      418,4186922,-1.209

      n        sample number, counts up by one, so a gap means a lost line
      micros   the board's clock in microseconds when the sample was read
      then     one number per channel, named in the #COLUMNS line

  The names carry the unit, and Sensor Studio reads the kind of sensor off
  them: emg1_mv is a muscle, ir_raw and red_raw are a pulse oximeter, ch1_raw
  is an unnamed signal it draws as it comes.

  Board   Arduino Nano 33 BLE / Nano 33 BLE Sense (the class board), whose
          USB port is a native one and carries a thousand lines a second.
          An Uno or a Nano works at 115200 baud up to about RATE_HZ 250.
  Wiring  EMG: a MyoWare, Grove or similar amplifier's signal pin to A0,
               its supply to 3.3 V and its ground to GND. A second amplifier
               goes to A1, and so on.
          PPG: an analog pulse sensor's signal pin to A0. For two
               wavelengths (SpO2) a red-LED sensor's output to A1.
  Speed   Tools > Serial Monitor > 115200 baud

  Lines that start with # are notes for the reader, not measurements.
  Send "?" from the Serial Monitor to print the header again.

  The built-in LED says what the board is doing.
      blinking once a second   waiting. Nothing has the cable open
      blinking twice a second  streaming

  This sketch has been compiled for both boards. It has not been run against
  an EMG amplifier or a pulse sensor in the lab yet: the first person to do so
  should check the header's vref against the board they have and that the
  signal moves when the muscle does.
*/

// ===============================================================
// Set these three, then upload.
// ===============================================================

#define MODE      1        // 1 = EMG in millivolts   2 = PPG, raw counts   3 = raw counts, unnamed
#define N_CH      1        // how many pins, A0 upwards: 1 to 4
#define RATE_HZ   1000     // samples a second. EMG wants 1000, PPG 100, an Uno at most 250

// ===============================================================

const char VERSION[] = "1.0";
const unsigned long BAUD = 115200;

#if defined(ARDUINO_ARCH_MBED) || defined(ARDUINO_ARCH_SAMD) || defined(ARDUINO_ARCH_RENESAS)
  #define ADC_BITS 12
  #define HAS_RESOLUTION 1
#else
  #define ADC_BITS 10               // AVR: an Uno, a classic Nano
  #define HAS_RESOLUTION 0
#endif

#if defined(ARDUINO_ARCH_AVR)
  const float VREF = 5.0f;          // an Uno runs its ADC from 5 V
#else
  const float VREF = 3.3f;          // the Nano 33 BLE and most 3.3 V boards
#endif

const int PINS[4] = {A0, A1, A2, A3};
const unsigned long PERIOD_US = 1000000UL / RATE_HZ;
const long ADC_MAX = (1L << ADC_BITS) - 1;

unsigned long n = 0;               // sample number
unsigned long next_us = 0;         // when the next sample is due
unsigned long led_us = 0;
bool led_on = false;
bool announced = false;            // has the header gone out to this listener
bool live = false;                 // is a program on the computer listening
unsigned long checked_ms = 0;      // last time that was asked


void print_header() {
  char line[120];
  const char *mode = (MODE == 1) ? "emg" : (MODE == 2) ? "ppg" : "raw";
  // %f is not available on every board's snprintf, so vref is written by hand
  snprintf(line, sizeof(line), "#analog_stream %s mode=%s hz=%d bits=%d vref=%d.%03d",
           VERSION, mode, (int)RATE_HZ, (int)ADC_BITS,
           (int)VREF, (int)(VREF * 1000.0f) % 1000);
  Serial.println(line);

  // #COLUMNS n,micros, then one name per channel
  Serial.print("#COLUMNS n,micros");
  for (int i = 0; i < N_CH; i++) {
    Serial.print(",");
#if MODE == 1
    Serial.print("emg");
    Serial.print(i + 1);
    Serial.print("_mv");
#elif MODE == 2
    // infrared first, red second: the order Sensor Studio's SpO2 page expects
    Serial.print(i == 0 ? "ir_raw" : (i == 1 ? "red_raw" : "ppg_raw"));
    if (i >= 2) Serial.print(i + 1);
#else
    Serial.print("ch");
    Serial.print(i + 1);
    Serial.print("_raw");
#endif
  }
  Serial.println();
}


// Millivolts as text, in integer arithmetic, because avr-libc's printf has no
// floating point and prints a question mark for %f.
void put_mv(char *out, size_t size, long counts) {
  // counts * VREF * 1000 / ADC_MAX, kept in tenths of a millivolt
  long tenths = (long)((counts * (VREF * 10000.0f)) / (float)ADC_MAX);
  // centred: a single-supply amplifier idles at half its supply, and a muscle
  // at rest should read near zero, not near 1650 mV
  tenths -= (long)(VREF * 5000.0f);
  const char *sign = tenths < 0 ? "-" : "";
  if (tenths < 0) tenths = -tenths;
  snprintf(out, size, "%s%ld.%01ld", sign, tenths / 10, tenths % 10);
}


void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.begin(BAUD);
#if HAS_RESOLUTION
  analogReadResolution(ADC_BITS);
#endif
  for (int i = 0; i < N_CH && i < 4; i++) {
    pinMode(PINS[i], INPUT);
  }
  next_us = micros();
  // Nothing is printed here. On the Nano 33 BLE, Serial goes over USB, and
  // writing to a USB port that no program has open blocks for ever.
}


void loop() {
  unsigned long now_ms = millis();

  // Asking Serial whether anyone is listening costs a millisecond on the
  // Nano 33 BLE (the core sleeps inside that test on purpose), so the answer
  // is kept and refreshed four times a second. On an AVR board it is always
  // true, and the sketch simply streams.
  if (now_ms - checked_ms >= 250) {
    checked_ms = now_ms;
    live = (bool)Serial;
  }

  if (!live) {
    announced = false;
    digitalWrite(LED_BUILTIN, (now_ms / 1000) % 2 ? HIGH : LOW);
    delay(10);
    next_us = micros();
    return;
  }

  if (!announced) {
    announced = true;
    n = 0;
    print_header();
    next_us = micros();
  }

  if (Serial.available() > 0) {
    if (Serial.read() == '?') {
      print_header();
    }
  }

  // A fixed schedule, so that the interval between samples is the interval
  // between measurements. If a line took too long to send and the schedule
  // has slipped by more than a few periods, it is reset rather than caught
  // up in a burst.
  unsigned long now_us = micros();
  if ((long)(now_us - next_us) < 0) {
    return;
  }
  next_us += PERIOD_US;
  if ((long)(now_us - next_us) > (long)(4 * PERIOD_US)) {
    next_us = now_us + PERIOD_US;
  }

  long counts[4];
  for (int i = 0; i < N_CH && i < 4; i++) {
    counts[i] = analogRead(PINS[i]);
  }
  unsigned long us = micros();
  n++;

  // The line is built first and sent in one piece: one USB transfer per line
  // rather than one per character.
  char line[96];
  int len = snprintf(line, sizeof(line), "%lu,%lu", n, us);
  for (int i = 0; i < N_CH && i < 4 && len < (int)sizeof(line) - 12; i++) {
#if MODE == 1
    char mv[16];
    put_mv(mv, sizeof(mv), counts[i]);
    len += snprintf(line + len, sizeof(line) - len, ",%s", mv);
#else
    len += snprintf(line + len, sizeof(line) - len, ",%ld", counts[i]);
#endif
  }
  if (len > (int)sizeof(line) - 3) {
    len = (int)sizeof(line) - 3;
  }
  line[len++] = '\r';
  line[len++] = '\n';
  Serial.write((const uint8_t *)line, (size_t)len);

  // Quarter of a second on, quarter off, so that streaming looks different
  // from waiting. If the LED stops blinking the sketch stopped running.
  if (us - led_us > 250000UL) {
    led_us = us;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}
