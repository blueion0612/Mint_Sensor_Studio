// Copyright (c) 2026 Yuhyeon Lee
// SPDX-License-Identifier: MIT

/*
  imu_stream.ino
  MOBI311 Week 2, IMU System setup

  Reads the IMU on an Arduino Nano 33 BLE Sense and sends one sample at a time
  over the USB cable, over Bluetooth Low Energy, or both at once. Nothing else
  runs on the board, so the interval between samples is the interval between
  measurements.

  Board   Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE
  Library Tools > Manage Libraries > "Arduino_BMI270_BMM150"
          and, if USE_BLE is 1 below,  "ArduinoBLE"
  Speed   Tools > Serial Monitor > 115200 baud

  Over the cable one line looks like this.

      417,4185922,-0.03137,0.27100,-0.95923,-0.3662,0.6104,-0.3052,102.31,-11.06,180.44

      n        sample number, counts up by one, so a gap means a lost line
      micros   the board's clock in microseconds when the sample was read
      ax ay az acceleration in g          (1 g = 9.80665 m/s^2)
      gx gy gz angular rate in degrees per second
      mx my mz magnetic field in microtesla

  Lines that start with # are notes for the reader, not measurements.
  Send "?" from the Serial Monitor to print the header again.

  Over Bluetooth the same numbers go out as 18 raw bytes per sample instead of
  as text, because a radio packet holds 20 and the text above is 78. Nothing
  is lost: the bytes are the same measurements with the decimal point agreed
  in advance. See BLE_UUID below for what is in them.

  The built-in LED says what the board is doing.
      blinking once a second   waiting. Nothing has the cable open and no
                               phone or computer has connected over Bluetooth
      blinking twice a second  streaming
      fast blink               the IMU did not answer. Check BOARD_REV below
*/

// ===============================================================
// Set this to the revision printed on your board, then upload.
// Rev2 is the one handed out in class.
// ===============================================================

#define BOARD_REV 2      // 2 = Nano 33 BLE Sense Rev2.  1 = Nano 33 BLE Sense.

#define USE_BLE 1        // 1 = also send over Bluetooth.  0 = cable only.
                         //
                         // Measured on the board this was written for:
                         //   cable, USE_BLE 0        96 samples a second
                         //   cable, USE_BLE 1        70 samples a second
                         //   radio                   97 samples a second
                         //
                         // The radio is not free, but the cost falls on the
                         // cable, not on the radio. Simply having the
                         // Bluetooth stack running takes a quarter of the
                         // sketch's time whether or not anything has
                         // connected, and what it takes comes out of the
                         // sending over USB. Nothing in Week 2 needs those
                         // thirty samples. Set this to 0 to have them back.

// ===============================================================

#include <Wire.h>

#if BOARD_REV == 2
  #include <Arduino_BMI270_BMM150.h>          // Rev2: BMI270 and BMM150
  #define IMU_NAME "BMI270_BMM150"
#elif BOARD_REV == 1
  #include <Arduino_LSM9DS1.h>                // Rev1: LSM9DS1
  #define IMU_NAME "LSM9DS1"
#else
  #error "BOARD_REV must be 1 or 2."
#endif

#if USE_BLE
  #include <ArduinoBLE.h>
  #include <utility/ATT.h>      // for setMaxMtu, which ArduinoBLE does not expose
#endif

const char VERSION[] = "1.1";
const unsigned long BAUD = 115200;

unsigned long n = 0;             // sample number
unsigned long led_us = 0;        // last time the LED changed
bool led_on = false;
bool imu_ok = false;             // did the IMU answer at start-up
bool announced = false;          // has the header gone out to this listener
bool live = false;               // is a program on the computer listening
unsigned long checked_ms = 0;    // last time that was asked

// The gyroscope and the magnetometer are not always ready when the
// accelerometer is. The last value read is kept and sent again.
float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;
float mx = 0, my = 0, mz = 0;

#if USE_BLE
/*
  What goes out over the radio.

  The service and its three characteristics spell what they are: 494d554c is
  "IMUL" written in hexadecimal and 4142 is "AB". The last digit says which
  one it is. Nothing here names a course or a week, because the same board and
  the same sketch are used all term.

    ...0001  imu   four samples at a time, 18 bytes each, 72 in a packet
             uint32 micros
             int16  ax ay az     divide by 4096 for g
             int16  gx gy gz     divide by 16   for degrees per second
             uint16 n            the sample number, lowest sixteen bits
    ...0002  mag    6 bytes, sent about twelve times a second
             int16  mx my mz     divide by 16   for microtesla
    ...0003  info  the same header line the cable prints, readable once

  Four at a time, not one. A radio packet holds twenty bytes unless both ends
  agree on more, and the two ends only get one packet across per connection
  event. Windows settles on an event every fifteen to twenty milliseconds,
  which is fifty samples a second at one sample a packet, and the board makes
  seventy. The rest were being thrown away. Asking for a larger packet and
  filling it costs sixty milliseconds of delay and gets all of them through.

  Every board answers to the same service, so they are told apart by name:
  each advertises IMU-XXXX, where XXXX is the last four digits of its own
  Bluetooth address. A room full of boards is a room full of different names.
*/
#define BLE_UUID(last) "494d554c-4142-4000-8000-00000000000" last

BLEService imuService(BLE_UUID("0"));
const int BLE_RECORD = 18;       // bytes in one sample
const int BLE_BATCH = 4;         // samples in one packet

BLECharacteristic imuChar(BLE_UUID("1"), BLERead | BLENotify, BLE_RECORD * BLE_BATCH);
BLECharacteristic magChar(BLE_UUID("2"), BLERead | BLENotify, 6);
BLECharacteristic infoChar(BLE_UUID("3"), BLERead, 120);

// How many counts stand for one unit. Chosen so that the largest reading each
// sensor can produce still fits in a signed sixteen-bit number.
const float A_SCALE = 4096.0f;   // counts per g,               reaches 8 g
const float G_SCALE = 16.0f;     // counts per degree a second,  reaches 2048
const float M_SCALE = 16.0f;     // counts per microtesla,       reaches 2048

uint8_t ble_buf[BLE_RECORD * BLE_BATCH];
int ble_fill = 0;                // bytes waiting to go out

bool ble_ok = false;             // did the radio start
bool ble_live = false;           // is something connected to it
unsigned long polled_ms = 0;     // last time the radio was given a turn
char ble_name[16] = "IMU";

int16_t clamp16(float v) {
  if (v > 32767.0f) return 32767;
  if (v < -32768.0f) return -32768;
  return (int16_t)v;
}

void put16(uint8_t *p, int16_t v) {      // little endian, the way both ends read it
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
}
#endif


void header(char *out, size_t size) {
  snprintf(out, size, "#imu_stream %s imu=%s accel_hz=%.1f gyro_hz=%.1f mag_hz=%.1f",
           VERSION, IMU_NAME,
           IMU.accelerationSampleRate(), IMU.gyroscopeSampleRate(),
           IMU.magneticFieldSampleRate());
}


void print_header() {
  char line[140];
  header(line, sizeof(line));
  Serial.println(line);
  Serial.println("#COLUMNS n,micros,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mx_ut,my_ut,mz_ut");
}


void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  Serial.begin(BAUD);
  imu_ok = IMU.begin();

  // The IMU sits on the board's own I2C bus, which starts at 100 kHz. Reading
  // nine numbers takes several transactions, and at 100 kHz those cost more
  // than the sensor's own sampling interval. 400 kHz is inside spec for both
  // chips and is what lets the sketch keep up.
  Wire1.setClock(400000);

#if USE_BLE
  if (imu_ok) {
    ble_ok = BLE.begin();
  }
  if (ble_ok) {
    // The address is fixed in the chip, so a board keeps its name across
    // uploads and a student can write it on the box.
    String addr = BLE.address();          // "aa:bb:cc:dd:ee:ff"
    addr.replace(":", "");
    String tail = addr.substring(addr.length() - 4);
    tail.toUpperCase();
    snprintf(ble_name, sizeof(ble_name), "IMU-%s", tail.c_str());

    BLE.setLocalName(ble_name);
    BLE.setDeviceName(ble_name);
    BLE.setAdvertisedService(imuService);
    imuService.addCharacteristic(imuChar);
    imuService.addCharacteristic(magChar);
    imuService.addCharacteristic(infoChar);
    BLE.addService(imuService);

    char line[140];
    header(line, sizeof(line));
    infoChar.writeValue((const uint8_t *)line, strlen(line));

    // 1.25 ms per unit, so this asks for a packet every 7.5 to 15 ms. The
    // computer at the other end has the final say and often wants slower.
    BLE.setConnectionInterval(6, 12);
    // Room for one full batch and the three bytes of header around it.
    ATT.setMaxMtu(BLE_RECORD * BLE_BATCH + 3);
    BLE.advertise();
  }
#endif

  // Nothing is printed here. On this board Serial goes over USB, and writing
  // to a USB port that no program has open blocks for ever. Everything waits
  // for loop() to see a listener.
}


void loop() {
  unsigned long now_ms = millis();

  // Asking Serial whether anyone is listening costs a millisecond every time:
  // the core sleeps inside that test on purpose, so that the common
  // "while (!Serial)" wait cannot starve the USB thread. Asked once per
  // sample it would cap the sketch well below the sensor's own rate, so the
  // answer is kept and refreshed four times a second. Writing is safe in
  // between; the core drops what it cannot send.
  if (now_ms - checked_ms >= 250) {
    checked_ms = now_ms;
    live = (bool)Serial;
  }

#if USE_BLE
  // Once a millisecond, not once a loop. This loop spins tens of thousands of
  // times a second waiting for the next accelerometer sample, and polling the
  // radio on every one of those turns cost a quarter of the cable's sample
  // rate: 95 a second fell to 72. A thousand polls a second is still eight
  // times more often than the radio has anything to say, at the fastest
  // connection interval a phone or computer will agree to.
  if (ble_ok && now_ms != polled_ms) {
    polled_ms = now_ms;
    BLE.poll();                    // let the radio answer. Does not wait.
    ble_live = BLE.connected();
  }
  const bool ble_on = ble_live;
#else
  const bool ble_on = false;
#endif

  // Nobody on either link, nothing to do. delay() hands the processor to the
  // USB stack; a tight loop here would starve it and the board would stop
  // answering the computer altogether, uploads included.
  if (!live && !ble_on) {
    announced = false;
#if USE_BLE
    ble_fill = 0;
#endif
    digitalWrite(LED_BUILTIN, (now_ms / 1000) % 2 ? HIGH : LOW);
    delay(10);
    return;
  }

  if (!announced) {
    announced = true;
    n = 0;
    if (live) {
      print_header();
    }
  }

  if (!imu_ok) {
    // The board is fine, the IMU did not answer. Almost always BOARD_REV
    // set to the revision you do not have.
    if (live) {
      Serial.println("#ERROR IMU.begin() failed. Check BOARD_REV against the board you have.");
    }
    for (int i = 0; i < 10; i++) {
      digitalWrite(LED_BUILTIN, (i % 2) ? HIGH : LOW);
      delay(100);
    }
    return;
  }

  if (live && Serial.available() > 0) {
    if (Serial.read() == '?') {
      print_header();
    }
  }

  // The accelerometer sets the pace. One sample goes out per acceleration
  // reading, so the rate of the samples is the rate of the accelerometer.
  // yield() costs no time and lets the USB stack run while we wait.
  if (!IMU.accelerationAvailable()) {
    yield();
    return;
  }
  IMU.readAcceleration(ax, ay, az);

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }

  // The magnetometer produces about ten readings a second while the
  // accelerometer produces a hundred. Asking it every time spends an I2C
  // transaction to be told "not yet" nine times out of ten.
  bool new_mag = false;
  if ((n % 8) == 0 && IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    new_mag = true;
  }

  unsigned long us = micros();
  n++;

  if (live) {
    // The line is built first and sent in one piece. Serial.print sends one
    // character per USB transfer and every transfer waits for the next USB
    // frame, so printing the eleven numbers separately holds the whole sketch
    // down to about 22 lines a second instead of the sensor's hundred.
    char line[160];
    int len = snprintf(line, sizeof(line),
                       "%lu,%lu,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\r\n",
                       n, us, ax, ay, az, gx, gy, gz, mx, my, mz);
    if (len > 0) {
      if (len > (int)sizeof(line) - 1) {
        len = (int)sizeof(line) - 1;
      }
      Serial.write((const uint8_t *)line, (size_t)len);
    }
  }

#if USE_BLE
  if (ble_on) {
    uint8_t *buf = ble_buf + ble_fill;
    buf[0] = (uint8_t)(us & 0xFF);
    buf[1] = (uint8_t)((us >> 8) & 0xFF);
    buf[2] = (uint8_t)((us >> 16) & 0xFF);
    buf[3] = (uint8_t)((us >> 24) & 0xFF);
    put16(buf + 4, clamp16(ax * A_SCALE));
    put16(buf + 6, clamp16(ay * A_SCALE));
    put16(buf + 8, clamp16(az * A_SCALE));
    put16(buf + 10, clamp16(gx * G_SCALE));
    put16(buf + 12, clamp16(gy * G_SCALE));
    put16(buf + 14, clamp16(gz * G_SCALE));
    put16(buf + 16, (int16_t)(n & 0xFFFF));
    ble_fill += BLE_RECORD;

    if (ble_fill >= (int)sizeof(ble_buf)) {
      imuChar.writeValue(ble_buf, ble_fill);
      ble_fill = 0;
    }

    if (new_mag) {
      uint8_t mbuf[6];
      put16(mbuf + 0, clamp16(mx * M_SCALE));
      put16(mbuf + 2, clamp16(my * M_SCALE));
      put16(mbuf + 4, clamp16(mz * M_SCALE));
      magChar.writeValue(mbuf, sizeof(mbuf));
    }
  }
#endif

  // Quarter of a second on, quarter off, so that streaming looks different
  // from waiting. If the LED stops blinking the sketch stopped running.
  if (us - led_us > 250000UL) {
    led_us = us;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}
