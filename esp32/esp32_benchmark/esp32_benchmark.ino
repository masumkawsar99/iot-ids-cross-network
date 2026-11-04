// ESP32 benchmark of compact IoT-IDS models (generated). Board: ESP32 Dev Module. Serial Monitor: 115200 baud.
#include <Arduino.h>
#include "test_data.h"
#include "models.h"
#define REPS 20
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.printf("INFO,chip=%s,rev=%d,cpu_mhz=%d,flash_kb=%u,free_heap_b=%u\n", ESP.getChipModel(),
                (int)ESP.getChipRevision(), (int)getCpuFrequencyMhz(), (unsigned)(ESP.getFlashChipSize() / 1024),
                (unsigned)ESP.getFreeHeap());
  Serial.println("RESULT_HEADER,model,data_bytes,us_per_pred,agree,n,heap_before,heap_after,min_free_heap");
  for (int m = 0; m < N_MODELS; m++) {
    uint32_t heap0 = ESP.getFreeHeap();
    int agree = 0;
    volatile int sink = 0;
    for (int i = 0; i < N_TEST; i++) agree += (MODELS[m].fn(X_TEST[i]) == MODELS[m].pc_pred[i]);
    uint32_t t0 = micros();
    for (int r = 0; r < REPS; r++)
      for (int i = 0; i < N_TEST; i++) sink = sink + MODELS[m].fn(X_TEST[i]);
    uint32_t dt = micros() - t0;
    uint32_t heap1 = ESP.getFreeHeap();
    Serial.printf("RESULT,%s,%u,%.3f,%d,%d,%u,%u,%u\n", MODELS[m].name, (unsigned)MODELS[m].data_bytes,
                  (double)dt / ((double)REPS * N_TEST), agree, N_TEST, (unsigned)heap0, (unsigned)heap1,
                  (unsigned)ESP.getMinFreeHeap());
  }
  Serial.println("DONE");
}
void loop() {}
