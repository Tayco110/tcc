#include "main_functions.h"

extern "C" void app_main(void) {
  model_setup();
  while (true) {
    inferences();
    vTaskDelay(5000 / portTICK_PERIOD_MS);
  }
}