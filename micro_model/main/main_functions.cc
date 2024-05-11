#include "main_functions.h"
#include "input.h"

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 6 * 2048;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

//static function to normalize
static float normalize(float input, float mean, float scale);
//static function to reverse
static float reverse(float input, float mean, float scale);

// Estrutura para representar os dados de entrada

void model_setup() {

  model = tflite::GetModel(g_model);

  static tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void inferences() {

  for (size_t i = 0; i < num_of_inputs; i++) {
    MicroPrintf("Valor para a entrada %d:", i+1);
    
    // Copiando os dados de entrada para o tensor
    float* input_buffer = input->data.f;
    input_buffer[0] = normalize(inputs[i].massa, 366.06, 261.79);
    input_buffer[1] = normalize(inputs[i].Na, 8.37, 5.73);
    input_buffer[2] = normalize(inputs[i].K, 4.48, 2.72);

    result();
  }
}

void result() {

  float cl_output;

  interpreter->Invoke();
  cl_output = reverse(output->data.f[0], 16.55, 9.38);

  MicroPrintf("Valor de Cl: %f\n", cl_output);
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}

static float normalize(float input, float mean, float scale){
  return (input - mean)/scale;
}

static float reverse(float input, float mean, float scale){
  return (input * scale) + mean;
}