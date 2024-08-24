#include "main_functions.h"
#include "input.h"

# define Na_mean 2.084
# define Na_scale 0.545
# define K_mean 1.5889
# define K_scale 0.478
# define Cl_mean 2.738
# define Cl_scale 0.504

namespace {
  const tflite::Model* micro_model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 40 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

//static function to normalize
static float normalize(float input, float mean, float scale);
//static function to reverse
static float reverse(float input, float mean, float scale);

void model_setup() {
  micro_model = tflite::GetModel(model);
  if (micro_model == nullptr) {
    MicroPrintf("Modelo não foi carregado corretamente.");
    return;
  }

  static tflite::MicroMutableOpResolver<3> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Falha ao adicionar FullyConnected.");
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    MicroPrintf("Falha ao adicionar Relu.");
  }

  static tflite::MicroInterpreter static_interpreter(micro_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() falhou com status %d", allocate_status);
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  MicroPrintf("Modelo configurado com sucesso.\n");
}

void inferences() {
  for (size_t i = 0; i < num_of_inputs; i++) {

    // Copiando os dados de entrada para o tensor
    float* input_buffer = input->data.f;
    input_buffer[0] = normalize(inputs[i].Na, Na_mean, Na_scale);
    input_buffer[1] = normalize(inputs[i].K, K_mean, K_scale);

    result(inputs[i].Na, inputs[i].K);
  }
}

void result(float desnormalized_Na, float desnormalized_K) {
  float cl_output;

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Falha ao invocar o interpretador.");
    return;
  }

  cl_output = reverse(output->data.f[0], Cl_mean, Cl_scale);
  MicroPrintf("Na: %.2f, K: %.2f, Cl: %.2f", desnormalized_Na, desnormalized_K, cl_output);
  vTaskDelay(3000 / portTICK_PERIOD_MS);
}

static float normalize(float input, float mean, float scale){
    //log(1 + input)
    float log_input = log1p(input);
    //standard_scaler
    return (log_input - mean) / scale;
}

static float reverse(float input, float mean, float scale){
    // reverte a normalização padrão
    float scaled_input = (input * scale) + mean;
    //exp(x) - 1
    return expm1(scaled_input);
}