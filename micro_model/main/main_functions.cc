#include "main_functions.h"
#include "input.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <stdio.h>

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

// Função para normalizar valores
static float normalize(float input, float mean, float scale);

// Função para reverter a normalização
static float reverse(float input, float mean, float scale);

void model_setup() {
    micro_model = tflite::GetModel(model);
    if (micro_model == nullptr) {
        MicroPrintf("Modelo não foi carregado corretamente.");
        return;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk) {
        MicroPrintf("Falha ao adicionar FullyConnected.");
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
    MicroPrintf("Modelo configurado com sucesso.");
}

void inferences() {
    MicroPrintf("Iniciando casos de testes!");
    for (size_t i = 0; i < num_of_inputs; i++) {
        float cl_result;

        // Captura o tempo inicial antes da inferência
        int64_t start_time = esp_timer_get_time();
        
        // Copiando os dados de entrada para o tensor
        float* input_buffer = input->data.f;
        input_buffer[0] = normalize(inputs[i].Na, Na_mean, Na_scale);
        input_buffer[1] = normalize(inputs[i].K, K_mean, K_scale);

        // Realizando a inferência
        cl_result = result();

        // Captura o tempo final após a inferência
        int64_t end_time = esp_timer_get_time();

        // Calcula o tempo de inferência em microssegundos
        int64_t inference_time = end_time - start_time;

        // Imprimindo valores de Na, K e Cl na mesma linha com separador
        MicroPrintf("\nNa:%.2f K:%.2f Cl:%.2f Inf.Time:%lldµs", inputs[i].Na, inputs[i].K, cl_result, inference_time);
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    MicroPrintf("\nEncerrados dados de teste!");
}

float result() {
    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Falha ao invocar o interpretador.");
        return 0;
    }
    
    float cl_output = reverse(output->data.f[0], Cl_mean, Cl_scale);
    return cl_output;
}

// Função para normalizar os valores de entrada
static float normalize(float input, float mean, float scale) {
    // log(1 + input)
    float log_input = log1p(input);
    // standard_scaler
    return (log_input - mean) / scale;
}

// Função para reverter a normalização
static float reverse(float input, float mean, float scale) {
    // Reverte a normalização padrão
    float scaled_input = (input * scale) + mean;
    // exp(x) - 1
    return expm1(scaled_input);
}