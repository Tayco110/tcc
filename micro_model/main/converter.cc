#include <stdio.h>
#include <math.h> // Incluir biblioteca de matemática para funções exp

double log_transform_inverse(double log_value) {
    return exp(log_value) - 1;
}

double standard_scaler_inverse(double standardized_value, double mean, double std_dev) {
    return standardized_value * std_dev + mean;
}
