#ifndef INPUT_H_
#define INPUT_H_

#include "stdlib.h"

typedef struct InputData{
  float Na;
  float K;
};

extern InputData inputs[];
extern const size_t num_of_inputs; 

#endif // INPUT_H_