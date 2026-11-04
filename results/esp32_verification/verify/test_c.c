#include <stdio.h>
#include <stdint.h>
#include "test_data.h"
#include "models.h"
int main(void) {
  for (int m = 0; m < N_MODELS; m++) {
    int agree = 0;
    for (int i = 0; i < N_TEST; i++) agree += (MODELS[m].fn(X_TEST[i]) == MODELS[m].pc_pred[i]);
    printf("%s,%d,%d\n", MODELS[m].name, agree, N_TEST);
  }
  return 0;
}
