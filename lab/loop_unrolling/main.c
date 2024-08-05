#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }
    
    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *output_file = argv[3];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, output;

    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    output.shape[0] = 1;
    output.shape[1] = 1;
    output.data = (float*)malloc(sizeof(float) * rows * cols);

    // Create a new 4 divisible size
    int size = ceil((float)rows * (float)cols / 4.0f) * 4;
    // Use calloc instead of malloc to zero fill.
    float* data = (float *)calloc(size, sizeof(float));
    // Use memcpy to move all the data.
    memcpy(data, host_a.data, sizeof(float) * rows * cols);

    // Sum all elements of the array
    //@@ Modify the below code in the remaining demos
    float sum = 0;

    for (int i = 0; i < rows * cols; i += 4)
    {
        sum += data[i];
        sum += data[i+1];
        sum += data[i+2];
        sum += data[i+3];
    }

    printf("sum: %f == %f\n", sum, host_b.data[0]);

    output.data[0] = sum;
    err = CheckMatrix(&host_b, &output);
    CHECK_ERR(err, "CheckMatrix");
    SaveMatrix(output_file, &output);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(output.data);
    free(data);

    return 0;
}