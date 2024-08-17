#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>

#include "matrix.h"

#define BLOCK_SIZE 4

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void BlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    //@@ Insert code to implement block matrix multiply here
    int rows, cols, ops, row_blocks, col_blocks, ops_blocks;

    rows = result->shape[0];
    cols = result->shape[1];
    ops = input0->shape[1];

    row_blocks = ceil((float)rows / (float)BLOCK_SIZE);
    col_blocks = ceil((float)cols / (float)BLOCK_SIZE);
    ops_blocks = ceil((float)ops / (float)BLOCK_SIZE);

    for (int rr = 0; rr < row_blocks + 1; rr++)
    {
        for (int cc = 0; cc < col_blocks + 1; cc++)
        {
            for (int kk = 0; kk < ops_blocks + 1; kk++)
            {
                for (int r = rr * BLOCK_SIZE; r < (rr + 1) * BLOCK_SIZE; r++)
                {
                    for (int c = cc * BLOCK_SIZE; c < (cc + 1) * BLOCK_SIZE; c++)
                    {
                        if (r < rows && c < cols)
                        {
                            int k = kk * BLOCK_SIZE;

                            if (k + (BLOCK_SIZE - 1) < ops)
                            {
                                float data1[BLOCK_SIZE];

                                data1[0] = input1->data[cols * (k + 0) + c];
                                data1[1] = input1->data[cols * (k + 1) + c];
                                data1[2] = input1->data[cols * (k + 2) + c];
                                data1[3] = input1->data[cols * (k + 3) + c];

                                float32x4_t vector0 = vld1q_f32(&input0->data[ops * r + k]);
                                float32x4_t vector1 = vld1q_f32(data1);

                                float32x4_t result_vector = vmulq_f32(vector0, vector1);
                                result->data[cols * r + c] += vaddvq_f32(result_vector);
                            }
                            else
                            {
                                // Do some operations
                                for (; k < (kk + 1) * BLOCK_SIZE; k++)
                                {
                                    if (k < ops)
                                    {
                                        result->data[cols * r + c] += input0->data[ops * r + k] * input1->data[cols * k + c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix
    rows = host_a.shape[0];
    cols = host_b.shape[1];

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (float *)calloc(sizeof(float), host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    BlockMatrixMultiply(&host_a, &host_b, &host_c);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Check the result of the matrix multiply
    err = CheckMatrix(&answer, &host_c);
    CHECK_ERR(err, "CheckMatrix");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}