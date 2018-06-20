#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <CL/opencl.h>

/**
*   OpenCL Kernel source.
    Takes Two Matrices A and B and outputs their multiplication in C
*
**/
const char *kernelSource =                                                              "\n"\
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                                            \n"\
"__kernel void cl_multiplier(const int A_row, const int B_col, const int Common_factor,     \n"\    
"                        const __global float* Mat_A,    // Read-Only                      \n"\
"                        const __global float* Mat_B,    // Read-Only                       \n"\
"                        __global float* C) // Read and Write (Output) i.e A*B=C           \n"\
"{                                                                                          \n"\
"    // Threads Identifiers for Global Space                                                \n"\
"                                                                                           \n"\
"    const int globalRow = get_global_id(0);                                                \n"\
"    const int globalCol = get_global_id(1);                                                \n"\
"                                                                                           \n"\
"    // Inner loop over common factor                                                       \n"\
"    float cell_value = 0.0f;                                                               \n"\
"    for (int k=0; k<Common_factor; k++) {                                                  \n"\
"        cell_value += Mat_A[k*A_row + globalRow] * Mat_B[globalCol*Common_factor + k];     \n"\
"    }                                                                                      \n"\
"                                                                                           \n"\
"    // Output matrix C                                                                     \n"\
"    C[globalCol*A_row + globalRow] = cell_value;                                           \n"\
"}                                                                                          \n"\
                                                                                            "\n";




#define DEFAULT_SIZE 32

int main(int argc, char* argv[])
{
    const int k = DEFAULT_SIZE;
    const int m = DEFAULT_SIZE;
    const int n = DEFAULT_SIZE;

    printf("Default Size is : %d\n",DEFAULT_SIZE);

    // Fill A and B

    float* A = (float*)malloc(m*k*sizeof(float*));
    float* B = (float*)malloc(k*n*sizeof(float*));
    float* C = (float*)malloc(m*n*sizeof(float*));
    for (int i=0; i<m*k; i++) {
        A[i] = (float)rand() / (float)DEFAULT_SIZE;
        printf("A[%d] = %f\n",i,A[i]);
    }
    for (int i=0; i<k*n; i++) {
        B[i] = (float)rand() / (float)DEFAULT_SIZE;
        printf("B[%d] = %f\n",i,B[i]);
    }

}
