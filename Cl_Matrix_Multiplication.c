#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <CL/opencl.h>


/**
*   OpenCL Kernel source.
*   Takes Two Matrices A and B and outputs their multiplication in C
*   
*   @Author: Mahir Gulzar (mahirgulzar@gmail.com)
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


#define DEFAULT_SIZE 1024

#define WORKERS 64

int main(int argc, char* argv[])
{

    const int k = DEFAULT_SIZE;
    const int m = DEFAULT_SIZE;
    const int n = DEFAULT_SIZE;

    printf("\n===========< CL Matrix Multiplier >=============\n\n");
    printf("Default Matrix Size is : %d\n\n",DEFAULT_SIZE);
    printf("Number of Workers : %d\n\n",WORKERS);

    printf("Choose Device type:\n1-CPU\n2-GPU\n");
    char choice[2];
    // fgets make sure user doesn't buffer overflow here (Just Secure Programming things :D)
    fgets(choice,2,stdin);
    int result=strcmp("1",choice);

    // free(choice);


    // Fill A and B

    float* A = (float*)malloc(m*k*sizeof(float*));
    float* B = (float*)malloc(k*n*sizeof(float*));
    float* C = (float*)malloc(m*n*sizeof(float*));
    for (int i=0; i<m*k; i++) {
        A[i] = (float)rand() / (float)DEFAULT_SIZE;
        // printf("A[%d] = %f\n",i,A[i]);
    }
    for (int i=0; i<k*n; i++) {
        B[i] = (float)rand() / (float)DEFAULT_SIZE;
        // printf("B[%d] = %f\n",i,B[i]);
    }
    for (int i=0; i<m*n; i++) { C[i] = 0.0; }

    cl_int err;

    cl_platform_id platform =0;       // available platforms
    cl_device_id device =0;           // device id
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
    cl_event event=NULL;

    err = clGetPlatformIDs(1,&platform,NULL);
    

    if(result==0)
    {
        printf("\n\nYour choice is CPU\n");
        err = clGetDeviceIDs(platform,CL_DEVICE_TYPE_CPU, 1 , &device, NULL);
    }
    else 
    {
        printf("\n\nYour choice is GPU\n");
        err = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU, 1 , &device, NULL);
    }

    context = clCreateContext(NULL,1,&device,NULL,NULL,NULL);
    queue = clCreateCommandQueue(context,device,0,NULL);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);


    // make cl memory objects with respective read/write permissions reflected in kernel arguments
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  m*k*sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  k*n*sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, NULL);


    // enqueue matrix buffers with local contents of the matrices A, B , C 
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, m*k*sizeof(float), A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, k*n*sizeof(float), B, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, m*n*sizeof(float), C, 0, NULL, NULL);


    // Set our kernel arguments
    kernel = clCreateKernel(program, "cl_multiplier", NULL);
    err = clSetKernelArg(kernel, 0, sizeof(int), (void*)&m);
    err |= clSetKernelArg(kernel, 1, sizeof(int), (void*)&n);
    err |= clSetKernelArg(kernel, 2, sizeof(int), (void*)&k);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);


    //---------------------------------------------------------------------------------------

    // For calculating performance factor
    struct timeval Tvalue;
    struct timezone dummy;
    gettimeofday(&Tvalue, &dummy);
    double starttime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

    // Allocate local and global sizes for device with number of workers and rows and columns
    const size_t local[2] = { WORKERS, WORKERS };
    const size_t global[2] = { m, n };

    // Run Kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

    // clWaitForEvents(1, &event);
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Code Reference for performance calculation
    // https://github.com/CNugteren/myGEMM

    gettimeofday(&Tvalue, &dummy);
    double endtime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);
    double runtime = (endtime - starttime) / (double)1;
    double gflop = ((long)k * (long)m * (long)n * 2) / (1000*1000*1000);
    printf("\n>>> Done: took %.3lf seconds per run, %.1lf GFLOPS\n", runtime, gflop/runtime);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, m*n*sizeof(float), C, 0, NULL, NULL);

    printf("\n===========< Matrices multiplied >=============\n\n");

    // Clean-up memory
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    free(A);
    free(B);
    free(C);

    return 0;

}




