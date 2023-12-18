%%cu
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#define N 100

__global__ void calcGauss(float *A, float *B) {
    int col = threadIdx.x;

    for (int k = 0; k < N - 1; ++k) {
        if (col > k) {
            float factor = A[N * (col) + k] / A[N * k + k];
            for (int j = k; j < N; ++j) {
                A[N * col + j] -= factor * A[N * k + j];
            }
            B[col] -= factor * B[k];
        }
        __syncthreads();
    }

    if (col == N - 1) {
        B[N - 1] /= A[N * (N - 1) + (N - 1)];
        for (int i = N - 2; i >= 0; --i) {
            float sum = 0.0;
            for (int j = i + 1; j < N; ++j) {
                sum += A[N * i + j] * B[j];
            }
            B[i] = (B[i] - sum) / A[N * i + i];
        }
    }
}

int main() {
    float A[N * N], A1[N*N];
    float B[N], B1[N], Res[N];

    srand(time(0));
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[N * i + j] = rand() % (N * 100);
            A1[N * i + j] = A[N * i + j];
        }
        B[i] = rand() % (N * 100);
        B1[i] = B[i];
    }

    if (N < 30)
    {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << A[N * i + j] << "\t";
            }
            std::cout << "| " << B1[i] << std::endl;
        }
    }

    float *device_A, *device_B;
    cudaMalloc((void**)&device_A, N * N * sizeof(float));
    cudaMalloc((void**)&device_B, N * sizeof(float));

    cudaMemcpy(device_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    calcGauss<<<1, N>>>(device_A, device_B);

    cudaMemcpy(A, device_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, device_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);

    float sum = 0;
    for (int i = 0; i < N; ++i){
        float res = 0;
        for (int j = 0; j < N; ++j)
        {
             res += A1[i*N+j] * B[j];  
        }   
        Res[i] = res;
        sum += abs(res - B1[i]);
    }

    if (N < 50)
    {
        std::cout << "\nСheck:" << std::endl;
        for (int i = 0; i < N; ++i)
        {
            std::cout << "Expected result: " << B1[i] << ", Calculated result: " << Res[i] << std::endl << ", Error: " << Res[i] - B1[i] << std::endl;
        }

        printf("Transformed vector B:\n");
        for (int i = 0; i < N; ++i) {
            printf("X%d = %.2f\n", i, B[i]);
        }
    }
    else
      printf("Mean error: %.5f", (float)sum / N);

    return 0;
}