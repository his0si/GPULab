/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Kim Hee Seo
 *			   ID:	2276093
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

/*
 * =====================================================================================
 *
 *       Filename:  main.cu
 *
 *    Description: 	Matrix Multiplication
 *
 *        Version:  1.0
 *        Created:  2021/07/30 10:07:38
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Kim Hee Seo
 *			   ID:	2276093
 *   Organization:  Ewha Womans University
 *
 * =====================================================================================
 */

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "clockMeasure.h"

#define checkCudaError(error) 					\
	if(error != cudaSuccess){ 				\
		printf("%s in %s at line %d\n", 		\
				cudaGetErrorString(error), 	\
				__FILE__ ,__LINE__); 		\
		exit(EXIT_FAILURE);				\
	}

const int A_H = 512;
const int A_W = 512;
const int B_H = A_W;  // A의 열 크기와 B의 행 크기는 같아야 함
const int B_W = 512;
const unsigned int MAX_NUM = 100;

unsigned int matrixA[A_H * A_W];
unsigned int matrixB[B_H * B_W];
unsigned int gpuOut[A_H * B_W];

void generateRandomValues(unsigned int *input, const int rowSize, const int colSize) {
	for (int i = 0; i < rowSize; i++) {
		for (int j = 0; j < colSize; j++) {
			input[i * colSize + j] = rand() % MAX_NUM;
		}
	}
}

__global__
void gpuMatrixMul(unsigned int *d_a, unsigned int *d_b, unsigned int *d_c, const int aRowSize, const int aColSize, const int bRowSize, const int bColSize) {
	if (aColSize != bRowSize) return;  // 안전한 검증 추가
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < aRowSize * bColSize) {
		int rowId = tId / bColSize;
		int colId = tId % bColSize;
		unsigned int tSum = 0;
		for (int i = 0; i < aColSize; i++) {
			tSum += (d_a[rowId * aColSize + i] * d_b[i * bColSize + colId]);
		}
		d_c[tId] = tSum;
	}
}

int main() {
	srand((unsigned int)time(NULL));

	generateRandomValues(matrixA, A_H, A_W);
	generateRandomValues(matrixB, B_H, B_W);

	unsigned int *d_a, *d_b, *d_c;
	size_t sizeA = sizeof(unsigned int) * A_H * A_W;
	size_t sizeB = sizeof(unsigned int) * B_H * B_W;
	size_t sizeC = sizeof(unsigned int) * A_H * B_W;

	checkCudaError(cudaMalloc((void **)&d_a, sizeA));
	checkCudaError(cudaMalloc((void **)&d_b, sizeB));
	checkCudaError(cudaMalloc((void **)&d_c, sizeC));

	checkCudaError(cudaMemcpy(d_a, matrixA, sizeA, cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_b, matrixB, sizeB, cudaMemcpyHostToDevice));

	const int tbSize = 256;
	dim3 gridSize((A_H * B_W + tbSize - 1) / tbSize, 1, 1);
	dim3 blockSize(tbSize, 1, 1);

	clockMeasure *ckGpu = new clockMeasure("GPU CODE");
	ckGpu->clockReset();

	ckGpu->clockResume();
	gpuMatrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, A_H, A_W, B_H, B_W);
	checkCudaError(cudaDeviceSynchronize());
	ckGpu->clockPause();

	checkCudaError(cudaMemcpy(gpuOut, d_c, sizeC, cudaMemcpyDeviceToHost));

	ckGpu->clockPrint();

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("GPU Matrix Multiplication completed successfully.\n");
	return 0;
}
