#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define SIZE 2048            // Matrix size
#define SCALE_FACTOR 1.1     // Scaling factor for random values
#define NUM_BLOCKS 5        // Number of block sizes to test

// Declare matrices and arrays to store results
double matrixA[SIZE][SIZE];
double matrixB[SIZE][SIZE];
double matrixResult[SIZE][SIZE];

// Arrays to store performance data
int blockSizes[NUM_BLOCKS] = {32, 64, 128, 256, 512};   // Different block sizes
double executionTime[NUM_BLOCKS] = {0.0};           // Execution time for each block size
double cpuStart, cpuEnd, cpuTimeUsed[NUM_BLOCKS];   // CPU time start, end, and results

// Function to initialize matrices with random values
void initializeMatrix() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            srand(i + j);  // Seed with sum of indices for variety
            matrixA[i][j] = (rand() % 10) * SCALE_FACTOR;
            matrixB[i][j] = (rand() % 10) * SCALE_FACTOR;
        }
    }
}

// Function to reset the result matrix to 0 (useful for repeated tests)
void resetResultMatrix() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrixResult[i][j] = 0.0;
        }
    }
}

// Parallel matrix multiplication with block optimization using OpenMP
void multiplyMatrixWithBlocks(int blockSize, int blockIndex) {
    clock_t start = clock();                  // Start wall-clock time

    #pragma omp parallel for collapse(2)      // Parallelize by blocks
    for (int i = 0; i < SIZE; i += blockSize) {
        for (int j = 0; j < SIZE; j += blockSize) {
            for (int k = 0; k < SIZE; k += blockSize) {
                // Multiply blocks
                for (int ii = i; ii < i + blockSize && ii < SIZE; ii++) {
                    for (int jj = j; jj < j + blockSize && jj < SIZE; jj++) {
                        double temp = 0.0;  // Temporary variable to hold sum
                        for (int kk = k; kk < k + blockSize && kk < SIZE; kk++) {
                            temp += matrixA[ii][kk] * matrixB[kk][jj];
                        }
                        matrixResult[ii][jj] += temp;
                    }
                }
            }
        }
    }

    // Record time and CPU usage metrics
    clock_t end = clock();                   // End wall-clock time
    executionTime[blockIndex] = (double)(end - start) / CLOCKS_PER_SEC;      // Wall-clock time
}

// Main function to execute tests for each block size and print results
int main() {
    initializeMatrix();  // Initialize matrix values

    // Loop through different block sizes and perform block-optimized multiplication
    for (int i = 0; i < NUM_BLOCKS; i++) {
        int currentBlockSize = blockSizes[i];
        resetResultMatrix();   // Reset the result matrix for each test

        // Run multiplication with the current block size
        multiplyMatrixWithBlocks(currentBlockSize, i);

        // Print detailed results for each block size
        printf("Block Size: %d\n", currentBlockSize);
        printf("Execution Time (Wall-clock): %f seconds\n", executionTime[i]);
        printf("-------------------------------------------------\n");
    }

    // Summary of results
    printf("Summary of Performance Metrics:\n");
    printf("Block Size | Execution Time\n");
    for (int i = 0; i < NUM_BLOCKS; i++) {
        printf("%10d | %14.4f\n", blockSizes[i], executionTime[i]);
    }

    return 0;
}
