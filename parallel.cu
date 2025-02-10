//
// Created by zak on 2/4/25.
//

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <device_launch_parameters.h>


struct HashingFunct {
    std::vector<double> a;
    double b;
    double w;
};

//  CUDA KERNELS

// CUDA Kernel to generate both normally distributed vector `a` and a uniform value `b`
__global__ void generateLSHParams(double* d_a, double* d_b, int n, double w, unsigned long long seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Compute global thread index
    curandState state;
    curand_init(seed, i, 0, &state); // Initialize cuRAND state

    if (i < n) {
        d_a[i] = curand_normal(&state); // Generate normally distributed value for a[i]
    }

    if (i == 0) { // Only thread 0 generates 'b'
        *d_b = curand_uniform(&state) * w; // Scale to range [0, w]
    }
}

// CUDA Kernel for element-wise multiplication and block-wise reduction
__global__ void multiplyElements(double* d_point, double* d_a, double* d_partialSum, int n) {
    __shared__ double sharedMem[256]; // Shared memory for block-level reduction
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    double product = (i < n) ? d_point[i] * d_a[i] : 0.0;
    sharedMem[tid] = product;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partialSum[blockIdx.x] = sharedMem[0]; //d_partial sum contains the sum of every block stored on it's index in the vector
    }
}

// CUDA Kernel for final reduction sum
__global__ void sumPartialSums(double* d_partialSum, double* d_finalSum, int numBlocks) {
    __shared__ double sharedMem[256];
    int tid = threadIdx.x;
    
    if (tid < numBlocks) {
        sharedMem[tid] = d_partialSum[tid];
    } else {
        sharedMem[tid] = 0.0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *d_finalSum = sharedMem[0];
    }
}

// CUDA Kernel to compute hashes in parallel
__global__ void computeHashes(double* d_points, int* d_hash1, int* d_hash2,
                              HashingFunct* d_hashfunctions1, HashingFunct* d_hashfunctions2,
                              int num_points, int L1, int L2, int D) {  // D est la dimension du point dans l'espace
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        for (int j = 0; j < L1; j++) {
            d_hash1[i * L1 + j] = hashingComputingCUDA(d_points + i * D, d_hashfunctions1[j]);
        }
        // à chaque point on applique L1 fonction de hachage et on stock dans i * L1 + j
        for (int j = 0; j < L2; j++) {
            d_hash2[i * L2 + j] = hashingComputingCUDA(d_points + i * D, d_hashfunctions2[j]);
        }
    }
}


// 2️⃣ HOST FUNCTIONS

std::vector<HashingFunct> generateLSHParamsOnGPU(int n, double w, int numFunctions) {
    HashingFunct* d_hashFunctions;
    cudaMalloc(&d_hashFunctions, numFunctions * sizeof(HashingFunct));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numFunctions + threadsPerBlock - 1) / threadsPerBlock;

    generateLSHParams<<<blocksPerGrid, threadsPerBlock>>>(d_hashFunctions, n, w, numFunctions, time(NULL));
    cudaDeviceSynchronize();

    std::vector<HashingFunct> hashFunctions(numFunctions);
    cudaMemcpy(hashFunctions.data(), d_hashFunctions, numFunctions * sizeof(HashingFunct), cudaMemcpyDeviceToHost);
    cudaFree(d_hashFunctions);

    return hashFunctions;
}

// Hashing computation using CUDA
// prend en paramètre le point qui est un vecteur et une fonction de hashage et return le hashage qui est un entier (bucket )
int hashingComputingCUDA(std::vector<double>& point, HashingFunct& h) {
    int n = point.size();
    double *d_point, *d_a, *d_partialSum, *d_finalSum;
    double h_finalSum;

    // Allocate memory on GPU
    cudaMalloc(&d_point, n * sizeof(double));
    cudaMalloc(&d_a, n * sizeof(double));

    int numBlocks = (n + 255) / 256; // fonction ceil(n/256)
    cudaMalloc(&d_partialSum, numBlocks * sizeof(double));
    cudaMalloc(&d_finalSum, sizeof(double));

    // Copy data from CPU to GPU
    cudaMemcpy(d_point, point.data(), n * sizeof(double), cudaMemcpyHostToDevice); // point.data() est bien un pointeur , point toute seul ne marche pas , un std::vector
    cudaMemcpy(d_a, h.a.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    multiplyElements<<<numBlocks, 256>>>(d_point, d_a, d_partialSum, n); // hypothese sur le hardware que le max de thread par bloc est 256 ( multiple de 32 pour le wraps )
    if (numBlocks > 1) {
        sumPartialSums<<<1, numBlocks>>>(d_partialSum, d_finalSum, numBlocks);
        cudaMemcpy(&h_finalSum, d_finalSum, sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(&h_finalSum, d_partialSum, sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Free GPU memory
    cudaFree(d_point);
    cudaFree(d_a);
    cudaFree(d_partialSum);
    cudaFree(d_finalSum);

    h_finalSum = (h_finalSum + h.b) / h.w;
    return static_cast<int>(h_finalSum);
}

// Host function to compute the final hash table
std::unordered_map<std::vector<int>, std::unordered_map<std::vector<int>, std::vector<std::vector<double>>>> finalHashCUDA(
        std::vector<std::vector<double>>& points, std::vector<HashingFunct>& hashfunctions1,
        std::vector<HashingFunct>& hashfunctions2, int L1, int L2) {

    int num_points = points.size();
    int D = points[0].size();
    int* d_hash1, * d_hash2; // vecteur ou sera stocké les hachages des points dans les 2 niveaux au niveau de la memoire global du gpu 
    double* d_points;
    HashingFunct* d_hashfunctions1, * d_hashfunctions2;

    cudaMalloc(&d_hash1, num_points * L1 * sizeof(int));
    cudaMalloc(&d_hash2, num_points * L2 * sizeof(int));
    cudaMalloc(&d_points, num_points * D * sizeof(double));
    cudaMalloc(&d_hashfunctions1, L1 * sizeof(HashingFunct));
    cudaMalloc(&d_hashfunctions2, L2 * sizeof(HashingFunct));

    cudaMemcpy(d_points, points.data(), num_points * D * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashfunctions1, hashfunctions1.data(), L1 * sizeof(HashingFunct), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashfunctions2, hashfunctions2.data(), L2 * sizeof(HashingFunct), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock; // vaut mieux remplacer par ceil num_points/threadsPerBlock
    // sou sl'hypothès que chaque point sera traité par un thread ( à revoir )
    computeHashes<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_hash1, d_hash2, d_hashfunctions1, d_hashfunctions2, num_points, L1, L2, D);  
    cudaDeviceSynchronize();

    std::vector<int> h_hash1(num_points * L1);
    std::vector<int> h_hash2(num_points * L2);
    cudaMemcpy(h_hash1.data(), d_hash1, num_points * L1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hash2.data(), d_hash2, num_points * L2 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_hash1);
    cudaFree(d_hash2);
    cudaFree(d_points);
    cudaFree(d_hashfunctions1);
    cudaFree(d_hashfunctions2);

    std::unordered_map<std::vector<int>, std::unordered_map<std::vector<int>, std::vector<std::vector<double>>>> result;
    for (int i = 0; i < num_points; i++) {
        std::vector<int> hash1(h_hash1.begin() + i * L1, h_hash1.begin() + (i + 1) * L1); // constructeur de vecteur en utilisant deux itérateurs 
        std::vector<int> hash2(h_hash2.begin() + i * L2, h_hash2.begin() + (i + 1) * L2);
        result[hash1][hash2].push_back(points[i]);
    }

    return result;
}

std::vector<std::vector<double>> loadDataFromCSV(const std::string& csvFilePath) {
    std::vector<std::vector<double>> dataPoints;
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Erreur : impossible d'ouvrir le fichier CSV");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> point;

        std::getline(ss, value, ',');  // Ignorer le premier champ

        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value));
        }

        dataPoints.push_back(point);
    }

    file.close();
    return dataPoints;
}

int main() {
    std::string csvFilePath = "../Data/fingerprints_class.csv";
    int L1 = 3, L2 = 2;
    double w1 = 10.0 , w2 = 5.0;
    
    std::vector<std::vector<double>> points = loadDataFromCSV(csvFilePath);
    int D = points[0].size();

    std::vector<HashingFunct> hashFunctions1 = generateLSHParamsOnGPU(D, w1, L1);
    std::vector<HashingFunct> hashFunctions2 = generateLSHParamsOnGPU(D, w2, L2);

    auto hashTable = finalHashCUDA(points, hashFunctions1, hashFunctions2, L1, L2);

    std::cout << "Hashing terminé !\n";

    return 0;
}