#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <device_launch_parameters.h>
#include <random>
#include <iostream>

struct HashingFunct {
    double* a;
    double b;
    double w;
};

__global__ void generateLSHParams(double* d_a, double* d_b, int n, double w, unsigned long long seed, int funcIndex) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    // une nouvelle initialization pour chaque fonction de hachage 
    curand_init(seed + funcIndex, i, 0, &state);
    
    int offset = funcIndex * n;
    if (i < n) {
        d_a[offset + i] = curand_normal(&state); 
        // les coordonnées du vecteur a sont générés d'une loi normale
    }
    if (i == 0) {
        d_b[funcIndex] = curand_uniform(&state) * w;
        // un unique b par fonction de hachage , généré par la loi uniforme 
    }
}

__device__ int hashingComputingCUDA(const double* d_point, const HashingFunct h, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += d_point[i] * h.a[i];
    }
    sum = (sum + h.b) / h.w;
    return static_cast<int>(sum);
}

__global__ void computeHashes(double* d_points, int* d_hash1, int* d_hash2,
                              HashingFunct* d_hashfunctions1, HashingFunct* d_hashfunctions2,
                              int num_points, int L1, int L2, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_points) {
        for (int j = 0; j < L1; j++) {
            d_hash1[i * L1 + j] = hashingComputingCUDA(d_points + i * D, d_hashfunctions1[j], D);
        }
        for (int j = 0; j < L2; j++) {
            d_hash2[i * L2 + j] = hashingComputingCUDA(d_points + i * D, d_hashfunctions2[j], D);
        }
    }
}

std::pair<std::vector<HashingFunct>, double*> generateLSHParamsOnGPU(int n, double w, int numFunctions) {
    double *d_a, *d_b;

    cudaMalloc(&d_a, numFunctions * n * sizeof(double));
    cudaMalloc(&d_b, numFunctions * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int funcIndex = 0; funcIndex < numFunctions; funcIndex++) {
        generateLSHParams<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n, w, time(NULL) + funcIndex * 1234, funcIndex);
        cudaDeviceSynchronize();
    }
    
    std::vector<double> h_b(numFunctions);
    cudaMemcpy(h_b.data(), d_b, numFunctions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_b);
    std::vector<HashingFunct> hashFunctions(numFunctions);
    for (int i = 0; i < numFunctions; i++) {
        hashFunctions[i].a = d_a + i * n;
        hashFunctions[i].b = h_b[i];
        hashFunctions[i].w = w;
    }
    return {hashFunctions, d_a};
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
        std::getline(ss, value, ',');  
        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value));
        }
        dataPoints.push_back(point);
    }
    file.close();
    return dataPoints;
}

std::unordered_map<std::vector<int>, std::unordered_map<std::vector<int>, std::vector<std::vector<double>>>> finalHashCUDA(
    const std::vector<std::vector<double>>& points, const std::vector<HashingFunct>& hashfunctions1,
    const std::vector<HashingFunct>& hashfunctions2, int L1, int L2) {
    std::unordered_map<std::vector<int>, std::unordered_map<std::vector<int>, std::vector<std::vector<double>>>> result;
    for (const auto& point : points) {
        std::vector<int> hash1(L1), hash2(L2);
        for (int i = 0; i < L1; i++) hash1[i] = hashingComputingCUDA(point.data(), hashfunctions1[i], point.size());
        for (int i = 0; i < L2; i++) hash2[i] = hashingComputingCUDA(point.data(), hashfunctions2[i], point.size());
        result[hash1][hash2].push_back(point);
    }
    return result;
}


int main() {
    std::string csvFilePath = "../Data/fingerprints_class.csv";
    int L1 = 3, L2 = 2;
    double w1 = 10.0, w2 = 5.0;

    std::vector<std::vector<double>> points = loadDataFromCSV(csvFilePath);
    int D = points[0].size();

    auto [hashFunctions1, d_a1] = generateLSHParamsOnGPU(D, w1, L1);
    auto [hashFunctions2, d_a2] = generateLSHParamsOnGPU(D, w2, L2);

    auto hashTable = finalHashCUDA(points, hashFunctions1, hashFunctions2, L1, L2);

    // Generate a random point
    std::vector<double> randomPoint(D);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1); // Assuming the range of input data

    for (int i = 0; i < D; i++) {
        randomPoint[i] = dis(gen);
    }

    // Compute hash values for the random point
    std::vector<int> hash1(L1), hash2(L2);
    for (int i = 0; i < L1; i++) hash1[i] = hashingComputingCUDA(randomPoint.data(), hashFunctions1[i], D);
    for (int i = 0; i < L2; i++) hash2[i] = hashingComputingCUDA(randomPoint.data(), hashFunctions2[i], D);

    // Print the hash values of the random point
    std::cout << "Random Point Hash Values:\nL1: ";
    for (int h : hash1) std::cout << h << " ";
    std::cout << "\nL2: ";
    for (int h : hash2) std::cout << h << " ";
    std::cout << "\n";

    // Check for neighbors in the same hash bucket
    if (hashTable.find(hash1) != hashTable.end() && hashTable[hash1].find(hash2) != hashTable[hash1].end()) {
        std::cout << "Neighbors found in the same hash bucket:\n";
        for (const auto& neighbor : hashTable[hash1][hash2]) {
            for (double val : neighbor) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << "No neighbors found in the same hash bucket.\n";
    }

    // Free GPU memory
    cudaFree(d_a1);
    cudaFree(d_a2);

    std::cout << "Hashing terminé !\n";
    return 0;
}

