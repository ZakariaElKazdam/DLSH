
#include "cnpy/cnpy.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>

struct Fingerprint {
    int classLabel;
    std::vector<float> data;
};

// Function to shuffle fingerprints
void shuffleFingerprints(std::vector<Fingerprint>& fingerprints) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(fingerprints.begin(), fingerprints.end(), g);
}
/*
int main() {

    // Paths to .npy files
    std::vector<std::string> filePaths = {
        "/home/ensimag/3A/DLSH/Data/n01440764.npy",
        "/home/ensimag/3A/DLSH/Data/n01443537.npy",
        "/home/ensimag/3A/DLSH/Data/n01484850.npy",
        "/home/ensimag/3A/DLSH/Data/01491361.npy"
    };

    //container for all fingerprints that we will transform into the csv file
    std::vector<Fingerprint> allFingerprints;


    //Loop for each class to load the fingerprints
    for(size_t classIdx = 0; classIdx < 4; ++classIdx) {
        const std::string fileName = filePaths[classIdx];

        //load  .npy file
        cnpy::NpyArray arr = cnpy::npy_load(fileName);
        float* rawData = arr.data<float>();
        size_t numFingerprints = arr.shape[0];
        size_t fingerprintSize = arr.shape[1];

        //Limit to 1000 fingerprints or less
        size_t numToSelect = std::min(numFingerprints, static_cast<size_t>(1000));

        //Extract fingerprints and assign  class labels
        for (size_t i = 0; i < numToSelect; ++i) {
          std::vector<float> fingerprint(rawData + i * fingerprintSize, rawData + (i + 1) * fingerprintSize);
          allFingerprints.push_back({static_cast<int>(classIdx + 1), fingerprint});
        }

    }

    // Shuffle fingerprints
    shuffleFingerprints(allFingerprints);

    // Write to a CSV file
    std::ofstream outFile("/home/ensimag/3A/DLSH/Data/fingerprints_class.csv");
    for (const auto& fp : allFingerprints) {
        outFile << fp.classLabel << ",";
        for (size_t i = 0; i < fp.data.size(); ++i) {
            outFile << fp.data[i];
            if (i < fp.data.size() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }
    outFile.close();

    return 0;
} */