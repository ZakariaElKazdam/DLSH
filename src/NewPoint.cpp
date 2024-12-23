//
// Created by hp on 23/12/24.
//

#include "../Include/NewPoint.h"

// Constructor
NewPoint::NewPoint(const std::map<std::vector<int>,
                                  std::map<std::vector<int>,
                                           std::set<std::vector<double>, VectorComparator<double>>,
                                           VectorComparator<int>>,
                                  VectorComparator<int>>& dataDict,
                   const std::vector<HashingFunct>& HashFunctions1,
                   const std::vector<HashingFunct>& HashFunctions2)
    : DataDict(dataDict), hashFunctions1(HashFunctions1), hashFunctions2(HashFunctions2) {}

// Method to place a new point in the hash table and return the set of similar points
std::set<std::vector<double>, VectorComparator<double>> NewPoint::GetSet(const std::vector<double>& point) {
  std::vector<int> primaryKey;
  std::vector<int> secondaryKey;

  size_t L1 = hashFunctions1.size();
  size_t L2 = hashFunctions2.size();

  std::vector<int> hash1(L1);
  std::vector<int> hash2(L2);

  // Compute hash keys for level 1
  for (size_t j = 0; j < L1; j++) {
    hash1[j] = hashingComputing(point, hashFunctions1[j]);
  }

  // Compute hash keys for level 2
  for (size_t j = 0; j < L2; j++) {
    hash2[j] = hashingComputing(point, hashFunctions2[j]);
  }

  // Insert the point into the DataDict
  auto& secondaryMap = DataDict[hash1];
  auto& pointSet = secondaryMap[hash2];
  pointSet.insert(point);

  // Return the set of similar points
  return pointSet;
}
