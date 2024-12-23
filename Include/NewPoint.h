//
// Created by hp on 23/12/24.
//

#ifndef NEWPOINT_H
#define NEWPOINT_H

#include <vector>
#include <string>
#include <map>
#include <set>
#include "../Include/hashing.h"

class NewPoint {
private:
    // Data dictionary for hash storage
    std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double>>, VectorComparator<int>>, VectorComparator<int>> DataDict;

    // Hash functions for both levels
    std::vector<HashingFunct> hashFunctions1;
    std::vector<HashingFunct> hashFunctions2;

public:
    // Constructor
    NewPoint(const std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double>>, VectorComparator<int>>, VectorComparator<int>>& dataDict,
             const std::vector<HashingFunct>& HashFunctions1,
             const std::vector<HashingFunct>& HashFunctions2);

    // Method to place a new point in the hash table and return the set of similar points
    std::set<std::vector<double>, VectorComparator<double>> GetSet(const std::vector<double>& point);
};

#endif //NEWPOINT_H
