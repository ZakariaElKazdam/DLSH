//
// Created by hp on 23/12/24.
//

#include "../Include/NewPoint.h"
#include "../Include/DLSH_algo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <iomanip>

// Constructor
NewPoint::NewPoint(const std::map<std::vector<int>,
                                  std::map<std::vector<int>,
                                           std::set<std::vector<double>, VectorComparator<double>>,
                                           VectorComparator<int>>,
                                  VectorComparator<int>>& dataDict,
                   const std::vector<HashingFunct>& HashFunctions1,
                   const std::vector<HashingFunct>& HashFunctions2)
    : DataDict(dataDict), hashFunctions1(HashFunctions1), hashFunctions2(HashFunctions2) {}

// Method to get the set of similar points without inserting the point
std::set<std::vector<double>, VectorComparator<double>> NewPoint::GetSet(const std::vector<double>& point) {
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

    // Retrieve the set of similar points without inserting the new point
    auto& secondaryMap = DataDict[hash1];
    auto& pointSet = secondaryMap[hash2];

    // Return the set of similar points
    return pointSet;
}

// Method to insert the point after retrieving the similar points
void NewPoint::InsertPoint(const std::vector<double>& point) {
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
}


std::vector<double> generateRandomPoint(int n, double min, double max) {
    std::vector<double> point;
    std::random_device rd; // Seed for random number engine
    std::mt19937 gen(rd()); // Standard random number generator
    std::uniform_real_distribution<> dis(min, max); // Uniform distribution in range [min, max]

    for (int i = 0; i < n; ++i) {
        point.push_back(dis(gen));
    }

    return point;
}



/*
// Fonction pour tester la classe NewPoint
int main() {
    try {
        // Paramètres pour l'algorithme DLSH
        std::string csvFilePath = "../Data/fingerprints_class.csv";
        int L1 = 3;   // Nombre de fonctions de hachage niveau 1
        int L2 = 2;   // Nombre de fonctions de hachage niveau 2
        int n = 0;    // Dimension des vecteurs
        double w1 = 10.0; // Largeur des bins niveau 1
        double w2 = 5.0;  // Largeur des bins niveau 2

        // Charger les données CSV pour déterminer la dimension
        std::ifstream file(csvFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Erreur : impossible d'ouvrir le fichier CSV.");
        }

        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                ++n;
            }
            --n; // Exclure le label
        }
        file.close();

        // Initialiser DLSH
        DLSH dlsh(csvFilePath, L1, L2, n, w1, w2);
        dlsh.loadDataFromCSV();

        // Générer la table de hachage initiale
        auto hashTable = dlsh.computeHashTable();

        // Initialiser NewPoint avec la table de hachage et les fonctions de hachage
        NewPoint newPoint(hashTable, dlsh.getHashFunctions1(), dlsh.getHashFunctions2());

        // Générer un nouveau point aléatoire
        std::vector<double> randomPoint = generateRandomPoint(n, -10.0, 10.0);

        // Récupérer les points similaires sans insérer le nouveau point
        auto similarPoints = newPoint.GetSet(randomPoint);

        // Afficher les points similaires avant l'insertion
        std::cout << "Nouveau point généré : ";
        for (const auto& val : randomPoint) {
            std::cout << val << " ";
        }
        std::cout << "\nPoints similaires avant insertion :\n";
        if (similarPoints.empty()) {
            std::cout << "Aucun point similaire trouvé.\n";
        } else {
            for (const auto& point : similarPoints) {
                for (const auto& val : point) {
                    std::cout << val << " ";
                }
                std::cout << "\n";
            }
        }

        // Insérer le nouveau point dans la structure
        newPoint.InsertPoint(randomPoint);

        // Confirmation de l'insertion
        std::cout << "Le nouveau point a été inséré dans la structure.\n";

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
*/

int main() {
    try {
        // Paramètres pour l'algorithme DLSH
        std::string csvFilePath = "../Data/fingerprints_class.csv";
        int L1 = 3;   // Nombre de fonctions de hachage niveau 1
        int L2 = 2;   // Nombre de fonctions de hachage niveau 2
        int n = 0;    // Dimension des vecteurs
        double w1 = 10.0; // Largeur des bins niveau 1
        double w2 = 5.0;  // Largeur des bins niveau 2

        // Charger les données CSV pour déterminer la dimension
        std::ifstream file(csvFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Erreur : impossible d'ouvrir le fichier CSV.");
        }

        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                ++n;
            }
            --n; // Exclure le label
        }
        file.close();

        // Initialiser DLSH
        DLSH dlsh(csvFilePath, L1, L2, n, w1, w2);
        dlsh.loadDataFromCSV();

        // Générer la table de hachage initiale
        auto hashTable = dlsh.computeHashTable();

        // Initialiser NewPoint avec la table de hachage et les fonctions de hachage
        NewPoint newPoint(hashTable, dlsh.getHashFunctions1(), dlsh.getHashFunctions2());

        // Créer un point connu et l'insérer
        std::vector<double> knownPoint = generateRandomPoint(n, -10.0, 10.0);
        newPoint.InsertPoint(knownPoint);

        // Générer un point légèrement modifié (voisin)
        std::vector<double> similarPoint = knownPoint;
        if (!similarPoint.empty()) {
            similarPoint[0] += 0.1; // Ajouter une petite valeur à la première dimension
        }

        // Récupérer les points similaires avant l'insertion du point modifié
        auto similarPoints = newPoint.GetSet(similarPoint);

        // Afficher les points similaires
        std::cout << "Point connu inséré : ";
        for (const auto& val : knownPoint) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << "\nPoint similaire généré : ";
        for (const auto& val : similarPoint) {
            std::cout << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << "\nPoints similaires trouvés :\n";
        if (similarPoints.empty()) {
            std::cout << "Aucun point similaire trouvé.\n";
        } else {
            for (const auto& point : similarPoints) {
                for (const auto& val : point) {
                    std::cout << std::fixed << std::setprecision(2) << val << " ";
                }
                std::cout << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}