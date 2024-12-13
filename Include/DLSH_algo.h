#ifndef DLSH_ALGO_H
#define DLSH_ALGO_H

#include <vector>
#include <string>
#include "/home/hp/DLSH/Include/hashing.h"

// Classe pour implémenter l'algorithme DLSH
class DLSH {
public:
    /* Classe qui définit l'implémentation de l'algo :
    csvFilePath : Chemin vers le fichier CSV qui contient les fingerprints.
    w : paramètre de l'objet choisit par l'utilisateur
    L : un entier qui represente le nombre de table de hachage utilisé
    n : dimensions des embeddings
    std::vector<HashingFunct> hashFunctions : l'ensemble des fct de hachage
    dataPoints : Données des fingerprints

 */
    DLSH(const std::string& csvFilePath, int L, int n, int w);

    // Méthode pour exécuter l'algorithme DLSH au niveau 1
    std::vector<std::vector<int>> computeHashTable_niv1();

    // Méthode pour exécuter l'algorithme DLSH au niveau 2
    std::vector<std::vector<int>> computeHashTable_niv2(std::vector<std::vector<int>>& hashTable_niv1);

    // Méthode pour charger les données depuis un fichier CSV
    void loadDataFromCSV();

private:
    std::string csvFilePath;
    int L;
    int n;
    int w;
    std::vector<HashingFunct> hashFunctions;
    std::vector<std::vector<double>> dataPoints;
};

#endif // DLSH_ALGO_H
