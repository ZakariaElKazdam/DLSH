#ifndef DLSH_ALGO_H
#define DLSH_ALGO_H

#include <vector>
#include <string>
#include "/home/ensimag/3A/DLSH/Include/hashing.h"

// Classe pour implémenter l'algorithme DLSH
class DLSH {
public:
    /* Classe qui définit l'implémentation de l'algo :
    csvFilePath : Chemin vers le fichier CSV qui contient les fingerprints.
    w1 : paramètre de l'objet choisit par l'utilisateur
    w2 : paramètre de l'objet choisit par l'utilisateur
    L1 : un entier qui represente le nombre de table de hachage utilisé
    L2 : un entier qui represente le nombre de table de hachage utilisé dans le niveau 2
    n : dimensions des embeddings
    std::vector<HashingFunct> hashFunctions1 : l'ensemble des fct de hachage dans le niveau 1
    std::vector<HashingFunct> hashFunctions2 : l'ensemble des fct de hachage dans le niveau 1

    dataPoints : Données des fingerprints

 */
    DLSH(const std::string& csvFilePath, int L1, int L2,  int n, double w1, double w2);

    // Méthode pour exécuter l'algorithme DLSH au niveau 1
    std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > computeHashTable_niv1();

    // Méthode pour exécuter l'algorithme DLSH au niveau 2
    std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > computeHashTable_niv2(
            std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> >& hashTable_niv1);

    // Méthode pour charger les données depuis un fichier CSV
    void loadDataFromCSV();

private:
    std::string csvFilePath;
    int L1;
    int L2;
    int n;
    double w1;
    double w2;
    std::vector<HashingFunct> hashFunctions1;
    std::vector<HashingFunct> hashFunctions2;
    std::vector<std::vector<double>> dataPoints;
};

#endif // DLSH_ALGO_H
