#ifndef HASHING_H
#define HASHING_H

#include <vector>
#include <map>
#include <set>

struct HashingFunct {
    /*Chaque fonction de hachage est definie par 3 paramètres :
  a : un vecteur de la même taille que les fingerprints ou embeddings des données ( vecteur directeur qui définit l'orientation du plan)
  b : bias générée aléatoirement de manière uniforme dans [0,w], ajuste la position du plan dans l'éspace
  w :  largeur de chaque bin (choisie par l'utilisateur)

  je pense que je doit calculer la probabilité de collision après #TODO */

    std::vector<double> a;
    double b;
    double w;
};

template <typename T>
struct VectorComparator {
    bool operator()(const std::vector<T>& v1, const std::vector<T>& v2) const ;
};

template <typename T>
void printVector(const std::vector<T>& vec);

void printDictionary(const std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> >& dict) ;

HashingFunct generateLSHParameters(int n, double w = 1);

double hashingComputing(std::vector<double> point, HashingFunct h);


std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > finalHash(
            std::vector<std::vector<double>>& points,
            std::vector<HashingFunct>& hashfunctions,
            int& L,
            int& n
    );



#endif //HASHING_H