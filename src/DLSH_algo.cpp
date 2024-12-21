#include "../Include/DLSH_algo.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Constructeur
DLSH::DLSH(const std::string& csvFilePath, int L1, int L2,  int n, double w1, double w2)
    : csvFilePath(csvFilePath), L1(L1), L2(L2), n(n), w1(w1), w2(w2) {
    for (int i = 0; i < L1; i++) {
        hashFunctions1.push_back(generateLSHParameters(n, w1));
    }
    for (int i = 0; i < L2; i++) {
        hashFunctions2.push_back(generateLSHParameters(n, w2));
    }
}

// Méthode pour charger les données depuis un fichier CSV
void DLSH::loadDataFromCSV() {
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Erreur1 : impossible d'ouvrir le fichier CSV");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> point;

        // Ignorer le premier champ (label de la classe)
        std::getline(ss, value, ',');

        // Lire les dimensions des points
        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value));
        }

        // Ajouter le point extrait à la liste des données
        dataPoints.push_back(point);
    }

    file.close();
}


// Méthode pour exécuter l'algorithme DLSH
std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > DLSH::computeHashTable_niv1() {
    if (dataPoints.empty()) {
        throw std::runtime_error("Erreur : les données sont vides. Chargez les données depuis le fichier CSV.");
    }

    return finalHash(dataPoints, hashFunctions1, L1, n);
}

std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > DLSH::computeHashTable_niv2(
        std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> >& hashTable_niv1){

    std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > finalResult;

    for (const auto& [key, valueSet] : hashTable_niv1) {
        // si le valueSet ne contient qu'un seul element , pas besoin de le decortiquer d'avance :)
        if (valueSet.size() == 1 ){
            // extract le seul point
            std::vector<double> point = *valueSet.begin();
            std::vector<int> cle (L2);

            for( int j =0 ; j<L2 ; j++){
                cle[j]= hashingComputing (point , hashFunctions2[j] );
            }
            finalResult[cle].insert(point);
        }
        else{
            for (const auto& point : valueSet){
                //pour chaque point , on calcule la nouvelle signature en utilisant les fonctions de hashage niveau 2
                std::vector<int> cle(L2);
                for( int j =0 ; j<L2 ; j++){
                    cle[j]= hashingComputing (point , hashFunctions2[j]);
                }
                finalResult[cle].insert(point);
            }
        }
    }
    return finalResult;
}



int main() {
    try {
        // Chemin vers le fichier CSV généré
        std::string csvFilePath = "../Data/fingerprints_class.csv";

        // Paramètres pour l'algorithme DLSH
        int L1 = 3;  // Nombre de fcts de hachage niveau 1
        int L2 = 6; // nombre de fonction de hachage niveau 2
        // Déduire la dimension des vecteurs (n) directement à partir du fichier CSV
        int n = 0; // Initialiser n avec 0 pour déterminer dynamiquement

        // Ouvrir le fichier CSV pour compter les colonnes
        std::ifstream file(csvFilePath);
        if (!file.is_open()) {
            throw std::runtime_error("Erreur : impossible d'ouvrir le fichier CSV.");
        }

        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;

            // Compter les colonnes en sautant la première colonne (label)
            while (std::getline(ss, value, ',')) {
                ++n;
            }
            --n; // Exclure la première colonne (le label)
        }
        file.close();
        double w1 = 5;  // Largeur des bins
        double w2 = 0.5 ; // largeur des bins mais pour niveau 2

        // Initialiser l'algorithme DLSH
        DLSH dlsh(csvFilePath, L1, L2, n, w1, w2);

        // Charger les données depuis le fichier CSV
        dlsh.loadDataFromCSV();

        // Calculer les tables de hachage niveau 1
        std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > hashTable_niv1 = dlsh.computeHashTable_niv1();
        std::map<std::vector<int>, std::set<std::vector<double>, VectorComparator<double> >, VectorComparator<int> > hashTable_niv2 = dlsh.computeHashTable_niv2(hashTable_niv1);

        // Afficher les résultats
        std::cout << "Résultats de la table de hachage (niveau 1) :\n";
        printDictionary(hashTable_niv1);


    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



