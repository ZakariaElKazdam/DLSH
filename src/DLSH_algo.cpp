#include "../Include/DLSH_algo.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Constructeur
DLSH::DLSH(  std::string& csvFilePath, int L1, int L2,  int n, double w1, double w2)
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
std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double> , VectorComparator<double> >, VectorComparator<int> >, VectorComparator<int> > DLSH::computeHashTable() {
    if (dataPoints.empty()) {
        throw std::runtime_error("Erreur : les données sont vides. Chargez les données depuis le fichier CSV.");
    }

    return finalHash(dataPoints, hashFunctions1, hashFunctions2, L1, L2);
}


int main() {
    try {
        // Chemin vers le fichier CSV généré
        std::string csvFilePath = "../Data/fingerprints_class.csv";

        // Paramètres pour l'algorithme DLSH
        int L1 = 3;  // Nombre de fcts de hachage niveau 1
        int L2 = 1; // nombre de fonction de hachage niveau 2
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
        double w1 = 10;  // Largeur des bins
        double w2 = 5 ; // largeur des bins mais pour niveau 2

        // Initialiser l'algorithme DLSH
        DLSH dlsh(csvFilePath, L1, L2, n, w1, w2);

        // Charger les données depuis le fichier CSV
        dlsh.loadDataFromCSV();

        // Calculer les tables de hachage niveau 1
        std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double> , VectorComparator<double> >, VectorComparator<int> >, VectorComparator<int> > hashTable = dlsh.computeHashTable();

        // Afficher les résultats
        std::cout << "Résultats de la table de hachage (niveau 1) :\n";
        printDictionary(hashTable);


    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



