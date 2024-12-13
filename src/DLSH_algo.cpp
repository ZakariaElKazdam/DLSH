#include "/home/hp/DLSH/Include/DLSH_algo.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>

// Constructeur
DLSH::DLSH(const std::string& csvFilePath, int L, int n, int w)
    : csvFilePath(csvFilePath), L(L), n(n), w(w), hashFunctions(L) {
    for (int i = 0; i < L; ++i) {
        hashFunctions.push_back(generateLSHParameters(n, w));
    }
}

// Méthode pour charger les données depuis un fichier CSV
void DLSH::loadDataFromCSV() {
    std::ifstream file(csvFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Erreur : impossible d'ouvrir le fichier CSV");
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
std::vector<std::vector<int>> DLSH::computeHashTable_niv1() {
    if (dataPoints.empty()) {
        throw std::runtime_error("Erreur : les données sont vides. Chargez les données depuis le fichier CSV.");
    }

    return finalHash(dataPoints, hashFunctions, L, n);
}



int main() {
    try {
        // Chemin vers le fichier CSV généré
        std::string csvFilePath = "/home/hp/DLSH/Data/fingerprints_class.csv";

        // Paramètres pour l'algorithme DLSH
        int L = 3;  // Nombre de fcts de hachage
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
        int w = 1;  // Largeur des bins

        // Initialiser l'algorithme DLSH
        DLSH dlsh(csvFilePath, L, n, w);

        // Charger les données depuis le fichier CSV
        dlsh.loadDataFromCSV();

        // Calculer les tables de hachage niveau 1
        std::vector<std::vector<int>> hashTable_niv1 = dlsh.computeHashTable_niv1();

        // Afficher les résultats
        std::cout << "Résultats de la table de hachage (niveau 1) :\n";
        for (size_t i = 0; i < hashTable_niv1.size(); ++i) {
            std::cout << "Point " << i << ": ";
            for (size_t j = 0; j < hashTable_niv1[i].size(); ++j) {
                std::cout << hashTable_niv1[i][j] << " ";
            }
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



