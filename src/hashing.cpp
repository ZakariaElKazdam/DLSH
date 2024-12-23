#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <set>

// Comparateur personnalisé pour std::vector
template <typename T>
struct VectorComparator {
    bool operator()(const std::vector<T>& v1, const std::vector<T>& v2) const {
        return v1 < v2 ;
    }
};

template <typename T>
void printVector(const std::vector<T>& vec) {
/* fonction qui print un vecteur peut importe dses élement int ou double */

    std::cout << "[";
    for (int i = 0; i < vec.size(); i++) {
        std::cout << vec[i] <<"     ";
    }
    std::cout << "] ";
}

void printDictionary(const std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double> , VectorComparator<double> >, VectorComparator<int> >, VectorComparator<int> >& dict) {
    /*fonction qui print un dictionnaire de manière clé valeur */
    for (const auto& [key, valueDict] : dict) {
        std::cout << "Clé niv 1 : ";
        printVector<int>(key);
        std::cout << "\nValeur : \n";

        for (const auto& [key2 , valueSet] : valueDict) {
            std::cout << "-----> Sous dict  : ";
            printVector<int>(key2);
            std::cout << "\n les points sont  : " << valueSet.size() <<"\n";

            /*

            for (const auto & point : valueSet){
                printVector(point);
                std::cout << "\n";
            }

             */
        }
        std::cout << "--------------------\n";
    }
}

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

HashingFunct generateLSHParameters(int n, double w = 1) {
    /*Cette fonction génére une fonction de hachage en tant qu'un objet de la class HashingFunct

    Params : n dimension des fingerprints des données
             w paramètre de l'objet choisit par l'utilisateur

    Return : un objet HashingFunc representant la fonction de hachage*/

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0.0, 1.0); // Distribution normale pour les valeurs du vecteur directeur a ,
    //centré reduite pour réduire les couts des calculs

    std::vector<double> a(n);
    for (int i = 0; i < n; ++i) {
        a[i] = normal_dist(gen);
    }

    std::uniform_real_distribution<> b_dist(0.0, w); // distribution uniforme de b
    double b = b_dist(gen);

    return {a, b, w};
}

int hashingComputing ( std::vector<double> point , HashingFunct h) {
    /*Cett fonction donne la valeur de la projection d'un point en utilisant un fonction de hachage

    Params : point  vecteur qui represente la fingerprint d'une donnée
             h une fonction de hachage

    Returns : un entier qui reprsente la position du point par rapport à l'hyperplan (nombre de pas de size w) */
    double result = 0;
    for (int i = 0 ; i < point.size() ; i++){
        result += point[i] * h.a[i];
    }
    result = (result+h.b)/h.w;
    return static_cast<int>(result);
}

std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double> , VectorComparator<double> >, VectorComparator<int> >, VectorComparator<int> >   finalHash (
        std::vector< std::vector<double>> & points ,
        std::vector<HashingFunct>& hashfunctions1 ,
        std::vector<HashingFunct>& hashfunctions2 ,
        int& L1 ,
        int& L2)
        {
    /* Calcule le vecteur final qui concatene les resultats des fonction de hachage

    Params : points  c'est un vecteur de  points  (embeddings des images, videos , texte ...  concernée)
             L1  un entier qui represente le nombre de table de hachage utilisé
             L2 un entier qui represente le nombre de table de hachage utilisé dans le second niveau
             hashfunctions1 vecteur des tables de hachage pour le premier niveau
             hashfunctions2 vecteur des tables de hachage pour le second niveau


    Return hachage final de tous les points
    */

    std::map<std::vector<int>, std::map<std::vector<int>, std::set<std::vector<double> , VectorComparator<double> >, VectorComparator<int> >, VectorComparator<int> > result;

    size_t num_points = points.size();
    for(int i =0 ; i<num_points ; i++){
        std::vector<int> hash1(L1);
        std::vector<int> hash2(L2);

        for( int j =0 ; j<L1 ; j++){
            hash1[j]= hashingComputing (points[i] , hashfunctions1[j] );
        }
        for( int j =0 ; j<L2 ; j++){
            hash2[j]= hashingComputing (points[i] , hashfunctions2[j] );
        }


        result[hash1][hash2].insert(points[i]);
    }
    return result;
}
