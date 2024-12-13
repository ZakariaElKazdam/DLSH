
#include "/home/hp/DLSH/Include/hashing.h"
#include <random>
#include <iostream>




HashingFunct generateLSHParameters(int n, double w ) {
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

double hashingComputing ( std::vector<double> point , HashingFunct h) {
    /*Cett fonction donne la valeur de la projection d'un point en utilisant un fonction de hachage

    Params : point  vecteur qui represente la fingerprint d'une donnée
             h une fonction de hachage
             
    Returns : un entier qui reprsente la position du point par rapport à l'hyperplan (nombre de pas de size w) */
    double result = 0;
    for (int i = 0 ; i < point.size() ; i++){
        result += point[i] * h.a[i];
    }
    result = (result+h.b)/h.w;
    return int(result);
}

std::vector< std::vector<int> > finalHash (std::vector< std::vector<double>> & points ,std::vector<HashingFunct>& hashfunctions , int& L , int& n ){
    /* Calcule le vecteur final qui concatene les resultats des fonction de hachage 

    Params : points  c'est un vecteur de  points  (embeddings des images, videos , texte ...  concernée)
             L  un entier qui represente le nombre de table de hachage utilisé
             n dimensions des embeddings
             hashfunctions vecteur des tables de hachage

    Return hachage final de tous les points 
    */
    std::vector<std::vector<int>> result(n, std::vector<int>(L));
    for(int i =0 ; i<n ; i++){
        for( int j =0 ; j<L ; j++){
            result[i][j]= hashingComputing (points[i] , hashfunctions[j] );
        }
    }
    return result;
}

/*
int main(){
    // Exemple : Générer des paramètres pour un vecteur de dimension 5
    int L = 4;
    int n = 2;
    std::vector<HashingFunct> hashfunctions(L);
    for(int i = 0 ; i<L ; i++){
        hashfunctions[i] = generateLSHParameters(n);
    }

    std::vector< std::vector<double> > x = { {1.485, -1} , {1.7, -1,5}};
    

    std::vector< std::vector<int> > resultat = finalHash (x , hashfunctions , L ,n);
    std::cout<<"les valeurs ddu hashage final sont:\n";
    for(int i =0 ; i<n ; i++){
        for( int j =0 ; j<L ; j++){
            std::cout<<resultat[i][j]<<"   ";
        }
        std::cout<<"\n";
    }
    return 0;
}
*/