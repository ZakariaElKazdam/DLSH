cmake_minimum_required(VERSION 3.10)
project(DLSH)

# Définit le standard C++ à utiliser
set(CMAKE_CXX_STANDARD 17)

# Chemins des fichiers d'en-tête
include_directories(${CMAKE_SOURCE_DIR}/Include)
include_directories(${CMAKE_SOURCE_DIR}/cnpy)

# Chemins des fichiers source
set(SOURCE_FILES
        src/DLSH_algo.cpp
        src/hashing.cpp
)

# Ajoute un exécutable
add_executable(dlsh_algo ${SOURCE_FILES})

# Lier les bibliothèques nécessaires
# Ajoutez d'autres bibliothèques ici si nécessaires
