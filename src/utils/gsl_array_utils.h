#include "gsl_utils.h"
#include <gsl/gsl_matrix.h>

inline void matrixTo2DArray(gsl_matrix* matrix, double**& array) {
   for (size_t i = 0; i < matrix->size1; i++) {
      for (size_t j = 0; j < matrix->size2; j++) {
         array[i][j] = gsl_matrix_get(matrix, i, j);
      }
   }
}

inline void vectorToArray(gsl_vector* vector, double*& array) {
   for (size_t i = 0; i < vector->size; i++) {
      array[i] = gsl_vector_get(vector, i);
   }
}

inline void delete2DArray(double**& array, int row) {
   for (size_t i = 0; i < row; i++) {
      delete [] array[i];
   }
   delete [] array;
}