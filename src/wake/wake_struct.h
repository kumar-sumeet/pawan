#include <stdio.h>

inline void allocate2DArray(double**& array, const int& row, const int& col, const double& init) {
   array = new double*[row];
   for (size_t i = 0; i < row; i++) {
         array[i] = new double[col];
         for (size_t j = 0; j < col; j++) {
            array[i][j] = init;
         }
   }
}

inline void allocate1DArray(double*& array, const int& len, const double& init) {
   array = new double[len];
   for (size_t i = 0; i < len; i++) {
      array[i] = init;
   }
}

inline void delete2DArray(double**& array, const int& row) {
   for (size_t i = 0; i < row; i++) {
      delete [] array[i];
   }
   delete [] array;
}

namespace pawan{
   struct wake_struct {
   size_t size;
   size_t numParticles;
   size_t numDimensions = 3;
   double** position;
   double** velocity;
   double** vorticity;
   double** retvorcity;
   double*  radius;
   double*  volumn;
   double*  birthstrength;

   wake_struct(const int& n) {
      numDimensions = 3;
      numParticles = n;
      size = 2 * numParticles * numDimensions;
      allocate2DArray(position, numParticles, numDimensions, 0.0);
      allocate2DArray(velocity, numParticles, numDimensions, 0.0);
      allocate2DArray(vorticity, numParticles, numDimensions, 0.0);
      allocate2DArray(retvorcity, numParticles, numDimensions, 0.0);
      
      allocate1DArray(radius, numParticles, 1.0);
      allocate1DArray(volumn, numParticles, 1.0);
      allocate1DArray(birthstrength, numParticles, 1.0);
   }

   ~wake_struct() {
      delete2DArray(position, numParticles);
      delete2DArray(velocity, numParticles);
      delete2DArray(vorticity, numParticles);
      delete2DArray(retvorcity, numParticles);

      delete [] radius;
      delete [] volumn;
      delete [] birthstrength;
   }

   void setStates(const double *state) {
      size_t np = size/2/numDimensions;
      size_t matrixsize = np * numDimensions;

      // gsl_matrix_const_view_vector
      for (size_t i = 0; i < np; i++) {
         for (size_t j = 0; j < numDimensions; j++) {
            position[i][j] = state[i * numDimensions + j];
         }
      }
      
      // gsl_vector_const_subvector + gsl_matrix_const_view_vector
      for (size_t i = 0; i < np; i++) {
         for (size_t j = 0; j < numDimensions; j++) {
            vorticity[i][j] = state[i * numDimensions + j + matrixsize];
         }
      }
   }

   void getStates(double *state) {
      for (size_t i = 0; i < numParticles; i++) {
         for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            state[ind] = position[i][j];
            ind += size/2;
            state[ind] = vorticity[i][j];
         }
      }
   }

   void getRates(double *rate) {
      for (size_t i = 0; i < numParticles; i++) {
         for (size_t j = 0; j < numDimensions; j++) {
            size_t ind = i * numDimensions + j;
            rate[ind] = velocity[i][j];
            ind += size/2;
            rate[ind] = retvorcity[i][j];
         }
      }
   }
};
}
