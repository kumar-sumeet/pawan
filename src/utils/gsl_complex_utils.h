/*! PArticle Wake ANalysis
 * \file gsl_complex_utils.h
 * \brief GNU Scientific Library Complex Matrix/Vector utilities for PAWAN
 *
 * @author Puneet Singh
 * @date 03/28/2021
 *
 */

#ifndef GSL_COMPLEX_UTILS_H_
#define GSL_COMPLEX_UTILS_H_

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "gsl_utils.h"

/*
 *
 * PRINT OPERATIONS
 *
 */

/*! \fn inline void DISP(std::string s, const gsl_complex &c, std::ostream &os = std::cout)
 * \brief Print string and complex number
 * \param	s	String
 * \param	c	Complex number
 * \param	os	Output stream
 */
inline void DISP(std::string s, const gsl_complex& c, std::ostream& os = std::cout) {
    os << "\t" << s << " = " << GSL_REAL(c) << " +i " << GSL_IMAG(c) << std::endl;
};

/*! \fn inline void DISP(std::string s, const gsl_vector_complex *v, std::ostream &os = std::cout)
 * \brief Print string and long complex array of values
 * \param	s	String
 * \param	v	gsl complex vector
 * \param	os	Output stream
 */
inline void DISP(std::string s, const gsl_vector_complex* v, std::ostream& os = std::cout) {
    if (v->size == 0) {
        os << "\t" << s << " is empty." << std::endl;
    } else {
        os << "\t" << s << " = " << std::endl;
        for (int i = 0; i < v->size; ++i) {
            gsl_complex C = gsl_vector_complex_get(v, i);
            os << "\t(" << GSL_REAL(C) << "," << GSL_IMAG(C) << ")" << std::endl;
        }
    }
};

/*! \fn inline void DISP(std::string s, const gsl_matrix_complex *m, std::ostream &os = std::cout)
 * \brief Print string and matrix of complex values
 * \param	s	String
 * \param	m	gsl matrix
 * \param	os	Output stream
 */
inline void DISP(std::string s, const gsl_matrix_complex* m, std::ostream& os = std::cout) {
    if (m->size1 == 0 && m->size2 == 0) {
        os << "\t" << s << " is empty." << std::endl;
    } else {
        os << "\t" << s << " = " << std::endl;
        for (int i = 0; i < m->size1; ++i) {
            os << "\t";
            for (int j = 0; j < m->size2; ++j) {
                gsl_complex Z = gsl_matrix_complex_get(m, i, j);
                os << "\t(" << GSL_REAL(Z) << "," << GSL_IMAG(Z) << ")";
            }
            os << std::endl;
        }
    }
};

/*
 *
 * VECTOR OPERATIONS
 *
 */

/*! \fn inline void conjugate_gsl_vector(const gsl_vector_complex *V, gsl_vector_complex *A)
 * \brief	Find the conjugate of a gsl vector
 * \param	V	gsl vector
 * \param	A	output vector
 */
inline void conjugate_gsl_vector(const gsl_vector_complex* V, gsl_vector_complex* A) {
    for (int i = 0; i < V->size; i++) {
        gsl_complex Z = gsl_complex_conjugate(gsl_vector_complex_get(V, i));
        gsl_vector_complex_set(A, i, Z);
    }
};

/*! \fn inline void complex_vector_product(const gsl_complex &C, const double D[3], gsl_vector_complex *A)
 * \brief Complex number multiplied with Double vector
 * \param C complex number
 * \param D double vector
 * \param A output complex vector
 * Returns Z = complex vector
 */
inline void complex_vector_product(const gsl_complex& C, const double D[3], gsl_vector_complex* A) {
    for (int i = 0; i < 3; ++i) {
        gsl_vector_complex_set(A, i, gsl_complex_mul_real(C, D[i]));
    }
}

/*! \fn inline void flip_sign(gsl_vector_complex *A)
 * \brief Flip sign of complex number vector
 * \param A complex number vector
 */
inline void flip_sign(gsl_vector_complex* A) {
    for (int i = 0; i < A->size; ++i) {
        gsl_vector_complex_set(A, i, gsl_complex_negative(gsl_vector_complex_get(A, i)));
    }
}

/*! \fn inline void gsl_vector_complex_mul_real(const double &r, gsl_vector_complex *A)
 * \brief Multiply complex vector with real number
 * \param r real number
 * \param A complex number vector
 */
inline void gsl_vector_complex_mul_real(const double& r, gsl_vector_complex* A) {
    for (int i = 0; i < A->size; ++i) {
        gsl_vector_complex_set(A, i, gsl_complex_mul_real(gsl_vector_complex_get(A, i), r));
    }
}

/*! \fn inline void gsl_vector_complex_div_real(const double &r, gsl_vector_complex *A)
 * \brief Multiply complex vector with real number
 * \param r real number
 * \param A complex number vector
 */
inline void gsl_vector_complex_div_real(const double& r, gsl_vector_complex* A) {
    for (int i = 0; i < A->size; ++i) {
        gsl_vector_complex_set(A, i, gsl_complex_div_real(gsl_vector_complex_get(A, i), r));
    }
}

/*! \fn inline void gsl_real_to_complex_vector(gsl_vector *real, gsl_vector_complex *comp)
 * \brief Create complex vector from real vector
 * \param real Real number vector
 * \param comp Complex number vector
 */
inline void gsl_real_to_complex_vector(gsl_vector* real, gsl_vector_complex* comp) {
    for (int i = 0; i < real->size; ++i) {
        gsl_vector_complex_set(comp, i, gsl_complex_rect(gsl_vector_get(real, i), 0.0));
    }
}

/*! \fn inline void gsl_conjugate_vector(gsl_vector_complex *A, gsl_vector_complex *B)
 * \brief Conjugate complex vector
 * \param A conjugate complex number vector
 * \param B original complex number vector
 */
inline void gsl_conjugate_vector(gsl_vector_complex* A, gsl_vector_complex* B) {
    for (int i = 0; i < A->size; ++i) {
        gsl_vector_complex_set(A, i, gsl_complex_conjugate(gsl_vector_complex_get(B, i)));
    }
}

#endif
