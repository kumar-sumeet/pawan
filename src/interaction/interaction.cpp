/*! PArticle Wake ANalysis
 * \file interaction.cpp
 * \brief Routines for interactions
 *
 * @author Puneet Singh
 * @date 04/24/2021
 */
#include "interaction.h"
#include "interaction_utils.h"

pawan::__interaction::__interaction(__wake* W) {
    _W.push_back(W);
    _size = _W[0]->_size;
    _nu = 2.0e-2;
    _nWake = 1;
}

pawan::__interaction::__interaction(__wake* W1, __wake* W2) {
    _W.push_back(W1);
    _W.push_back(W2);
    _size = _W[0]->_size + _W[1]->_size;
    _nu = 2.0e-2;
    _nWake = 2;
}

void pawan::__interaction::interact() {
    for (auto& w : _W) {
        gsl_matrix_set_zero(w->_velocity);
        gsl_matrix_set_zero(w->_retvorcity);
    }
    for (auto& w : _W) {
        interact(w);
    }
    for (size_t i = 0; i < _nWake; ++i) {
        for (size_t j = i + 1; j < _nWake; ++j) {
            interact(_W[i], _W[j]);
        }
    }
}

void pawan::__interaction::interact(__wake* W) {
    for (size_t i_src = 0; i_src < W->_numParticles; ++i_src) {
        gsl_vector_const_view r_src = gsl_matrix_const_row(W->_position, i_src);
        gsl_vector_const_view a_src = gsl_matrix_const_row(W->_vorticity, i_src);
        gsl_vector_view dr_src = gsl_matrix_row(W->_velocity, i_src);
        gsl_vector_view da_src = gsl_matrix_row(W->_retvorcity, i_src);
        double s_src = gsl_vector_get(W->_radius, i_src);
        double v_src = gsl_vector_get(W->_volume, i_src);
        for (size_t i_trg = i_src + 1; i_trg < W->_numParticles; ++i_trg) {
            gsl_vector_const_view r_trg = gsl_matrix_const_row(W->_position, i_trg);
            gsl_vector_const_view a_trg = gsl_matrix_const_row(W->_vorticity, i_trg);
            gsl_vector_view dr_trg = gsl_matrix_row(W->_velocity, i_trg);
            gsl_vector_view da_trg = gsl_matrix_row(W->_retvorcity, i_trg);
            double s_trg = gsl_vector_get(W->_radius, i_trg);
            double v_trg = gsl_vector_get(W->_volume, i_trg);
            INTERACT(_nu, s_src, s_trg, &r_src.vector, &r_trg.vector, &a_src.vector, &a_trg.vector, v_src, v_trg, &dr_src.vector, &dr_trg.vector, &da_src.vector, &da_trg.vector);
        }
    }
}

void pawan::__interaction::interact(__wake* W1, __wake* W2) {
    for (size_t i_src = 0; i_src < W1->_numParticles; ++i_src) {
        gsl_vector_const_view r_src = gsl_matrix_const_row(W1->_position, i_src);
        gsl_vector_const_view a_src = gsl_matrix_const_row(W1->_vorticity, i_src);
        gsl_vector_view dr_src = gsl_matrix_row(W1->_velocity, i_src);
        gsl_vector_view da_src = gsl_matrix_row(W1->_retvorcity, i_src);
        double s_src = gsl_vector_get(W1->_radius, i_src);
        double v_src = gsl_vector_get(W1->_volume, i_src);
        for (size_t i_trg = 0; i_trg < W2->_numParticles; ++i_trg) {
            gsl_vector_const_view r_trg = gsl_matrix_const_row(W2->_position, i_trg);
            gsl_vector_const_view a_trg = gsl_matrix_const_row(W2->_vorticity, i_trg);
            gsl_vector_view dr_trg = gsl_matrix_row(W2->_velocity, i_trg);
            gsl_vector_view da_trg = gsl_matrix_row(W2->_retvorcity, i_trg);
            double s_trg = gsl_vector_get(W2->_radius, i_trg);
            double v_trg = gsl_vector_get(W2->_volume, i_trg);
            INTERACT(_nu, s_src, s_trg, &r_src.vector, &r_trg.vector, &a_src.vector, &a_trg.vector, v_src, v_trg, &dr_src.vector, &dr_trg.vector, &da_src.vector, &da_trg.vector);
        }
    }
}

void pawan::__interaction::write(FILE* f) {
    fwrite(&_nWake, sizeof(size_t), 1, f);
    for (auto& w : _W) {
        w->write(f);
    }
}

void pawan::__interaction::setStates(const gsl_vector* state) {
    size_t offset = 0;
    for (auto& w : _W) {
        gsl_vector_const_view st = gsl_vector_const_subvector(state, offset, w->_size);
        w->setStates(&st.vector);
        offset += w->_size;
    }
}

void pawan::__interaction::getRates(gsl_vector* rate) {
    size_t offset = 0;
    for (auto& w : _W) {
        gsl_vector_view rt = gsl_vector_subvector(rate, offset, w->_size);
        w->getRates(&rt.vector);
        offset += w->_size;
    }
}

void pawan::__interaction::getStates(gsl_vector* state) {
    size_t offset = 0;
    for (auto& w : _W) {
        gsl_vector_view st = gsl_vector_subvector(state, offset, w->_size);
        w->getStates(&st.vector);
        offset += w->_size;
    }
}
