/*! PArticle Wake ANalysis
 * \file resolve.cpp
 * \brief Routines for Resolve class 
 *
 * @author Puneet Singh
 * @date 09/12/2021
 */
#include "resolve.h"
#include <chrono>
#include <thread>

void pawan::__resolve::rebuild(__interaction *IN, __io *IO){
    DOUT("----------------in pawan::__resolve::rebuild()");
	gsl_vector *states = gsl_vector_calloc(IN->_size);
	gsl_matrix *influence = gsl_matrix_calloc(IN->_size,IN->_size);
	FILE *f = IO->create_binary_file(".wakeinfluence");
	double t = 0.0;
	fwrite(&t,sizeof(double),1,f);	
	IN->write(f);
	IN->getStates(states);
	double tStart = TIME();
	OUT("Calculating influence matrix of all particles ");
	//printf("IN->_size = %d",IN->_size);
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    for(size_t i = 0; i<IN->_size; ++i){
		gsl_vector_set(states,i,1.0);
		IN->setStates(states);
		IN->resolve();
		gsl_vector_view column = gsl_matrix_column(influence,i);
		IN->getRates(&column.vector);
		OUT("\ti",i);
		gsl_vector_set(states,i,0.0);
		t += 1.0;
		fwrite(&t,sizeof(double),1,f);
		IN->write(f);
	}
	fclose(f);
	//OUT("influence",influence);
	OUT("Calculating expected vorticity field.");
	gsl_vector *vrxfield = gsl_vector_calloc(IN->_size);
	IN->getIdealRates(vrxfield);
	//OUT("vrxfield",vrxfield);
	
	OUT("Solving vorticity strengths.");
	gsl_vector *result = gsl_vector_calloc(IN->_size);
	int s;
	gsl_permutation *p = gsl_permutation_alloc(IN->_size);
	gsl_linalg_LU_decomp(influence,p,&s);
	gsl_linalg_LU_solve(influence,p,vrxfield,result);
	//OUT("Result",result);
	
	double tEnd = TIME();
	OUT("Total Time (s)",tEnd - tStart);
	
	OUT("Applying new vorticity strengths.");
	IN->setStates(result);
	
	gsl_matrix_free(influence);
	gsl_vector_free(result);
	gsl_vector_free(states);
	gsl_vector_free(vrxfield);
    DOUT("----------------out pawan::__resolve::rebuild()");
}
