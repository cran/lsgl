/* Routines for linear multiple output using sparse group lasso regression.
 Intended for use with R.
 Copyright (C) 2014 Martin Vincent

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

//Uncomment to turn on debuging
//#undef NDEBUG

//Configuration
//Debugging
#ifndef NDEBUG
#define SGL_DEBUG
#endif

//Runtime checking for numerical problems
#define SGL_RUNTIME_CHECKS

//Check dimension of input objects
#define SGL_DIM_CHECKS

//Converges checks
#define SGL_CONVERGENCE_CHECK

//Exception handling
#define SGL_CATCH_EXCEPTIONS

//Should the timers be activated (only needed for profiling the code)
//#define SGL_TIMING

//Sgl optimizer
#include <sgl.h>

/**********************************
 *
 *  lsgl dense module
 *
 *********************************/

// Module name
#define MODULE_NAME lsgl_dense

//Objective
#include "frobenius_norm_objective.h"

#define OBJECTIVE frobenius

#include <sgl/RInterface/sgl_lambda_seq.h>
#include <sgl/RInterface/sgl_fit.h>

#define PREDICTOR sgl::LinearPredictor < sgl::matrix , sgl::LinearResponse >

#include <sgl/RInterface/sgl_predict.h>
#include <sgl/RInterface/sgl_subsampling.h>

/*********************************
 *
 *  lsgl sparse module
 *
 *********************************/
// Reset macros
#undef MODULE_NAME
#undef OBJECTIVE
#undef PREDICTOR

// Module name
#define MODULE_NAME lsgl_sparse

//Objective
#include "frobenius_norm_objective.h"

#define OBJECTIVE frobenius_spx

#include <sgl/RInterface/sgl_lambda_seq.h>
#include <sgl/RInterface/sgl_fit.h>

#define PREDICTOR sgl::LinearPredictor < sgl::sparse_matrix , sgl::LinearResponse >

#include <sgl/RInterface/sgl_predict.h>
#include <sgl/RInterface/sgl_subsampling.h>

/* **********************************
 *
 *  Registration of methods
 *
 ***********************************/

#include <R_ext/Rdynload.h>

static const R_CallMethodDef sglCallMethods[] = {
		SGL_LAMBDA(lsgl_dense), SGL_LAMBDA(lsgl_sparse),
		SGL_FIT(lsgl_dense), SGL_FIT(lsgl_sparse),
		SGL_PREDICT(lsgl_dense), SGL_PREDICT(lsgl_sparse),
		SGL_SUBSAMPLING(lsgl_dense), SGL_SUBSAMPLING(lsgl_sparse),
		NULL};

extern "C" {
	void R_init_lsgl(DllInfo *info);
}

void R_init_lsgl(DllInfo *info)
{
	// Print warnings
#ifndef SGL_OPENMP_SUPP
    Rcout << "NOTE : openMP (multithreading) is not supported on this system" << std::endl;
#endif

#ifdef SGL_DEBUG
	Rcout << "WARNING : debugging is turned on -- this may increase the runtime" << std::endl;
#endif

// Register the .Call routines.
	R_registerRoutines(info, NULL, sglCallMethods, NULL, NULL);
}
