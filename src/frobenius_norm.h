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

#ifndef FOBENIUS_NORM_HPP_
#define FOBENIUS_NORM_HPP_

//type_X : sgl::matrix or sgl::sparse_matrix
//type_Y : sgl::matrix or sgl::sparse_matrix

template < typename type_X, typename type_Y >
class FrobeniusLoss {

public:

	const sgl::natural n_samples;
	const sgl::natural n_responses;

private:

	type_Y const& Y; //response - matrix of size n_samples x n_responses
	sgl::matrix lp; //linear predictors - matrix of size n_samples x n_responses

public:

	typedef sgl::hessian_identity<true> hessian_type; //constant hessians of type double * Id
	//typedef sgl::hessian_full hessian_type;

	typedef sgl::DataPackage_2< sgl::MatrixData<type_X>,
				sgl::MultiResponse<type_Y, 'Y'> > data_type;



	FrobeniusLoss()
			: 	n_samples(0),
				n_responses(0),
				Y(sgl::null_matrix),
				lp(n_samples, n_responses)	{
	}

	FrobeniusLoss(data_type const& data)
			: 	n_samples(data.get_A().n_samples),
				n_responses(data.get_B().n_groups),
				Y(data.get_B().response),
				lp(n_samples, n_responses) {
	}

	void set_lp(sgl::matrix const& lp)
	{
		this->lp = lp;
	}

	void set_lp_zero()
	{
		lp.zeros(n_samples, n_responses);
	}

	const sgl::matrix gradients() const
	{
		return static_cast<double>(2)/static_cast<double>(n_samples)*trans(lp-Y);
	}

	void compute_hessians() const
	{
		return;
	}

    const double hessians(sgl::natural i) const
	{
		return static_cast<double>(2)/static_cast<double>(n_samples);
	}

	const sgl::numeric sum_values() const
	{
		return trace(trans(lp-Y)*(lp-Y))/static_cast<double>(n_samples);
	}

};

// Std design matrix
typedef sgl::ObjectiveFunctionType < sgl::GenralizedLinearLossDense < FrobeniusLoss < sgl::matrix, sgl::matrix > > ,
		FrobeniusLoss < sgl::matrix, sgl::matrix >::data_type > frobenius;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossSparse < FrobeniusLoss < sgl::sparse_matrix, sgl::matrix > > ,
		FrobeniusLoss < sgl::sparse_matrix, sgl::matrix >::data_type > frobenius_spx;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossDense < FrobeniusLoss < sgl::matrix, sgl::sparse_matrix > > ,
		FrobeniusLoss < sgl::matrix, sgl::sparse_matrix >::data_type > frobenius_spy;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossSparse < FrobeniusLoss < sgl::sparse_matrix, sgl::sparse_matrix > > ,
		FrobeniusLoss < sgl::sparse_matrix, sgl::sparse_matrix >::data_type > frobenius_spx_spy;

#endif /* FOBENIUS_NORM_HPP_ */
