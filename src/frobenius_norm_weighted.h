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

#ifndef FOBENIUS_NORM_WEIGHTED_HPP_
#define FOBENIUS_NORM_WEIGHTED_HPP_

//type_X : sgl::matrix or sgl::sparse_matrix
//type_Y : sgl::matrix or sgl::sparse_matrix

template < typename type_X, typename type_Y >
class FrobeniusLossWeighted {

public:

	const sgl::natural n_samples;
	const sgl::natural n_responses;

private:

	type_Y const& Y; //response - matrix of size n_samples x n_responses
	sgl::matrix const& W; //vector of size n_samples x n_responses

	sgl::matrix lp; //linear predictors - matrix of size n_samples x n_responses

public:

	typedef sgl::hessian_diagonal<false> hessian_type;

	typedef sgl::DataPackage_3< sgl::MatrixData<type_X>,
				sgl::MultiResponse<type_Y, 'Y'>,
				sgl::Data<sgl::matrix, 'W'> > data_type;



	FrobeniusLossWeighted()
			: 	n_samples(0),
				n_responses(0),
				Y(sgl::null_matrix),
				W(sgl::null_matrix),
				lp(n_samples, n_responses)	{
	}

	FrobeniusLossWeighted(data_type const& data)
			: 	n_samples(data.get_A().n_samples),
				n_responses(data.get_B().n_groups),
				Y(data.get_B().response),
				W(data.get_C().data),
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
		return static_cast<double>(2)*trans(W%(lp-Y));
	}

	void compute_hessians() const
	{
		return;
	}

    const sgl::vector hessians(sgl::natural i) const
	{
		return static_cast<double>(2)*W.row(i);
	}

	const sgl::numeric sum_values() const
	{
		return accu(W%(lp-Y)%(lp-Y));
//		return trace(trans(lp-Y)*(lp-Y))/static_cast<double>(n_samples);
	}

};

typedef sgl::ObjectiveFunctionType < sgl::GenralizedLinearLossDense < FrobeniusLossWeighted < sgl::matrix, sgl::matrix > > ,
		FrobeniusLossWeighted < sgl::matrix, sgl::matrix >::data_type > frobenius_w;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossSparse < FrobeniusLossWeighted < sgl::sparse_matrix, sgl::matrix > > ,
		FrobeniusLossWeighted < sgl::sparse_matrix, sgl::matrix >::data_type > frobenius_w_spx;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossDense < FrobeniusLossWeighted < sgl::matrix, sgl::sparse_matrix > > ,
		FrobeniusLossWeighted < sgl::matrix, sgl::sparse_matrix >::data_type > frobenius_w_spy;

typedef sgl::ObjectiveFunctionType <
		sgl::GenralizedLinearLossSparse < FrobeniusLossWeighted < sgl::sparse_matrix, sgl::sparse_matrix > > ,
		FrobeniusLossWeighted < sgl::sparse_matrix, sgl::sparse_matrix >::data_type > frobenius_w_spx_spy;

#endif /* FOBENIUS_NORM_WEIGHTED_HPP_ */
