#ifndef GPUFIT_MODELS_CUH_INCLUDED
#define GPUFIT_MODELS_CUH_INCLUDED

#include <assert.h>
#include "linear_1d.cuh"
#include "gauss_1d.cuh"
#include "gauss_2d.cuh"
#include "gauss_2d_elliptic.cuh"
#include "gauss_2d_rotated.cuh"
#include "cauchy_2d_elliptic.cuh"
#include "fletcher_powell_helix.cuh"
#include "brown_dennis.cuh"
#include "spline_1d.cuh"
#include "spline_2d.cuh"
#include "spline_3d.cuh"
#include "spline_3d_multichannel.cuh"
#include "spline_3d_phase_multichannel.cuh"
#include "lorentz_1d.cuh"
#include "lorentz_1d_double.cuh"
#include "lorentz_1d_oct.cuh"
#include "lorentz_1d_oct_single_offset.cuh"
#include "lorentz_1d_double_single_offset.cuh"
#include "lorentz_1d_triple.cuh"
#include "lorentz_1d_quad.cuh"
#include "gauss_1d_double.cuh"
#include "double_exponential_decay_1d.cuh"
#include "single_exponential_decay_1d.cuh"

__device__ void calculate_model(
    ModelID const model_id,
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    int const user_info_size)
{
    switch (model_id)
    {
    case GAUSS_1D:
        calculate_gauss1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D:
        calculate_gauss2d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ELLIPTIC:
        calculate_gauss2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ROTATED:
        calculate_gauss2drotated(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case CAUCHY_2D_ELLIPTIC:
        calculate_cauchy2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LINEAR_1D:
        calculate_linear1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case FLETCHER_POWELL_HELIX:
        calculate_fletcher_powell_helix(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BROWN_DENNIS:
        calculate_brown_dennis(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_1D:
        calculate_spline1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_2D:
        calculate_spline2d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D:
        calculate_spline3d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D_MULTICHANNEL:
        calculate_spline3d_multichannel(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D_PHASE_MULTICHANNEL:
        calculate_spline3d_phase_multichannel(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D:
        calculate_lorentz1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_DOUBLE:
        calculate_lorentz_1d_double(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_OCT:
        calculate_lorentz_1d_oct(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_OCT_SINGLE_OFFSET:
        calculate_lorentz_1d_oct_single_offset(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_DOUBLE_SINGLE_OFFSET:
        calculate_lorentz_1d_double_single_offset(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_TRIPLE:
        calculate_lorentz_1d_triple(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LORENTZ_1D_QUAD:
        calculate_lorentz_1d_quad(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_1D_DOUBLE:
        calculate_gauss_1d_double(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case DOUBLE_EXPONENTIAL_DECAY_1D:
        calculate_double_exponential_decay_1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SINGLE_EXPONENTIAL_DECAY_1D:
        calculate_single_exponential_decay_1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    default:
        assert(0); // unknown model ID
    }
}

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions)
{
    switch (model_id)
    {
    case GAUSS_1D:                          n_parameters = 4; n_dimensions = 1; break;
    case GAUSS_2D:                          n_parameters = 5; n_dimensions = 2; break;
    case GAUSS_2D_ELLIPTIC:                 n_parameters = 6; n_dimensions = 2; break;
    case GAUSS_2D_ROTATED:                  n_parameters = 7; n_dimensions = 2; break;
    case CAUCHY_2D_ELLIPTIC:                n_parameters = 6; n_dimensions = 2; break;
    case LINEAR_1D:                         n_parameters = 2; n_dimensions = 1; break;
    case FLETCHER_POWELL_HELIX:             n_parameters = 3; n_dimensions = 1; break;
    case BROWN_DENNIS:                      n_parameters = 4; n_dimensions = 1; break;
    case SPLINE_1D:                         n_parameters = 3; n_dimensions = 1; break;
    case SPLINE_2D:                         n_parameters = 4; n_dimensions = 2; break;
    case SPLINE_3D:                         n_parameters = 5; n_dimensions = 3; break;
    case SPLINE_3D_MULTICHANNEL:            n_parameters = 5; n_dimensions = 4; break;
    case SPLINE_3D_PHASE_MULTICHANNEL:      n_parameters = 6; n_dimensions = 4; break;
    case LORENTZ_1D:                        n_parameters = 4; n_dimensions = 1; break;
    case LORENTZ_1D_DOUBLE:                 n_parameters = 4*2; n_dimensions = 1; break;
    case LORENTZ_1D_OCT:                    n_parameters = 4*8; n_dimensions = 1; break;
    case LORENTZ_1D_OCT_SINGLE_OFFSET:      n_parameters = 3*8 + 1; n_dimensions = 1; break;
    case LORENTZ_1D_DOUBLE_SINGLE_OFFSET:   n_parameters = 3*2 + 1; n_dimensions = 1; break;
    case LORENTZ_1D_TRIPLE:                 n_parameters = 4*3; n_dimensions = 1; break;
    case LORENTZ_1D_QUAD:                   n_parameters = 4*4; n_dimensions = 1; break;
    case GAUSS_1D_DOUBLE:                   n_parameters = 4*2; n_dimensions = 1; break;
    case DOUBLE_EXPONENTIAL_DECAY_1D:       n_parameters = 3*2 + 1; n_dimensions = 1; break;
    case SINGLE_EXPONENTIAL_DECAY_1D:       n_parameters = 3*1 + 1; n_dimensions = 1; break;
    default: throw std::runtime_error("unknown model ID");
    }
}

#endif // GPUFIT_MODELS_CUH_INCLUDED