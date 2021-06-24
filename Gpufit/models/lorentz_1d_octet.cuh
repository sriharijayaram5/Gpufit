#ifndef GPUFIT_LORENTZ1D_CUH_INCLUDED
#define GPUFIT_LORENTZ1D_CUH_INCLUDED

/* Description of the calculate_loentz1d function
* ==============================================
*
* This function calculates the values of one-dimensional lorentzian model functions
* and their partial derivatives with respect to the model parameters. 
*
* This function makes use of the user information data to pass in the 
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* There are three possibilities regarding the X values:
*
*   No X values provided: 
*
*       If no user information is provided, the (X) coordinate of the 
*       first data value is assumed to be (0.0).  In this case, for a 
*       fit size of M data points, the (X) coordinates of the data are 
*       simply the corresponding array index values of the data array, 
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then 
*       the same X values will be used for all fits.  In this case, the 
*       size of the user_info array (in bytes) must equal 
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array 
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: width (standard deviation)
*             p[3]: offset
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. 
*
* user_info_size: The size of user_info in bytes. 
*
* Calling the calculate_lorentz1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_lorentz1d(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL * user_info_float = (REAL*)user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters

    REAL const * p = parameters;
    
    // value
    // REAL const ex_2 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));

    REAL const ex_1 =  ((p[2] * p[2]) / (((x - p[1]) * (x - p[1])) + (p[2] * p[2]))); 
    value[point_index] = (p[0] * ex_1 + p[3]);

    int n = 1;
    REAL const ex_2 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_2 + p[3+(n*4)]);

    n = 2;
    REAL const ex_3 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_3 + p[3+(n*4)]);

    n = 3;
    REAL const ex_4 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_4 + p[3+(n*4)]);
    
    n = 4;
    REAL const ex_5 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_5 + p[3+(n*4)]);

    n = 5;
    REAL const ex_6 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_6 + p[3+(n*4)]);

    n = 6;
    REAL const ex_7 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_7 + p[3+(n*4)]);

    n = 7;
    REAL const ex_8 =  ((p[2+(n*4)] * p[2+(n*4)]) / (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    value[point_index] += (p[0+(n*4)] * ex_8 + p[3+(n*4)]);

    // derivative

    REAL * current_derivative = derivative + point_index;

    // current_derivative[0 * n_points]  = ex_2;
    // current_derivative[1 * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    // current_derivative[2 * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    // current_derivative[3 * n_points]  = 1;

    current_derivative[0 * n_points]  = ex_1;
    current_derivative[1 * n_points]  = (2 * p[0] * p[2] * p[2]) * (x - p[1])/ ((((x - p[1]) * (x - p[1])) + (p[2] * p[2])) * (((x - p[1]) * (x - p[1])) + (p[2] * p[2])));
    current_derivative[2 * n_points]  = 2 * p[0] * p[2] * (((p[1] - x) * (p[1] - x)) * ((p[1] - x) * (p[1] - x))) / (  ( (((p[1] - x) * (p[1] - x)) * ((p[1] - x) * (p[1] - x))) + (p[2] * p[2]) ) * ( (((p[1] - x) * (p[1] - x)) * ((p[1] - x) * (p[1] - x))) + (p[2] * p[2]) ));
    current_derivative[3 * n_points]  = 1;

    n=1;
    current_derivative[(0+(n*4)) * n_points]  = ex_2;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=2;
    current_derivative[(0+(n*4)) * n_points]  = ex_3;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=3;
    current_derivative[(0+(n*4)) * n_points]  = ex_4;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=4;
    current_derivative[(0+(n*4)) * n_points]  = ex_5;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=5;
    current_derivative[(0+(n*4)) * n_points]  = ex_6;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=6;
    current_derivative[(0+(n*4)) * n_points]  = ex_7;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;

    n=7;
    current_derivative[(0+(n*4)) * n_points]  = ex_8;
    current_derivative[(1+(n*4)) * n_points]  = (2 * p[0+(n*4)] * p[2+(n*4)] * p[2+(n*4)]) * (x - p[1+(n*4)])/ ((((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])) * (((x - p[1+(n*4)]) * (x - p[1+(n*4)])) + (p[2+(n*4)] * p[2+(n*4)])));
    current_derivative[(2+(n*4)) * n_points]  = 2 * p[0+(n*4)] * p[2+(n*4)] * (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) / (  ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ) * ( (((p[1+(n*4)] - x) * (p[1+(n*4)] - x)) * ((p[1+(n*4)] - x) * (p[1+(n*4)] - x))) + (p[2+(n*4)] * p[2+(n*4)]) ));
    current_derivative[(3+(n*4)) * n_points]  = 1;
}

#endif
