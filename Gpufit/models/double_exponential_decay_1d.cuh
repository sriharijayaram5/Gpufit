#ifndef GPUFIT_DOUBLE_EXPONENTIAL_DECAY_1D_CUH_INCLUDED
#define GPUFIT_DOUBLE_EXPONENTIAL_DECAY_1D_CUH_INCLUDED

/* Description of the calculate_loentz1d_ntet function
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

__device__ void calculate_double_exponential_decay_1d(
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
    else 
    {
        x = user_info_float[point_index];
    }
    
    // parameters

    REAL const * p = parameters;
    
    // value
    // REAL const ex_2 =  ((p[2+(n*3)] * p[2+(n*3)]) / (((x - p[1+(n*3)]) * (x - p[1+(n*3)])) + (p[2+(n*3)] * p[2+(n*3)])));

	const int N = 2;
    float *ex_n = new float[N];
    value[point_index] = 0;
    for (int n = 0; n < N; n++){
        ex_n[n] = exp(-1 * (pow(x/p[1+(n*3)], p[2+(n*3)])));
        value[point_index] += (p[0+(n*3)] * ex_n[n]);
    };
    value[point_index] += p[3*N];

    // derivative

    REAL * current_derivative = derivative + point_index;

    for (int n = 0; n < N; n++){
        current_derivative[(0+(n*3)) * n_points]  = ex_n[n];
        current_derivative[(1+(n*3)) * n_points]  = (ex_n[n] * p[0+(n*3)] * p[2+(n*3)] * pow(x/p[1+(n*3)], p[2+(n*3)])) / p[1+(n*3)];
        current_derivative[(2+(n*3)) * n_points]  = -1 * ex_n[n] * p[0+(n*3)] * pow(x/p[1+(n*3)], p[2+(n*3)]) * log(x/p[1+(n*3)]);
    };
    current_derivative[(3*N) * n_points]  = 1.0;

	delete [] ex_n;

  
}

#endif
