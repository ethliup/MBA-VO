#include "core/common/CudaDefs.h"
#include "core/cuda/cuda_reduction.h"
#include "optimizer/ceres_cuda/cost_function_reprojection_error.h"
#include "vo/blur_aware_tracker/reduction.h"

__global__ void kernel_eval_cost(ceres_cuda::CostFunctionReprojectionErrorFunctor *cost, double *T_b2w, double *P3d, double *residuals, double *jacob_tangent, double *jacob_p3d)
{
    cost->evaluate_residual_and_jacobians(T_b2w, P3d, residuals, jacob_tangent, jacob_p3d);
}

void evaluate_cost_cuda(ceres_cuda::CostFunctionReprojectionErrorFunctor *cost, double *T_b2w, double *P3d, double *residuals, double *jacob_tangent, double *jacob_p3d)
{
    dim3 gridDim = dim3(1);
    dim3 blockDim = dim3(1);

    kernel_eval_cost<<<gridDim, blockDim>>>(cost, T_b2w, P3d, residuals, jacob_tangent, jacob_p3d);
    cudaDeviceSynchronize();
}

void sum_reduction(double *data, int H, int W, int C, double *sum)
{
    SLAM::Core::cuda_channel_wise_sum_reduction(data, H, W, C, sum);
}
