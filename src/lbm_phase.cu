#include "kernels.cuh"

__global__ void gpuPhi(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    float pop[GLINKS];
    pop[0] = d.g[gpu_idx_global4(x,y,z,0)];
    pop[1] = d.g[gpu_idx_global4(x,y,z,1)];
    pop[2] = d.g[gpu_idx_global4(x,y,z,2)];
    pop[3] = d.g[gpu_idx_global4(x,y,z,3)];
    pop[4] = d.g[gpu_idx_global4(x,y,z,4)];
    pop[5] = d.g[gpu_idx_global4(x,y,z,5)];
    pop[6] = d.g[gpu_idx_global4(x,y,z,6)];

    const float phi_val = (pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6]) + 1.0f;
        
    d.phi[idx3] = phi_val;
}

__global__ void gpuGradients(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float gradx = 0.375f * (d.phi[gpu_idx_global3(x+1,y,z)] - d.phi[gpu_idx_global3(x-1,y,z)]);
    const float grady = 0.375f * (d.phi[gpu_idx_global3(x,y+1,z)] - d.phi[gpu_idx_global3(x,y-1,z)]);
    const float gradz = 0.375f * (d.phi[gpu_idx_global3(x,y,z+1)] - d.phi[gpu_idx_global3(x,y,z-1)]);

    const float grad2 = gradx*gradx + grady*grady + gradz*gradz;
    const float mag = rsqrtf(grad2 + 1e-6f);
    const float ind_val = grad2 * mag;
    const float normx_val = gradx * mag;
    const float normy_val = grady * mag;
    const float normz_val = gradz * mag;

    d.normx[idx3] = normx_val;
    d.normy[idx3] = normy_val;
    d.normz[idx3] = normz_val;
    d.ind[idx3] = ind_val;
}

__global__ void gpuForces(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;

    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float ind_val = d.ind[idx3];
    const float normx_val = d.normx[idx3];
    const float normy_val = d.normy[idx3];
    const float normz_val = d.normz[idx3];

    const float curvature = -0.375f * (d.normx[gpu_idx_global3(x+1,y,z)] - d.normx[gpu_idx_global3(x-1,y,z)] +
                                       d.normy[gpu_idx_global3(x,y+1,z)] - d.normy[gpu_idx_global3(x,y-1,z)] +
                                       d.normz[gpu_idx_global3(x,y,z+1)] - d.normz[gpu_idx_global3(x,y,z-1)]);

    const float coeff_force = SIGMA * curvature;
    d.ffx[idx3] = coeff_force * normx_val * ind_val;
    d.ffy[idx3] = coeff_force * normy_val * ind_val;
    d.ffz[idx3] = coeff_force * normz_val * ind_val;
}

__global__ void gpuEvolvePhaseField(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ || 
        x == 0 || x == NX-1 || 
        y == 0 || y == NY-1 || 
        z == 0 || z == NZ-1) return;
        
    const idx_t idx3 = gpu_idx_global3(x,y,z);

    const float phi_val = d.phi[idx3];
    const float ux_val = d.ux[idx3];
    const float uy_val = d.uy[idx3];
    const float uz_val = d.uz[idx3];
    const float normx_val = d.normx[idx3];
    const float normy_val = d.normy[idx3];
    const float normz_val = d.normz[idx3];
    const float phi_norm = GAMMA * phi_val * (1.0f - phi_val);
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_val,ux_val,uy_val,uz_val,Q);
        const float anti_diff = W_G[Q] * phi_norm * (CIX[Q] * normx_val + CIY[Q] * normy_val + CIZ[Q] * normz_val);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq + anti_diff;
    }
} 
