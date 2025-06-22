#include "kernels.cuh"

#define INFLOW_CASE_THREE

__global__ void gpuApplyInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    float center_x = (NX-1) * 0.5f;
    float center_y = (NY-1) * 0.5f;

    float dx = x-center_x, dy = y-center_y;
    float radial_dist = sqrtf(dx*dx + dy*dy);
    float radius = 0.5f * DIAM;
    if (radial_dist > radius) return;

    #ifdef INFLOW_CASE_ONE 
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
        float phi_in = profile * envelope; 
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
        #else
            float uz_in = U_JET * phi_in;
        #endif
    #elif defined(INFLOW_CASE_TWO)
        float radial_dist_norm = radial_dist / radius;
        float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        float phi_in = 1.0f;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * envelope;
        #else
            float uz_in = U_JET * envelope;
        #endif
    #elif defined(INFLOW_CASE_THREE) 
        float phi_in = 1.0f;
        #ifdef PERTURBATION
            float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
        #else
            float uz_in = U_JET
        #endif
    #endif

    float rho_val = 1.0f;
    float uu = 1.5f * (uz_in * uz_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = 0.0f;
    d.uz[idx3_in] = uz_in;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,0.0f,uz_in,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}

__global__ void gpuApplyLatInflow(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;  
    const int y = 0;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || z >= NZ) return;

    float center_x = (NX-1) * 0.5f;
    float center_z = (NZ-1) * 0.5f;

    float dx = x-center_x, dz = z-center_z;
    float radial_dist = sqrtf(dx*dx + dz*dz);
    float radius = 0.5f * DIAM;

    if (radial_dist > radius) return;

    float phi_in = 0.0f;               
    float uy_in = U_JET;              
    float rho_val = 1.0f;
    float uu = 1.5f * (uy_in * uy_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = uy_in;
    d.uz[idx3_in] = 0.0f;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float feq = gpu_compute_equilibria(rho_val,0.0f,uy_in,0.0f,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }

    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,uy_in,0.0f,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}

__global__ void gpuReconstructBoundaries(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    bool is_valid_edge = (x < NX && y < NY && z < NZ) &&
                         (x == 0 || x == NX-1 || y == NY-1 || z == NZ-1); 
    if (!is_valid_edge) return;
    const idx_t idx3 = gpu_idx_global3(x,y,z);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
            d.f[streamed_idx4] = to_dtype(W[Q] * d.rho[idx3] - W[Q]);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        if (xx >= 0 && xx < NX && yy >= 0 && yy < NY && zz >= 0 && zz < NZ) {
            const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
            d.g[streamed_idx4] = W_G[Q] * d.phi[idx3] - W_G[Q];
        }
    }
}

// ============================================================================================================== //

__global__ void gpuApplyOutflowX(LBMFields d) {
    const int x = NX-1;
    const int y = threadIdx.x + blockIdx.x * blockDim.x;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;

    if (y >= NY || z >= NZ) return;

    d.phi[gpu_idx_global3(x,y,z)] = d.phi[gpu_idx_global3(x-1,y,z)];
    d.rho[gpu_idx_global3(x,y,z)] = d.rho[gpu_idx_global3(x-1,y,z)];
    d.ux[gpu_idx_global3(x,y,z)] = d.ux[gpu_idx_global3(x-1,y,z)];
    d.uy[gpu_idx_global3(x,y,z)] = d.uy[gpu_idx_global3(x-1,y,z)];
    d.uz[gpu_idx_global3(x,y,z)] = d.uz[gpu_idx_global3(x-1,y,z)];
}

__global__ void gpuApplyOutflowY(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = NY-1;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || z >= NZ) return;

    d.phi[gpu_idx_global3(x,y,z)] = d.phi[gpu_idx_global3(x,y-1,z)];
    d.rho[gpu_idx_global3(x,y,z)] = d.rho[gpu_idx_global3(x,y-1,z)];
    d.ux[gpu_idx_global3(x,y,z)] = d.ux[gpu_idx_global3(x,y-1,z)];
    d.uy[gpu_idx_global3(x,y,z)] = d.uy[gpu_idx_global3(x,y-1,z)];
    d.uz[gpu_idx_global3(x,y,z)] = d.uz[gpu_idx_global3(x,y-1,z)];
}

__global__ void gpuApplyOutflowZ(LBMFields d) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = NZ-1;

    if (x >= NX || y >= NY) return;

    d.phi[gpu_idx_global3(x,y,z)] = d.phi[gpu_idx_global3(x,y,z-1)];
    d.rho[gpu_idx_global3(x,y,z)] = d.rho[gpu_idx_global3(x,y,z-1)];
    d.ux[gpu_idx_global3(x,y,z)] = d.ux[gpu_idx_global3(x,y,z-1)];
    d.uy[gpu_idx_global3(x,y,z)] = d.uy[gpu_idx_global3(x,y,z-1)];
    d.uz[gpu_idx_global3(x,y,z)] = d.uz[gpu_idx_global3(x,y,z-1)];
}





