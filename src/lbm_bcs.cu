#include "kernels.cuh"

#define INFLOW_CASE_THREE

__global__ void gpuApplyRisingInflow(LBMFields d, const int STEP) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 0;

    if (x >= NX || y >= NY) return;

    const float center_x = (NX - 1) * 0.5f;
    const float center_y = 3.0f * DIAM_OIL;

    const float dx = x-center_x, dy = y-center_y;
    const float radial_dist = sqrtf(dx*dx + dy*dy);
    const float radius = 0.5f * DIAM_OIL;
    if (radial_dist > radius) return;

    #ifdef INFLOW_CASE_ONE 
        const float radial_dist_norm = radial_dist / radius;
        const float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        const float profile = 0.5f + 0.5f * tanhf(2.0f * (radius - radial_dist) / 3.0f);
        const float phi_in = profile * envelope; 
        #ifdef PERTURBATION
            const float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * phi_in;
        #else
            const float uz_in = U_JET * phi_in;
        #endif
    #elif defined(INFLOW_CASE_TWO)
        const float radial_dist_norm = radial_dist / radius;
        const float envelope = 1.0f - gpu_smoothstep(0.6f, 1.0f, radial_dist_norm);
        const float phi_in = 1.0f;
        #ifdef PERTURBATION
            const float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f) * envelope;
        #else
            const float uz_in = U_JET * envelope;
        #endif
    #elif defined(INFLOW_CASE_THREE) 
        const float phi_in = 1.0f;
        #ifdef PERTURBATION
            const float uz_in = U_JET * (1.0f + DATAZ[STEP/MACRO_SAVE] * 10.0f);
        #else
            const float uz_in = U_JET
        #endif
    #endif

    const float rho_val = 1.0f;
    const float uu = 1.5f * (uz_in * uz_in);

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
        const float feq = gpu_compute_equilibria(rho_val,0.0f,0.0f,uz_in,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int xx = x + CIX[Q];
        const int yy = y + CIY[Q];
        const int zz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,0.0f,uz_in,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(xx,yy,zz,Q);
        d.g[streamed_idx4] = geq;
    }
}

__global__ void gpuApplyLateralInflow(LBMFields d, const int STEP) {
    if (!inflow_active(STEP,5000)) return;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;  
    const int y = 0;
    const int z = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= NX || z >= NZ) return;

    const float center_x = (NX-1) * 0.5f;
    const float center_z = 10.0f * DIAM_OIL;

    const float dx = x-center_x, dz = z-center_z;
    const float radial_dist = sqrtf(dx*dx + dz*dz);
    const float radius = 0.5f * DIAM_WATER;

    if (radial_dist > radius) return;

    const float phi_in = 0.0f;               
    const float uy_in = U_JET;
    const float rho_val = 1.0f;
    const float uu = 1.5f * (uy_in * uy_in);

    const idx_t idx3_in = gpu_idx_global3(x,y,z);
    d.rho[idx3_in] = rho_val;
    d.phi[idx3_in] = phi_in;
    d.ux[idx3_in] = 0.0f;
    d.uy[idx3_in] = uy_in;
    d.uz[idx3_in] = 0.0f;

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float feq = gpu_compute_equilibria(rho_val,0.0f,uy_in,0.0f,uu,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(sx,sy,sz,Q);
        d.f[streamed_idx4] = to_dtype(feq);
    }

    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_in,0.0f,uy_in,0.0f,Q);
        const idx_t streamed_idx4 = gpu_idx_global4(sx,sy,sz,Q);
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

    const idx_t boundary_idx3 = gpu_idx_global3(x,y,z);
    const float rho_val = d.rho[boundary_idx3];
    const float phi_val = d.phi[boundary_idx3];
    const float ux_val = d.ux[boundary_idx3];
    const float uy_val = d.uy[boundary_idx3];
    const float uz_val = d.uz[boundary_idx3];
    const float uu = 1.5f * (ux_val*ux_val + uy_val*uy_val + uz_val*uz_val);

    const float local_viscosity = phi_val * VISC_OIL + (1.0f - phi_val) * VISC_WATER;
    const float omega = 1.0f / (0.5f + 3.0f * local_viscosity);

    #pragma unroll FLINKS
    for (int Q = 0; Q < FLINKS; ++Q) {
        const int sx = x + CIX[Q],      sy = y + CIY[Q],      sz = z + CIZ[Q];
        const int fx = x + CIX[OPP[Q]], fy = y + CIY[OPP[Q]], fz = z + CIZ[OPP[Q]];
        const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q); 
        const float feq = gpu_compute_equilibria(rho_val,ux_val,uy_val,uz_val,uu,Q);
        if (fx >= 0 && fx < NX && fy >= 0 && fy < NY && fz >= 0 && fz < NZ && sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t neighbor_fluid_idx3 = gpu_idx_global3(fx,fy,fz); 
            const float fneq_reg = (W[Q] * 4.5f) * ((CIX[Q]*CIX[Q] - CSSQ) * d.pxx[neighbor_fluid_idx3] +
                                                    (CIY[Q]*CIY[Q] - CSSQ) * d.pyy[neighbor_fluid_idx3] +
                                                    (CIZ[Q]*CIZ[Q] - CSSQ) * d.pzz[neighbor_fluid_idx3] +
                                                     2.0f * CIX[Q]*CIY[Q] * d.pxy[neighbor_fluid_idx3] +
                                                     2.0f * CIX[Q]*CIZ[Q] * d.pxz[neighbor_fluid_idx3] +
                                                     2.0f * CIY[Q]*CIZ[Q] * d.pyz[neighbor_fluid_idx3]);
            d.f[streamed_boundary_idx4] = to_dtype(feq + (1-omega) * fneq_reg);
        }
    }
    #pragma unroll GLINKS
    for (int Q = 0; Q < GLINKS; ++Q) {
        const int sx = x + CIX[Q];
        const int sy = y + CIY[Q];
        const int sz = z + CIZ[Q];
        const float geq = gpu_compute_truncated_equilibria(phi_val,ux_val,uy_val,uz_val,Q);
        if (sx >= 0 && sx < NX && sy >= 0 && sy < NY && sz >= 0 && sz < NZ) {
            const idx_t streamed_boundary_idx4 = gpu_idx_global4(sx,sy,sz,Q);
            d.g[streamed_boundary_idx4] = geq;
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





