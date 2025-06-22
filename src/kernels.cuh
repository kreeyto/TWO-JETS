#pragma once
#include "device_header.cuh"
#include "device_functions.cuh"

// =======================================================================================
// INITIALIZATION
// =======================================================================================
__global__ void gpuInitFieldsAndDistributions(LBMFields d);

// =======================================================================================
// BOUNDARY CONDITIONS
// =======================================================================================
__global__ void gpuApplyInflow(LBMFields d, const int STEP);
__global__ void gpuApplyLatInflow(LBMFields d);
__global__ void gpuApplyOutflowX(LBMFields d);
__global__ void gpuApplyOutflowY(LBMFields d);
__global__ void gpuApplyOutflowZ(LBMFields d);
__global__ void gpuReconstructBoundaries(LBMFields d); // streaming-safe wrap/periodic

// =======================================================================================
// PHASE FIELD CALCULATIONS
// =======================================================================================
__global__ void gpuEvolvePhaseField(LBMFields d); // AD-based

// =======================================================================================
// FLUID FIELD EVOLUTION
// =======================================================================================
__global__ void gpuMomCollisionStream(LBMFields d); // fused BGK + streaming

// =======================================================================================
// DERIVED FIELDS
// =======================================================================================
__global__ void gpuDerivedFields(LBMFields lbm, DerivedFields dfields); // vorticity, Q-criterion, etc.
