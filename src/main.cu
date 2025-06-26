#include "kernels.cuh"
#include "host_functions.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: Usage: " << argv[0] << " <velocity set> <ID>" << std::endl;
        return 1;
    }
    std::string VELOCITY_SET = argv[1];
    std::string SIM_ID = argv[2];

    std::string SIM_DIR = createSimulationDirectory(VELOCITY_SET,SIM_ID);
    //computeAndPrintOccupancy();
    initDeviceVars();

    // ================================================================================================== //

    dim3 threadsPerBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);
                        
    dim3 threadsPerBlockX(BLOCK_SIZE_Y*2,BLOCK_SIZE_Z*2);
    dim3 numBlocksX((NY + threadsPerBlockX.x - 1) / threadsPerBlockX.x,
                    (NZ + threadsPerBlockX.y - 1) / threadsPerBlockX.y);

    dim3 threadsPerBlockY(BLOCK_SIZE_X*2,BLOCK_SIZE_Z*2);
    dim3 numBlocksY((NX + threadsPerBlockY.x - 1) / threadsPerBlockY.x,
                    (NZ + threadsPerBlockY.y - 1) / threadsPerBlockY.y);

    dim3 threadsPerBlockZ(BLOCK_SIZE_X*2,BLOCK_SIZE_Y*2);  
    dim3 numBlocksZ((NX + threadsPerBlockZ.x - 1) / threadsPerBlockZ.x,
                    (NY + threadsPerBlockZ.y - 1) / threadsPerBlockZ.y);
                    
    cudaStream_t mainStream;
    checkCudaErrors(cudaStreamCreate(&mainStream));

    gpuInitFieldsAndDistributions<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
    getLastCudaError("gpuInitFieldsAndDistributions");

    auto START_TIME = std::chrono::high_resolution_clock::now();
    for (int STEP = 0; STEP <= NSTEPS ; ++STEP) {
        std::cout << "Passo " << STEP << " de " << NSTEPS << " iniciado..." << std::endl;

        // =================================== INFLOW =================================== //

            gpuApplyInflow<<<numBlocksZ,threadsPerBlockZ,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm,STEP); 
            getLastCudaError("gpuApplyInflow");
            gpuApplyLatInflow<<<numBlocksY,threadsPerBlockY,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm); 
            getLastCudaError("gpuApplyLatInflow");

        // =============================================================================  //
        
        // ========================= COLLISION & STREAMING ========================= //
            
            gpuEvolvePhaseField<<<numBlocks,threadsPerBlock,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm); 
            getLastCudaError("gpuEvolvePhaseField");
            gpuMomCollisionStream<<<numBlocks,threadsPerBlock,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm); 
            getLastCudaError("gpuMomCollisionStream");

        // ========================================================================= //    

        // =================================== BOUNDARIES =================================== //

            gpuReconstructBoundaries<<<numBlocks,threadsPerBlock,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm); 
            getLastCudaError("gpuReconstructBoundaries");
            gpuApplyOutflowX<<<numBlocksX,threadsPerBlockX,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm);
            getLastCudaError("gpuApplyOutflowZ");
            gpuApplyOutflowY<<<numBlocksY,threadsPerBlockY,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm);
            getLastCudaError("gpuApplyOutflowZ");
            gpuApplyOutflowZ<<<numBlocksZ,threadsPerBlockZ,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm);
            getLastCudaError("gpuApplyOutflowZ");

        // ================================================================================== //

        // =================================== DERIVED FIELDS =================================== //

            //gpuDerivedFields<<<numBlocks,threadsPerBlock,DYNAMIC_SHARED_SIZE,mainStream>>> (lbm,dfields); 
            //getLastCudaError("gpuDerivedFields");

        // ====================================================================================== //

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {

            copyAndSaveToBinary(lbm.phi,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"phi");
            copyAndSaveToBinary(lbm.rho,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"rho");
            copyAndSaveToBinary(lbm.uy,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"uy");
            copyAndSaveToBinary(lbm.uz,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"uz");
            //copyAndSaveToBinary(dfields.vorticity_mag,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"vorticity_mag");
            //copyAndSaveToBinary(dfields.q_criterion,NX*NY*NZ,SIM_DIR,SIM_ID,STEP,"q_criterion");

            std::cout << "Passo " << STEP << ": Dados salvos em " << SIM_DIR << std::endl;
        }
    }
    auto END_TIME = std::chrono::high_resolution_clock::now();

    checkCudaErrors(cudaStreamDestroy(mainStream));

    // lbmfields
    cudaFree(lbm.f); 
    cudaFree(lbm.g);
    cudaFree(lbm.phi); 
    cudaFree(lbm.rho);
    cudaFree(lbm.normx);
    cudaFree(lbm.normy); 
    cudaFree(lbm.normz);
    cudaFree(lbm.ux); 
    cudaFree(lbm.uy); 
    cudaFree(lbm.uz);
    cudaFree(lbm.pxx);
    cudaFree(lbm.pyy);
    cudaFree(lbm.pzz);
    cudaFree(lbm.pxy);
    cudaFree(lbm.pxz);
    cudaFree(lbm.pyz);
    cudaFree(lbm.ffx); 
    cudaFree(lbm.ffy); 
    cudaFree(lbm.ffz);

    // derivedfields
    cudaFree(dfields.vorticity_mag);
    cudaFree(dfields.q_criterion);

    std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    long long TOTAL_CELLS = static_cast<long long>(NX) * NY * NZ * NSTEPS;
    double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " seconds\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n" << std::endl;

    generateSimulationInfoFile(SIM_DIR,SIM_ID,VELOCITY_SET,NSTEPS,MACRO_SAVE,TAU,MLUPS);
    getLastCudaError("Final sync");
    return 0;
}
