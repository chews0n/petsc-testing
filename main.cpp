#include <iostream>
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <mpi.h>
#include <chrono>
#include <unistd.h>
#include <vector>

using namespace std::chrono;
int main(int argc, char* argv[]) {
	// works with 2024.1, breaks with 2025.1 versions of mkl/pardiso

	int locmpisize, locmpirank;
	PetscInitialize(&argc, &argv, nullptr, nullptr);

	int globalvecsize = 61494;
	std::vector<int> procsize ={7813, 7674, 7458, 7597, 7651, 7914, 7913, 7474};

	MPI_Comm_size(PETSC_COMM_WORLD, &locmpisize);
	MPI_Comm_rank(PETSC_COMM_WORLD, &locmpirank);

	Vec x, b;
	Mat A;      /* linear system matrix */
	PetscViewer viewer; /* viewer */
	PetscViewer viewerprint;
	KSP ksp;
	PC pc;

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "examples/J_SOD EFACF0p5_NetP200_604_1.bin", FILE_MODE_READ, &viewer);

	MatCreate(PETSC_COMM_WORLD, &A);
	MatSetSizes(A, procsize[locmpirank], procsize[locmpirank], globalvecsize, globalvecsize);
	MatSetFromOptions(A);
	MatLoad(A, viewer);

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "examples/rhs_SOD EFACF0p5_NetP200_604_1.bin", FILE_MODE_READ, &viewer);

	VecCreate(PETSC_COMM_WORLD, &b);
	VecSetSizes(b, procsize[locmpirank], globalvecsize);
	VecSetFromOptions(b);
	VecLoad(b, viewer);

	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "examples/dx_SOD EFACF0p5_NetP200_604_1.bin", FILE_MODE_READ, &viewer);

	VecCreate(PETSC_COMM_WORLD, &x);
	VecSetSizes(x, procsize[locmpirank], globalvecsize);
	VecSetFromOptions(x);
	VecLoad(x, viewer);

	PetscViewerDestroy(&viewer);

	int localvectorsize = 0;
	VecGetLocalSize(x, &localvectorsize);

	std::cout << "rank: " << locmpirank << " vec size: " << localvectorsize << std::endl;

	KSPCreate(PETSC_COMM_WORLD, &ksp);

	KSPSetOperators(ksp, A, A);

	KSPSetType(ksp, KSPPREONLY);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCLU);

	MatSolverType directsolvertype;
	directsolvertype = MATSOLVERMKL_CPARDISO;
	PCFactorSetMatSolverType(pc, directsolvertype);
	PCFactorSetUpMatSolverType(pc);

	Mat F;
	PCFactorGetMatrix(pc, &F);

	MatMkl_CPardisoSetCntl(F, 34, 0);
	MatMkl_CPardisoSetCntl(F, 65, 1);
	MatMkl_CPardisoSetCntl(F, 27, 1);
	MatMkl_CPardisoSetCntl(F, 68, 1);

	KSPSetTolerances(ksp,1e-11,1e-50,1e5,1e4);

	KSPSetFromOptions(ksp);
	KSPSetUp(ksp);

	auto start = high_resolution_clock::now();
	KSPSolve(ksp, b, x);
	auto stop = high_resolution_clock::now();

	if (locmpirank == 0) {
		auto duration = duration_cast<milliseconds>(stop - start);

		std::cout << "Solve duration: " << duration.count() << std::endl;
	}


	KSPConvergedReason reason;
	KSPGetConvergedReason(ksp,&reason);

	PetscViewerASCIIOpen(PETSC_COMM_WORLD,"divergence_dx.bin",&viewerprint);
	VecView(x,viewerprint);
	PetscViewerDestroy(&viewerprint);

	// Finalize Petsc
	PetscFinalize();


	return 0;
}
