#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[]) {

	int n = 4;
	int rank, nproc, ndims, p1, p2;
	int dims[2], coords[2], cyclic[2], reorder;
	MPI_Comm proc_grid, proc_row, proc_col;

	double *tempB,*tempA;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Request request;

	MPI_Status status;

	MPI_Datatype vec_type;

	p1 = sqrt(nproc);
	p2 = sqrt(nproc);
	ndims = 2;
	cyclic[0] = 0;
	cyclic[1] = 0;
	reorder = 1;

	dims[0] = p1;
	dims[1] = p2;

	int r1 = n / p1;
	int r2 = n / p2;

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &proc_grid);
	MPI_Cart_coords(proc_grid, rank, ndims, coords);

	//printf("%4d %4d %4d\n",coords[0], coords[1], rank);

	MPI_Comm_split(proc_grid, coords[0], coords[1], &proc_row);
	MPI_Comm_split(proc_grid, coords[1], coords[0], &proc_col);

	//MPI_Datatype vec_type;
	MPI_Type_vector(r1, r2, n, MPI_DOUBLE, &vec_type);
	MPI_Type_commit(&vec_type);


	int row_rank,column_rank,row_size,column_size;
	 // Create a communicator for each row
  	
  	MPI_Comm_rank(proc_row,&row_rank);
  	MPI_Comm_size(proc_row,&row_size);

  	// Create a communicator for each column
  	
  	MPI_Comm_rank(proc_col,&column_rank);
    MPI_Comm_size(proc_col,&column_size);

	

	if (rank == 0) {
		double *matrixA, *matrixB,* matrixC;
		matrixA = malloc(n * n * sizeof(double));
		matrixB = malloc(n * n * sizeof(double));
		matrixC = malloc(n * n * sizeof(double));

		int i, j;
		time_t t;
		srand(time(&t));

		for (i = 0; i < n * n; i++) {
			//matrixA[i] = (double) rand() / RAND_MAX;
			//matrixB[i] = (double) rand() / RAND_MAX;
			matrixA[i] = i;
			matrixB[i] = i + n * n;
		}

		// printf("Matrix A:\n");
		// for (i = 0; i < n * n; i++) {
		// 	printf("%f, ", matrixA[i]);
		// }
		// printf("Matrix B:\n");
		// for (i = 0; i < n * n; i++) {
		// 	printf("%f, ", matrixB[i]);
		// }


		for (i = 0; i < p1; i++) {
			for (j = 0; j < p2; j++) {

				printf("Top corner: %4f\n", matrixA[i * p1 + j * n * p2]);

				int grid_rank;
				int loc[2] = {i, j};
				MPI_Cart_rank(proc_grid, loc, &grid_rank);
				MPI_Isend(&matrixA[i * p1 + j * n * p2], 1, vec_type, grid_rank, grid_rank, proc_grid, &request);
				MPI_Wait(&request, &status);
				MPI_Isend(&matrixB[i * p1 + j * n * p2], 1, vec_type, grid_rank, grid_rank, proc_grid, &request);
				MPI_Wait(&request, &status);
			}
		}

		//printf("%4d %4d", r1, r2);


		free(matrixA);
		free(matrixB);

	}

	double *my_blockA, *my_blockB,*my_blockC;
	my_blockA = malloc(r1 * r2 * sizeof(double));
	my_blockB = malloc(r1 * r2 * sizeof(double));
	my_blockC = malloc(r1 * r2 * sizeof(double));

	int my_rank;
	MPI_Cart_rank(proc_grid, coords, &my_rank);

	MPI_Recv(my_blockA, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);
	MPI_Recv(my_blockB, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);

	
	printf("A: %4f\n", my_blockA[0]);
	printf("B: %4f\n", my_blockB[0]);



	tempA=(double*)calloc(r1*r1,sizeof(double));
	tempB=(double*)calloc(r1*r1,sizeof(double));


	int k;
	for (k = 0; k < p1; k++) {
		int m = (coords[0] + k) % p1;
		if (row_rank==m){
			MPI_Bcast(my_blockA, r1 * r2, MPI_DOUBLE, m, proc_row);
		}
		else {
			MPI_Bcast(tempB, r1 * r2, MPI_DOUBLE, m, proc_row);
			

		}
		
	}

	free(my_blockA);
	free(my_blockB);

	MPI_Finalize();

	return 0;
}
