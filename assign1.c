#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[]) {

	int n = 4;

	int rank, nproc, ndims, p1, p2;
	int dims[2], coords[2], cyclic[2], reorder;
	MPI_Comm proc_grid, proc_row, proc_col;

	double *my_blockA, *my_blockB, *my_blockC, *tempA, *tempB;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Request request, send_request, recv_request;

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

	int my_rank;
	MPI_Cart_rank(proc_grid, coords, &my_rank);

	//printf("%4d %4d %4d\n",coords[0], coords[1], rank);

	int row_rank, column_rank, row_size, column_size;

	// Create a communicator for each row
	MPI_Comm_split(proc_grid, coords[0], coords[1], &proc_row);
  	MPI_Comm_rank(proc_row, &row_rank);
  	MPI_Comm_size(proc_row, &row_size);


  	// Create a communicator for each column
	MPI_Comm_split(proc_grid, coords[1], coords[0], &proc_col);
  	MPI_Comm_rank(proc_col, &column_rank);
    MPI_Comm_size(proc_col, &column_size);

    //printf("My row rank: %d\n", row_rank);

	MPI_Type_vector(r1, r2, n, MPI_DOUBLE, &vec_type);
	MPI_Type_commit(&vec_type);

	double *matrixC;

	if (rank == 0) {
		double *matrixA, *matrixB;//, *matrixC;
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

		printf("Matrix A:\n");
		for (i = 0; i < n * n; i++) {
			printf("%f, ", matrixA[i]);
		}
		printf("\n");
		printf("Matrix B:\n");
		for (i = 0; i < n * n; i++) {
			printf("%f, ", matrixB[i]);
		}
		printf("\n");


		for (i = 0; i < p1; i++) {
			for (j = 0; j < p2; j++) {

				// printf("Top corner A: %4f\n", matrixA[i * p1 + j * n * p2]);
				// printf("Top corner B: %4f\n", matrixB[i * p1 + j * n * p2]);

				int grid_rank;
				int loc[2] = {i, j};
				MPI_Cart_rank(proc_grid, loc, &grid_rank);
				MPI_Isend(&matrixA[i * r1 * n + j * r2], 1, vec_type, grid_rank, grid_rank, proc_grid, &send_request);
				MPI_Wait(&send_request, &status);
				MPI_Isend(&matrixB[i * r1 * n + j * r2], 1, vec_type, grid_rank, grid_rank, proc_grid, &send_request);
				MPI_Wait(&send_request, &status);
			}
		}

		//printf("%4d %4d", r1, r2);


		free(matrixA);
		free(matrixB);

	}

	
	my_blockA = malloc(r1 * r2 * sizeof(double));
	my_blockB = malloc(r1 * r2 * sizeof(double));
	my_blockC = calloc(r1 * r2, sizeof(double));


	MPI_Recv(my_blockA, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);
	MPI_Recv(my_blockB, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);

	
	printf("A: %4f, %d\n", my_blockA[0], my_rank);
	printf("B: %4f, %d\n", my_blockB[0], my_rank);


	tempA = malloc(r1 * r2 * sizeof(double));
	//tempB = malloc(r1 * r2 * sizeof(double));


	int k, row, col, l;
	for (k = 0; k < p1; k++) {
		int m = (coords[0] + k) % p1;
		MPI_Isend(my_blockB, r1 * r2, MPI_DOUBLE, (p1 + column_rank - 1) % p1, column_rank, proc_col, &send_request);
		if (row_rank == m) {
			MPI_Bcast(my_blockA, r1 * r2, MPI_DOUBLE, m, proc_row);
			for (row = 0; row < r1; row++) {
				for (col = 0; col < r2; col++) {
					for (l = 0; l < r2; l++) {
						my_blockC[row * r2 + col] += my_blockA[row * r2 + l] * my_blockB[l * r2 + col];
					}
				}
			}
		} else {
			MPI_Bcast(tempA, r1 * r2, MPI_DOUBLE, m, proc_row);
			for (row = 0; row < r1; row++) {
				for (col = 0; col < r2; col++) {
					for (l = 0; l < r2; l++) {
						my_blockC[row * r2 + col] += tempA[row * r2 + l] * my_blockB[l * r2 + col];
					}
				}
			}
		}

		//MPI_Isend(my_blockB, r1 * r2, MPI_DOUBLE, (p1 + column_rank - 1) % p1, column_rank, proc_col, &send_request);
		MPI_Irecv(my_blockB, r1 * r2, MPI_DOUBLE, (column_rank + 1) % p1, (column_rank + 1) % p1, proc_col, &recv_request);
		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		//memcpy(my_blockB, tempB, r1 * r2 * sizeof(double));
	}

	printf("C: %4f, %d\n", my_blockC[0], my_rank);

	MPI_Isend(my_blockC, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &request);

	if (rank == 0) {
		int i;
		for (i = 0; i < nproc; i++) {
			MPI_Probe(i, i, proc_grid, &status);
			int loc[2];
			MPI_Cart_coords(proc_grid, i, ndims, loc);
			MPI_Recv(&matrixC[loc[0] * r1 * n + loc[1] * r2], 1, vec_type, i, i, proc_grid, &status);
		}
	}

	MPI_Wait(&request, &status);

	if (rank == 0) {
		int i;
		printf("Matrix C:\n");
		for (i = 0; i < n * n; i++) {
			printf("%f, ", matrixC[i]);
		}
		printf("\n");
	}
	

	free(my_blockA);
	free(my_blockB);
	free(my_blockC);
	free(tempA);

	MPI_Finalize();

	return 0;
}
