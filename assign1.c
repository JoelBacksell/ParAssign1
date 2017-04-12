/*
 * Solution to Compulsory Assignment 1
 * Parallel and Distributed Programming
 * Spring 2017
 * Authors: Joel Backsell and Charalampos Kominos
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// Function for time measurements
double timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double seconds = tv.tv_sec + (double) tv.tv_usec / 1000000;
	return seconds;
}

int main(int argc, char *argv[]) {

	// Variables of world communication
	int rank, nproc;

	// Initialize MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n;

	// Set size of matrix to command line argument
	if (argc != 2) {
		if (rank == 0)
			printf("Usage: %s size-of-matrix\n", argv[0]);
		MPI_Finalize();
		return 0;
	} else {
		n = atoi(argv[1]);
	}

	// Time variable
	double t;

	// Variables for Cartesian topology processor grid
	int ndims, p1, p2;
	int dims[2], coords[2], cyclic[2], reorder;

	// 2D-grid, row and column communicators
	MPI_Comm proc_grid, proc_row, proc_col;

	// MPI request variables
	MPI_Request send_request, recv_request;

	// MPI status variable
	MPI_Status status;

	// Blocks local to each processor
	double *my_blockA, *my_blockB, *my_blockC, *tempA, *tempB;	

	// MPI datatype vector
	MPI_Datatype vec_type;

	// Set number of blocks in vertical and horizontal direction (equal)
	p1 = sqrt(nproc);
	p2 = sqrt(nproc);

	ndims = 2;
	dims[0] = p1;
	dims[1] = p2;
	cyclic[0] = 0;
	cyclic[1] = 0;
	reorder = 1;

	// Set sizes of blocks
	int r1 = n / p1;
	int r2 = n / p2;

	// Create Cartesian grid
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &proc_grid);
	
	// Set my coordinates in grid
	MPI_Cart_coords(proc_grid, rank, ndims, coords);

	// Set my rank in grid
	int my_rank;
	MPI_Cart_rank(proc_grid, coords, &my_rank);

	// Row and column communicator variables
	int row_rank, column_rank, row_size, column_size;

	// Create a communicator for each row
	MPI_Comm_split(proc_grid, coords[0], coords[1], &proc_row);
  	MPI_Comm_rank(proc_row, &row_rank);
  	MPI_Comm_size(proc_row, &row_size);

  	// Create a communicator for each column
	MPI_Comm_split(proc_grid, coords[1], coords[0], &proc_col);
  	MPI_Comm_rank(proc_col, &column_rank);
    MPI_Comm_size(proc_col, &column_size);

	// Create new vector type to contain blocks
	MPI_Type_vector(r1, r2, n, MPI_DOUBLE, &vec_type);
	MPI_Type_commit(&vec_type);

	// Result matrix
	double *matrixC;	

	// First process handles data generation
	if (rank == 0) {
		// Allocate for multiplication matrices A and B and result matrix C
		double *matrixA, *matrixB;
		matrixA = malloc(n * n * sizeof(double));
		matrixB = malloc(n * n * sizeof(double));
		matrixC = malloc(n * n * sizeof(double));

		int i, j;
		time_t t;
		srand(time(&t));

		// Fill matrices A and B with random numbers
		for (i = 0; i < n * n; i++) {
			matrixA[i] = (double) rand() / RAND_MAX;
			matrixB[i] = (double) rand() / RAND_MAX;
			//matrixA[i] = i;
			//matrixB[i] = i;
		}

		// printf("Matrix A:\n");
		// for (i = 0; i < n * n; i++) {
		// 	printf("%f, ", matrixA[i]);
		// }
		// printf("\n");
		// printf("Matrix B:\n");
		// for (i = 0; i < n * n; i++) {
		// 	printf("%f, ", matrixB[i]);
		// }
		// printf("\n");

		//t = timer();

		// Run through grid and distribute blocks
		for (i = 0; i < p1; i++) {
			for (j = 0; j < p2; j++) {
				int grid_rank;
				int loc[2] = {i, j};
				MPI_Cart_rank(proc_grid, loc, &grid_rank);
				MPI_Isend(&matrixA[i * r1 * n + j * r2], 1, vec_type, grid_rank, grid_rank, proc_grid, &send_request);
				MPI_Wait(&send_request, &status);
				MPI_Isend(&matrixB[i * r1 * n + j * r2], 1, vec_type, grid_rank, grid_rank, proc_grid, &send_request);
				MPI_Wait(&send_request, &status);
			}
		}
		// Free memory
		free(matrixA);
		free(matrixB);

	}

	// Allocate room for my blocks
	my_blockA = malloc(r1 * r2 * sizeof(double));
	my_blockB = malloc(r1 * r2 * sizeof(double));
	my_blockC = calloc(r1 * r2, sizeof(double));

	// Receive my blocks from first process
	MPI_Recv(my_blockA, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);
	MPI_Recv(my_blockB, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &status);

	// Allocate for intermediate storage
	tempA = malloc(r1 * r2 * sizeof(double));
	tempB = malloc(r1 * r2 * sizeof(double));

	if (rank == 0) {
		t = timer();
	}

	// Run Fox's algorithm
	int k, row, col, l;
	for (k = 0; k < p1; k++) {
		int m = (coords[0] + k) % p1;
		// Prepare sending my B block for the rotation
		MPI_Isend(my_blockB, r1 * r2, MPI_DOUBLE, (p1 + column_rank - 1) % p1, column_rank, proc_col, &send_request);
		if (row_rank == m) {
			// Broadcast diagonal block of A and multiply with B 
			MPI_Bcast(my_blockA, r1 * r2, MPI_DOUBLE, m, proc_row);
			for (row = 0; row < r1; row++) {
				for (col = 0; col < r2; col++) {
					for (l = 0; l < r2; l++) {
						my_blockC[row * r2 + col] += my_blockA[row * r2 + l] * my_blockB[l * r2 + col];
					}
				}
			}
		} else {
			// Broadcast other blocks of A and multiply with B
			MPI_Bcast(tempA, r1 * r2, MPI_DOUBLE, m, proc_row);
			for (row = 0; row < r1; row++) {
				for (col = 0; col < r2; col++) {
					for (l = 0; l < r2; l++) {
						my_blockC[row * r2 + col] += tempA[row * r2 + l] * my_blockB[l * r2 + col];
					}
				}
			}
		}

		// Receive transmission of B blocks within column communicators
		MPI_Irecv(tempB, r1 * r2, MPI_DOUBLE, (column_rank + 1) % p1, (column_rank + 1) % p1, proc_col, &recv_request);
		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		memcpy(my_blockB, tempB, r1 * r2 * sizeof(double));
	}

	// if (rank == 0) {
	// 	int i;
	// 	printf("my_blockC:\n");
	// 	for (i = 0; i < p1 * p2; i++) {
	// 		printf("%f, ", my_blockC[i]);
	// 	}
	// 	printf("\n");
	// }

	// Send my C block to first process
	MPI_Isend(my_blockC, r1 * r2, MPI_DOUBLE, 0, my_rank, proc_grid, &send_request);

	// Receive C blocks in first-come order
	if (rank == 0) {
		int i;
		for (i = 0; i < nproc; i++) {
			MPI_Status stat;
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, proc_grid, &stat);
			int loc[2];
			MPI_Cart_coords(proc_grid, stat.MPI_SOURCE, ndims, loc);
			MPI_Recv(&matrixC[loc[0] * r1 * n + loc[1] * r2], 1, vec_type, stat.MPI_SOURCE, stat.MPI_TAG, proc_grid, &status);
		}
	}
	MPI_Wait(&send_request, &status);

	if (rank == 0) {
		t = timer() - t;
		printf("Time: %f s\n", t);
	}

	// if (rank == 0) {
	// 	int i;
	// 	printf("Matrix C:\n");
	// 	for (i = 0; i < n * n; i++) {
	// 		printf("%f, ", matrixC[i]);
	// 	}
	// 	printf("\n");
	// }
	
	// Free memory
	free(my_blockA);
	free(my_blockB);
	free(my_blockC);
	free(tempA);
	free(tempB);

	// Close MPI
	MPI_Finalize();

	return 0;
}
