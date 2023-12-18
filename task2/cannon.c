#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void read_size(int *n_p, int my_rank, MPI_Comm comm)
{
    if (my_rank == 0)
        scanf("%d", n_p);

    MPI_Bcast(n_p, 1, MPI_DOUBLE, 0, comm);
}

void read_matrix(double *local_matrix, int block_size, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    double *matrix = NULL;
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));
    
    if (my_rank == 0)
    {
        int current_offset = 0;
        int grid_size = sqrt(comm_size);

        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = block_size * block_size;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        matrix = (double *)malloc(n * n * sizeof(double));
        int current_block = 0;
        int shift_block = 0;
        int last_block = 0;
        for (int i = 0; i < n; i++, shift_block++)
        {
            if (shift_block < block_size)
                current_block = last_block;
            else
            {
                last_block = current_block;
                shift_block = 0;
            }
            for (int j = 0; j < grid_size; j++)
            {
                for (int k = shift_block * block_size; k < shift_block * block_size + block_size; k++)
                    scanf("%lf", &matrix[current_block * block_size * block_size + k]);
                current_block++;
            }
        }
    }

    MPI_Scatterv(matrix, recv_counts, displacements, MPI_DOUBLE, local_matrix, block_size * block_size, MPI_DOUBLE, 0, comm);
    free(matrix);
}

void print_matrix(double *local_C, int block_size, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    double *matrix = malloc(n * n * sizeof(double));
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(double));

    if (my_rank == 0)
    {
        int current_offset = 0;
        int grid_size = sqrt(comm_size);

        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = block_size * block_size;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        MPI_Gatherv(local_C, block_size * block_size, MPI_DOUBLE, matrix, recv_counts, displacements, MPI_DOUBLE, 0, comm);
        
        printf("\nThe result after matrix multiplication (using Cannon`s algorithm):\n");

        int current_block = 0;
        int last_block = 0;
        int shift_block = 0;
        int offset;
        for (int i = 0; i < n; i++, shift_block++)
        {
            if (shift_block < block_size)
                current_block = last_block;
            else
            {
                last_block = current_block;
                shift_block = 0;
            }

            printf("[ ");
            offset = shift_block * block_size;
            for (int j = 0; j < grid_size; j++)
            {
                for (int k = offset; k < offset + block_size; k++)
                    printf("%.2lf ", matrix[current_block * block_size * block_size + k]);
            }
            printf("]\n");
        }
        free(matrix);
    
    }
    else
        MPI_Gatherv(local_C, block_size * block_size, MPI_DOUBLE, matrix, recv_counts, displacements, MPI_DOUBLE, 0, comm);
    
}

void matrix_multiply(double* matrix_A, double* matrix_B, double* matrix_C, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            matrix_C[i * size + j] = 0.0;
            for (int k = 0; k < size; k++) 
            {
                matrix_C[i * size + j] += matrix_A[i * size + k] * matrix_B[k * size + j];
            }
        }
    }
}

void cannon_multiply(double* local_A, double* local_B, double* local_C, int block_size, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    int grid_rank, grid_coords[2];
    MPI_Comm grid_comm;
    MPI_Cart_create(comm, 2, (int[]){sqrt(comm_size), sqrt(comm_size)}, (int[]){1, 1}, 0, &grid_comm);
    MPI_Comm_rank(grid_comm, &grid_rank);
    MPI_Cart_coords(grid_comm, grid_rank, 2, grid_coords);

    int grid_size = sqrt(comm_size);

    // Initialize the shift values
    int shift_source[2], shift_dest[2];
    MPI_Cart_shift(grid_comm, 0, grid_coords[0], &shift_source[0], &shift_dest[0]);
    MPI_Cart_shift(grid_comm, 1, grid_coords[1], &shift_source[1], &shift_dest[1]);

    // Perform the initial alignment of A and B
    MPI_Cart_shift(grid_comm, 0, -grid_coords[1], &shift_source[0], &shift_dest[0]);
    MPI_Cart_shift(grid_comm, 1, -grid_coords[0], &shift_source[1], &shift_dest[1]);

    MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE, shift_dest[1], 0, shift_source[1], 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE, shift_dest[0], 0, shift_source[0], 0, grid_comm, MPI_STATUS_IGNORE);

    // Cannon's algorithm for matrix multiplication
    for (int i = 0; i < grid_size; i++) {
        // Local matrix multiplication
        matrix_multiply(local_A, local_B, local_C, block_size);

        // Shift A left and B up
        MPI_Cart_shift(grid_comm, 0, -1, &shift_source[1], &shift_dest[1]);
        MPI_Cart_shift(grid_comm, 1, -1, &shift_source[0], &shift_dest[0]);

        MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE, shift_dest[1], 0, shift_source[1], 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE, shift_dest[0], 0, shift_source[0], 0, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Comm_free(&grid_comm);
}

int main()
{
    int my_rank;
    int comm_size;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n;
    read_size (&n, my_rank, MPI_COMM_WORLD);

    int grid_size = sqrt(comm_size);
    int block_size = n / grid_size;

    if (n % grid_size != 0)
    {
        if (my_rank == 0)
            printf("\nError: matrix size must be multiple of the grid size.\n");
        MPI_Finalize();
    }

    if (comm_size != grid_size * grid_size)
    {
        if (my_rank == 0)
            printf("\nError: number of process must be a perfect square.\n");
        MPI_Finalize();
        return 0;
    }

    double *local_A = (double *)malloc(block_size * block_size * sizeof(double)); // Input matrix A
    double *local_B = (double *)malloc(block_size * block_size * sizeof(double)); // Input matrix B
    double *local_C = (double *)malloc(block_size * block_size * sizeof(double)); // Output matrix C

    read_matrix(local_A, block_size, n, my_rank, comm_size, MPI_COMM_WORLD);
    read_matrix(local_B, block_size, n, my_rank, comm_size, MPI_COMM_WORLD);
    cannon_multiply(local_A, local_B, local_C, block_size, grid_size, my_rank, comm_size, MPI_COMM_WORLD);
    print_matrix(local_C, block_size, n, my_rank, comm_size, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}