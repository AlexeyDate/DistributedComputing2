#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void read_size(int *n_p, int my_rank, MPI_Comm comm)
{
    if (my_rank == 0)
        scanf("%d", n_p);

    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
}

void read_matrix(double *local_matrix, int block_size, int n, int my_rank, int grid_size, MPI_Comm comm)
{
    double *matrix = NULL;

    if (my_rank == 0)
    {
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
            for (int i = 0; i < grid_size; i++)
            {
                for (int k = shift_block * block_size; k < shift_block * block_size + block_size; k++)
                    scanf("%lf", &matrix[current_block * block_size * block_size + k]);
                current_block++;
            }
        }
    }

    MPI_Scatter(matrix, block_size * block_size, MPI_DOUBLE, local_matrix, block_size * block_size, MPI_DOUBLE, 0, comm);
    free(matrix);
}

void print_matrix(double *local_matrix, int block_size, int n, int my_rank, int grid_size, MPI_Comm comm)
{
    double *matrix = NULL;

    if (my_rank == 0)
    {
        matrix = (double *)malloc(n * n * sizeof(double));      

        MPI_Gather(local_matrix, block_size * block_size, MPI_DOUBLE, matrix, block_size * block_size, MPI_DOUBLE, 0, comm);

        int current_block = 0;
        int shift_block = 0;
        int last_block = 0;
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
                current_block++;
            }
            printf("]\n");
        }

        free(matrix);
    }
    else
        MPI_Gather(local_matrix, block_size * block_size, MPI_DOUBLE, matrix, block_size * block_size, MPI_DOUBLE, 0, comm);
}

void matrix_multiply(double *matrix_A, double *matrix_B, double *matrix_C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                matrix_C[i * n + j] += matrix_A[i * n + k] * matrix_B[k * n + j];
            }
        }
    }
}

void cannon_multiply(double *local_A, double *local_B, double *local_C, int block_size, int n, int my_rank, int grid_size, MPI_Comm comm)
{
    MPI_Comm grid_comm;
    // Arguments: (comm_old, ndims, dims, periods, reorder, OUT comm_cart)
    MPI_Cart_create(comm, 2, (int[]){grid_size, grid_size}, (int[]){1, 1}, 0, &grid_comm);

    int grid_coords[2];
    // Arguments: (comm_cart, rank, ndims, OUT coord)
    MPI_Cart_coords(grid_comm, my_rank, 2, grid_coords); // Getting coordinates of the process

    int shift_source[2], shift_dest[2];
    // Arguments: (comm_cart, direction, disp, OUT rank_source, OUT rank_dest)
    MPI_Cart_shift(grid_comm, 1, -grid_coords[0], &shift_source[1], &shift_dest[1]); // Initial alignmnet of A by columns
    MPI_Cart_shift(grid_comm, 0, -grid_coords[1], &shift_source[0], &shift_dest[0]); // Initial alignment of B by rows

    int count_recv = block_size * block_size;
    // Arguments: (buffer, count, MPI_DATATYPE, dest, sendtag, source, recvtag, comm, MPI_Status)
    MPI_Sendrecv_replace(local_A, count_recv, MPI_DOUBLE, shift_dest[1], 0, shift_source[1], 0, grid_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(local_B, count_recv, MPI_DOUBLE, shift_dest[0], 0, shift_source[0], 0, grid_comm, MPI_STATUS_IGNORE);

    // Cannon`s algorithm for matrix multiplication
    for (int i = 0; i < grid_size; i++)
    {
        matrix_multiply(local_A, local_B, local_C, block_size); // Local matrix multiplication

        MPI_Cart_shift(grid_comm, 1, -1, &shift_source[1], &shift_dest[1]); // Shift of matrix A to the left
        MPI_Cart_shift(grid_comm, 0, -1, &shift_source[0], &shift_dest[0]); // Shift of matrix B to the up

        MPI_Sendrecv_replace(local_A, count_recv, MPI_DOUBLE, shift_dest[1], 0, shift_source[1], 0, grid_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(local_B, count_recv, MPI_DOUBLE, shift_dest[0], 0, shift_source[0], 0, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Comm_free(&grid_comm);
}

int main()
{
    int my_rank;
    int comm_size;
    double start, finish;
    double time_elapsed, total_time_elapsed = 0.0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int grid_size = sqrt(comm_size);
    if (comm_size != grid_size * grid_size)
    {
        if (my_rank == 0)
            printf("\nError: number of process must be a perfect square.\n");
        MPI_Finalize();
        return 0;
    }

    int n;
    read_size (&n, my_rank, MPI_COMM_WORLD);

    int block_size = n / grid_size;

    if (n % grid_size != 0)
    {
        if (my_rank == 0)
            printf("\nError: matrix size must be multiple of the grid size.\n");
        MPI_Finalize();
        return 0;
    }

    double *local_matrix_A = (double *)malloc(block_size * block_size * sizeof(double)); // Input matrix A
    double *local_matrix_B = (double *)malloc(block_size * block_size * sizeof(double)); // Inpit matrix B
    double *local_matrix_C = (double *)calloc(block_size * block_size, sizeof(double)); // Output matrix C

    read_matrix(local_matrix_A, block_size, n, my_rank, grid_size, MPI_COMM_WORLD);
    read_matrix(local_matrix_B, block_size, n, my_rank, grid_size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    cannon_multiply(local_matrix_A, local_matrix_B, local_matrix_C, block_size, n, my_rank, grid_size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    time_elapsed = finish - start;
    MPI_Reduce(&time_elapsed, &total_time_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    print_matrix(local_matrix_C, block_size, n, my_rank, grid_size, MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("Total time: %lf\n", total_time_elapsed);

    free(local_matrix_A);
    free(local_matrix_B);
    free(local_matrix_C);

    MPI_Finalize();

    return 0;
}