#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void read_size(int *n_p, int my_rank)
{
    if (my_rank == 0)
        scanf("%d", n_p);

    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void read_vector(double *local_vector, int blockSize, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    double *vector = NULL;
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        int gridSize = sqrt(comm_size);

        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = n / gridSize;
            displacements[i] = current_offset % n;

            current_offset += recv_counts[i];
        }

        vector = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i++)
            scanf("%lf", &vector[i]);
    }

    MPI_Scatterv(vector, recv_counts, displacements, MPI_DOUBLE, local_vector, blockSize, MPI_DOUBLE, 0, comm);
    free(vector);
}

void read_matrix(double *local_matrix, int blockSize, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    double *matrix = NULL;
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        int gridSize = sqrt(comm_size);

        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = blockSize * blockSize;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        matrix = (double *)malloc(n * n * sizeof(double));
        int currentBlock = 0;
        int shiftBlock = 0;
        int lastBlock = 0;
        for (int i = 0; i < n; i++, shiftBlock++)
        {
            if (shiftBlock < blockSize)
                currentBlock = lastBlock;
            else
            {
                lastBlock = currentBlock;
                shiftBlock = 0;
            }
            for (int j = 0; j < gridSize; j++)
            {
                for (int k = shiftBlock * blockSize; k < shiftBlock * blockSize + blockSize; k++)
                    scanf("%lf", &matrix[currentBlock * blockSize * blockSize + k]);
                currentBlock++;
            }
        }
    }

    MPI_Scatterv(matrix, recv_counts, displacements, MPI_DOUBLE, local_matrix, blockSize * blockSize, MPI_DOUBLE, 0, comm);
    free(matrix);
}

void print_vector(double *local_vector, int blockSize, int n, int my_rank, int comm_size, MPI_Comm comm)
{
    double *vector = malloc(n * n * sizeof(double));
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        int gridSize = sqrt(comm_size);

        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = blockSize;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        MPI_Gatherv(local_vector, blockSize, MPI_DOUBLE, vector, recv_counts, displacements, MPI_DOUBLE, 0, comm);

        printf("\nThe result after multiplication matrix by vector (using splitting matrix by blocks):\n[ ");
        double currentElement;
        int offset = 0;
        for (int i = 0; i < gridSize; i++)
        {
            offset = i * gridSize * blockSize;
            for (int j = 0; j < blockSize; j++)
            {
                currentElement = 0;
                for (int k = offset; k < offset + blockSize * gridSize; k += blockSize)
                    currentElement += vector[k + j];

                printf("%.2lf ", currentElement);
            }
        }
        printf("]\n");
        free(vector);
    }
    else
    {
        MPI_Gatherv(local_vector, blockSize, MPI_DOUBLE, vector, recv_counts, displacements, MPI_DOUBLE, 0, comm);
    }
}

void mult_splitting_by_blocks(double *local_matrix, int blockSize, double *local_vector, double *local_out_vector)
{
    for (int i = 0; i < blockSize; i++)
    {
        local_out_vector[i] = 0;
        for (int j = 0; j < blockSize; j++)
            local_out_vector[i] += local_matrix[blockSize * i + j] * local_vector[j];
    }
}

int main()
{
    int my_rank;
    int comm_size;
    double start, finish, elapsed = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n;
    read_size(&n, my_rank);

    int gridSize = sqrt(comm_size);
    int blockSize = n / gridSize;

    if (n % gridSize != 0)
    {
        if (my_rank == 0)
            printf("\nError: matrix size must be multiple of the grid size.\n");
        MPI_Finalize();
        return 0;
    }

    if (comm_size != gridSize * gridSize)
    {
        if (my_rank == 0)
            printf("\nError: number of processes must be a perfect square.\n");
        MPI_Finalize();
        return 0;
    }

    double *local_matrix = (double *)malloc(blockSize * blockSize * sizeof(double));
    double *local_vector = (double *)malloc(blockSize * sizeof(double));
    double *local_out_vector = (double *)malloc(blockSize * sizeof(double));

    read_matrix(local_matrix, blockSize, n, my_rank, comm_size, MPI_COMM_WORLD);
    read_vector(local_vector, blockSize, n, my_rank, comm_size, MPI_COMM_WORLD);
    /*
        for (int i = 0; i < blockSize; i++)
            printf("VECTOR, Process: %d, element: %lf\n", my_rank, local_vector[i]);

        for (int i = 0; i < blockSize * blockSize; i++)
            printf("MATRIX, Process: %d, element: %lf\n", my_rank, local_matrix[i]);
    */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    mult_splitting_by_blocks(local_matrix, blockSize, local_vector, local_out_vector);

    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    double time_elapsed = finish - start;
    MPI_Reduce(&time_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("%lf\n", elapsed);
    }
    print_vector(local_out_vector, blockSize, n, my_rank, comm_size, MPI_COMM_WORLD);

    free(local_matrix);
    free(local_vector);
    free(local_out_vector);

    MPI_Finalize();

    return 0;
}