#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void read_size(int *n_p, int my_rank)
{
    if (my_rank == 0)
        scanf("%d", n_p);

    MPI_Bcast(n_p, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void read_vector(double *local_vector, int local_m, int n, int m, int my_rank, int comm_size, MPI_Comm comm)
{
    double *vector = NULL;
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = (m / comm_size + (i < m % comm_size ? 1 : 0));
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        vector = (double *)malloc(m * sizeof(double));

        for (int i = 0; i < m; i++)
            scanf("%lf", &vector[i]);
    }

    MPI_Scatterv(vector, recv_counts, displacements, MPI_DOUBLE, local_vector, local_m, MPI_DOUBLE, 0, comm);
    free(vector);
}

void read_matrix(double *local_matrix, int local_m, int n, int m, int my_rank, int comm_size, MPI_Comm comm)
{
    double *matrix = NULL;
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = (m / comm_size + (i < m % comm_size ? 1 : 0)) * n;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        matrix = (double *)malloc(n * m * sizeof(double));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                scanf("%lf", &matrix[j * n + i]);
    }

    MPI_Scatterv(matrix, recv_counts, displacements, MPI_DOUBLE, local_matrix, local_m * n, MPI_DOUBLE, 0, comm);
    free(matrix);
}

void print_vector(double *local_vector, int local_m, int n, int m, int my_rank, int comm_size, MPI_Comm comm)
{
    double *vector = malloc(n * m * sizeof(double));
    int *displacements = (int *)malloc(comm_size * sizeof(int));
    int *recv_counts = (int *)malloc(comm_size * sizeof(int));

    if (my_rank == 0)
    {
        int current_offset = 0;
        for (int i = 0; i < comm_size; i++)
        {
            recv_counts[i] = (m / comm_size + (i < m % comm_size ? 1 : 0)) * n;
            displacements[i] = current_offset;
            current_offset += recv_counts[i];
        }

        MPI_Gatherv(local_vector, local_m * n, MPI_DOUBLE, vector, recv_counts, displacements, MPI_DOUBLE, 0, comm);

        printf("\nThe result after multiplication matrix by vector (using splitting matrix by columns):\n[ ");
        double current_element;
        for (int i = 0; i < n; i++)
        {
            current_element = 0;
            for (int j = 0; j < m; j++)
                current_element += vector[j * n + i];
            printf("%.2lf ", current_element);
        }
        printf("]\n");
        free(vector);
    }
    else
    {
        MPI_Gatherv(local_vector, local_m * n, MPI_DOUBLE, vector, recv_counts, displacements, MPI_DOUBLE, 0, comm);
    }
}

void mult_splitting_by_columns(double *local_matrix, int local_m, int n, double *local_vector, double *local_out_vector)
{
    for (int i = 0; i < local_m; i++)
        for (int j = 0; j < n; j++)
            local_out_vector[i * n + j] = local_matrix[i * n + j] * local_vector[i];
}

int main()
{
    int my_rank;
    int comm_size;
    double start, finish, elapsed = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n, m;
    read_size(&n, my_rank);
    read_size(&m, my_rank);

    int local_m = m / comm_size + (my_rank < m % comm_size ? 1 : 0);

    if (m < comm_size)
    {
        if (my_rank == 0)
            printf("\nError: number of processes must not be less matrix size.\n");
        MPI_Finalize();
        return 0;
    }

    double *local_matrix = (double *)malloc(local_m * n * sizeof(double));
    double *local_vector = (double *)malloc(local_m * sizeof(double));
    double *local_out_vector = (double *)malloc(local_m * n * sizeof(double));

    read_matrix(local_matrix, local_m, n, m, my_rank, comm_size, MPI_COMM_WORLD);
    read_vector(local_vector, local_m, n, m, my_rank, comm_size, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    mult_splitting_by_columns(local_matrix, local_m, n, local_vector, local_out_vector);

    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    double time_elapsed = finish - start;
    MPI_Reduce(&time_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("%lf\n", elapsed);
    }

    print_vector(local_out_vector, local_m, n, m, my_rank, comm_size, MPI_COMM_WORLD);

    free(local_matrix);
    free(local_vector);
    free(local_out_vector);

    MPI_Finalize();

    return 0;
}