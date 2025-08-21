#include <stdio.h>
#include "mpi.h"

int main (int argc, char *argv[]) {
    int rank, nranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    printf("Rank%d, nranks=%d\n", rank, nranks);

    MPI_Finalize();
    return 0;
}