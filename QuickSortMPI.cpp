#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define MAX_PROCS 32
#define MAX_PROBLEM_SIZE 65536


#pragma warning(disable:4996)
#pragma comment(linker, "/STACK:640000000")



void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);

        }   
    }
    swap(&arr[i + 1], &arr[high]);

    return (i + 1);
}

void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    int rank, root = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    FILE* fp;
    fp = fopen("results_quicksort.csv", "w");
    fprintf(fp, "Size,Threads,RUN,Time,Speedup,Efficiency\n");
    for (int i = 1024; i <= MAX_PROBLEM_SIZE; i *= 2) {
        
        int* test_data = (int*)malloc(i * sizeof(int));
        for (int j = 0; j < i; j++) {
            test_data[j] = j % i;

        }
        double seq_time = 0;

        for (int num_procs = 1; num_procs <= MAX_PROCS; num_procs *= 2)
        {
            
            for (int run_no = 1; run_no <= 3; run_no++) {

                //MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
                int data_size = i;
                int data_per_proc = data_size / num_procs;
                int* data = (int*)malloc(data_size * sizeof(int));
                int* local_data = (int*)malloc(data_per_proc * sizeof(int));
                double start_time, end_time;


                // Initialize the data on the root process
                if (rank == root) {

                    for (int j = 0; j < i; j++) {
                        data[j] = test_data[j];

                    }
                    start_time = MPI_Wtime();

                }

                // Scatter the data to the other processes
                MPI_Scatter(data, data_per_proc, MPI_INT, local_data, data_per_proc, MPI_INT, root, MPI_COMM_WORLD);

                // Sort the local data
                quicksort(local_data, 0, data_per_proc - 1);

                // Gather the sorted data back to the root process
                MPI_Gather(local_data, data_per_proc, MPI_INT, data, data_per_proc, MPI_INT, root, MPI_COMM_WORLD);

                // Print the sorted data on the root process

                printf("\n");
                if (rank == root) {

                    end_time = MPI_Wtime();
                    double time = end_time - start_time;

                    if (num_procs == 1) {
                        seq_time = time;
                    }
                    double speedup = seq_time / time;
                    double efficiency = speedup / num_procs;
                    printf("Array size : %d , process: %d, RUN : %d , execution time:%f, speedup: %f, efficiency: %f\n", i, num_procs, run_no, time, speedup, efficiency);
                    fprintf(fp, "%d,%d,%d,%f,%f,%f\n", i, num_procs, run_no, time, speedup, efficiency);



                }
             

                free(data);
                free(local_data);
                if (rank != 0) {
                    MPI_Finalize();
                    exit(0);
                }
            }

        }
  
    }
    MPI_Finalize();


    return 0;
}
