#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, const char *filename, int *num_rows, int *num_cols, int *num_vals) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int *row_ptr_t = (int *) malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *) malloc(*num_vals * sizeof(int));
    float *values_t = (float *) malloc(*num_vals * sizeof(float));
    
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *) malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++) {
        row_occurances[i] = 0;
    }
    
    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        
        row_occurances[row]++;
    }
    
    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++) {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);
    
    // Set the file position to the beginning of the file
    rewind(file);
    
    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++) {
        col_ind_t[i] = -1;
    }
    
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF) {
        row--;
        column--;
        
        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1) {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        i = 0;
    }
    
    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
}

// CPU implementation of SPMV using CSR, DO NOT CHANGE THIS
void spmv_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    for (int i = 0; i < num_rows; i++) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        y[i] = dotProduct;
    }
}

__global__ void spmv_csr_gpu(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_rows - 1) {
        float dotProduct = 0;
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            dotProduct += values[j] * x[col_ind[j]];
        }
        y[i] = dotProduct;
      if(i<5)
        printf("%d %f",i,y[i]);
    }

}

int main(int argc, const char * argv[]) { 

    //
    if (argc != 2) {
        printf("Usage: ./exec matrix_file");
        return 0;
    }
    
    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    
    const char *filename = argv[1];

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;
    
    double  delta_cpu;
    double  delta_gpu;

    read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
    
    float *x = (float *) malloc(num_rows * sizeof(float));
    float *y_sw = (float *) malloc(num_rows * sizeof(float));
    float *y_hw = (float *) malloc(num_rows * sizeof(float));
    
    // Generate a random vector

    srand(time(NULL));

    for (int i = 0; i < num_rows; i++) {
        x[i] = (rand()%100)/(rand()%100+1); //the number we use to divide cannot be 0, that's the reason of the +1
    }

    float *x_d;
    float *y_d;
    int *row_ptr_d;
    int *col_ind_d;
    float *values_d;
    int BLOCKDIM=1024;
    CHECK(cudaMalloc(&x_d, num_rows * sizeof(float))); // allocate data for input on gpu
    CHECK(cudaMalloc(&y_d, num_rows * sizeof(float))); // allocate data for input on gpu
    CHECK(cudaMalloc(&row_ptr_d, (num_rows+1) * sizeof(int))); // allocate data for input on gpu
    CHECK(cudaMalloc(&col_ind_d, num_vals * sizeof(int))); // allocate data for input on gpu
    CHECK(cudaMalloc(&values_d, num_vals * sizeof(float))); // allocate data for input on gpu    
    CHECK(cudaMemset(y_d, 0, num_rows));  

    CHECK(cudaMemcpy(x_d, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // send input data to the gpu
    CHECK(cudaMemcpy(row_ptr_d, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice)); // send input data to the gpu
    CHECK(cudaMemcpy(col_ind_d, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice)); // send input data to the gpu
    CHECK(cudaMemcpy(values_d, values, num_vals * sizeof(float), cudaMemcpyHostToDevice)); // send input data to the gpu
    
    printf("Rows : %d\n",num_rows);
    printf("Vals : %d\n",num_vals);

    // Compute in sw
    start_cpu = get_time();
    spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
    end_cpu = get_time();
    delta_cpu = end_cpu - start_cpu;
    // Print time
    printf("SPMV Time CPU: %.10lf\n", delta_cpu);
    start_gpu = get_time();
    dim3 blocksPerGrid((num_rows + BLOCKDIM - 1) / BLOCKDIM, 1, 1); // we schedule a number of blocks to equally distribute the load
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);                      // num of threads is equal to the specified input
    spmv_csr_gpu<<<blocksPerGrid, threadsPerBlock>>>(row_ptr_d, col_ind_d, values_d, num_rows, x_d, y_d); // run the gpu kernel
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize()); // sync to take only kernel time
    end_gpu = get_time();
    delta_gpu = end_gpu - start_gpu;
    printf("SPMV Time GPU: %.10lf\n", delta_gpu);
    CHECK(cudaMemcpy(y_hw, y_d, num_rows * sizeof(float), cudaMemcpyDeviceToHost)); // send input data to the gpu    
    // Free    
    free(row_ptr);
    free(col_ind);
    free(values);
    free(y_sw);
    
    return 0;
}
