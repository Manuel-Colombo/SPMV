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
    
__global__ void partialproduct(float *values, int *columns, int *rows, float *x, float *output, int num_vals,int stride,int num_rows,int number_of_thread_utilized) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
      if(tid<number_of_thread_utilized){
        int firstpos = tid * stride;
          for (size_t z=0; z<stride; z++) {
                  atomicAdd(&(output[rows[firstpos+z]]), values[firstpos+z] * x[columns[firstpos+z]]);
          }
      } 
      if(tid==0){
          for (size_t z=stride*number_of_thread_utilized; z<num_vals-1; z++) {
                  atomicAdd(&(output[rows[z]]), values[z] * x[columns[z]]);
          }
      }     
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

    int main(int argc, const char * argv[]) {
        int BLOCKDIM = 1024;
        if (argc != 2) {
            printf("Usage: ./exec matrix_file");
            return 0;
        }
        
        int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
        float *values;

        const char *filename = argv[1];

        double start_cpu, end_cpu;
        double start_gpu,end_gpu;
        
        read_matrix(&row_ptr, &col_ind, &values, filename, &num_rows, &num_cols, &num_vals);
        //start  of CSR TO COO 
        int *coo_row_indices = (int *)malloc(num_vals * sizeof(int));
        int *coo_col_indices = (int *)malloc(num_vals * sizeof(int));
        float *coo_values = (float *)malloc(num_vals * sizeof(float));

        int coo_index = 0;

        for (int i = 0; i < num_rows; i++) {
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];
    
            for (int j = row_start; j < row_end; j++) {
                coo_row_indices[coo_index] = i;
                coo_col_indices[coo_index] = col_ind[j];
                coo_values[coo_index] = values[j];
                coo_index++;
            }
        }
        //finish  of CSR TO COO 

        
        float *x = (float *) malloc(num_rows * sizeof(float));  //input vector
        float *y_sw = (float *) malloc(num_rows * sizeof(float));   //output of cpu
        float *y_hw = (float *) malloc(num_rows * sizeof(float));   //output of gpu
        
        // Generate a random vector

        srand(time(NULL));
        for (int i = 0; i < num_rows; i++) {
            x[i] = (rand()%100)/(rand()%100+1); //the number we use to divide cannot be 0, that's the reason of the +1
        }
         // Compute in sw
        start_cpu = get_time();
        spmv_csr_sw(row_ptr, col_ind, values, num_rows, x, y_sw);
        end_cpu = get_time();
        printf("SPMV Time CPU: %.10lf\n", end_cpu - start_cpu);


        int stride = 10000;     //number of partial product for each thread
        int number_of_thread_utilized=num_vals/stride;
        //GPU pointer
        float *coo_values_d;
        int  *coo_col_indices_d, *coo_row_indices_d;
        float *x_d;
        float *y_d;
        
        //Allocate memory for GPU        
        CHECK(cudaMalloc(&x_d, num_rows * sizeof(float))); // allocate data for input on gpu
        CHECK(cudaMalloc(&y_d, num_rows * sizeof(float))); // allocate data for input on gpu
        CHECK(cudaMalloc(&coo_col_indices_d, num_vals * sizeof(int))); // allocate data for input on gpu
        CHECK(cudaMalloc(&coo_row_indices_d, num_vals * sizeof(int))); // allocate data for input on gpu
        CHECK(cudaMalloc(&coo_values_d, num_vals * sizeof(float))); // allocate data for input on gpu    
        CHECK(cudaMemset(y_d, 0, num_rows* sizeof(float)));          

        CHECK(cudaMemcpy(x_d, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // send input data to the gpu
        CHECK(cudaMemcpy(coo_col_indices_d, coo_col_indices,  num_vals * sizeof(int), cudaMemcpyHostToDevice)); // send input data to the gpu
        CHECK(cudaMemcpy(coo_row_indices_d, coo_row_indices, num_vals * sizeof(int), cudaMemcpyHostToDevice)); // send input data to the gpu
        CHECK(cudaMemcpy(coo_values_d, coo_values,  num_vals * sizeof(float), cudaMemcpyHostToDevice)); // send input data to the gpu
        
        start_gpu = get_time();
        dim3 blocksPerGrid((num_rows + BLOCKDIM - 1) / BLOCKDIM, 1, 1); // # of blocks = (51813503 + 1024 - 1)  / 1024 = 50600
        dim3 threadsPerBlock(BLOCKDIM, 1, 1);  //#of thread for blocks = 1024  #of thread = 50600 * 1024 = 51814400
        partialproduct<<<blocksPerGrid, threadsPerBlock>>>(coo_values_d, coo_col_indices_d, coo_row_indices_d, x_d, y_d, num_vals,stride,num_rows,number_of_thread_utilized);//kernel call
        CHECK_KERNELCALL();
        end_gpu = get_time();
        printf("SPMV Time GPU: %.10lf\n", end_gpu - start_gpu); 
        CHECK(cudaMemcpy(y_hw, y_d, num_rows * sizeof(float), cudaMemcpyDeviceToHost)); // send input data to the gpu

        // Free    
        free(row_ptr);
        free(col_ind);
        free(values);
        free(y_sw);
        free(coo_row_indices);
        free(coo_col_indices);
        free(coo_values);
        free(y_hw);
        //cudafree
        cudaFree(x_d);
        cudaFree(y_d);
        cudaFree(coo_col_indices_d);
        cudaFree(coo_row_indices_d);
        cudaFree(coo_row_indices_d);
        

        return 0;
    }
