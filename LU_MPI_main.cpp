#include<stdio.h>
#include<iostream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include "mpi.h"

#define n 10000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, p, row_procs, col_procs, num_of_iter, local_mat_size,local_iter;
    int i,j,k,iter;
    int g_sizes[2], distribs[2], dargs[2], p_sizes[2], dims[2], periods[2], lsizes[2],start_indices[2];
    int cart_rank, coords[2];

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Status status;
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &p);

    //decomposing the domain across sqrt(p)Xsqrt(p) processes
    row_procs = col_procs = sqrt(p);

    //number of rows and columns assigned to each process
    int num_local_rows = n/row_procs;
    int num_local_cols = n/col_procs;
    int local_n = num_local_rows;//local size for each process

    //assigning memory to the local submatrices
    float *sub_matrix = (float *)malloc(num_local_rows*num_local_cols*sizeof(float));    
    
    //creating cartesian topology
    MPI_Comm cart;
    dims[0] = row_procs;
    dims[1] = col_procs;
    periods[0] = periods[1] = 0;   
    MPI_Cart_create(comm, 2, dims, periods, 0, &cart);

    //getting the rank and coordinates w.r.t cartesian topology
    MPI_Comm_rank(cart, &cart_rank);
    MPI_Cart_coords(cart, cart_rank, 2, coords);
    
    // MPI file read using MPI_File_read_all
    g_sizes[0] = g_sizes[1] = n;
    distribs[0] = distribs[1] = MPI_DISTRIBUTE_BLOCK;
    dargs[0] = dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;
    p_sizes[0] = row_procs;
    p_sizes[1] = col_procs;

    MPI_Datatype filetype;
    MPI_File fh;
    MPI_Type_create_darray(p, rank, 2, g_sizes, distribs, dargs, p_sizes, MPI_ORDER_C, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    
    MPI_File_open(comm, "matrix.txt", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    local_mat_size = num_local_rows * num_local_cols;
    MPI_File_read_all(fh, sub_matrix, local_mat_size, MPI_FLOAT, &status);
    
    MPI_File_close(&fh);

    //Gauss elimination / LU decomposition
    double start = MPI_Wtime();
    iter = 0;
    num_of_iter = n;    
    while(iter < n)
    {        
        local_iter = iter % local_n; //local iteration of the submatrices(diagonal submatrices)

        if(coords[0] == iter/local_n && coords[1] == iter/local_n) 
        {//diagonal submatrices
            int left_rank, right_rank, top_rank, bottom_rank;
            MPI_Request request_r,request_b;
            int flag_r, flag_b;
            MPI_Status status_r, status_b;                        
            
            // sending the row corresponding to local iteration to bottom process
            int sb_size = num_local_cols-local_iter;
            float *sb_buff = (float *)malloc(sb_size*sizeof(float)); 
            for(i=0;i<num_local_cols-local_iter;i++)
                sb_buff[i]=sub_matrix[(local_iter*num_local_cols)+(i+local_iter)];                       
            MPI_Cart_shift(cart, 0, 1, &top_rank, &bottom_rank);
            if(bottom_rank != MPI_PROC_NULL)
                MPI_Isend(sb_buff, sb_size, MPI_FLOAT,  bottom_rank, 0, cart, &request_b);//Carrying on with computation as the communication is happening

            //modifying the columns corresponding to current iteration by dividing with the daigonal element
            //this column will be stored in the lower triangular matrix
            k = (iter%local_n);
            for(j = k+1; j < num_local_rows; j++)        
                sub_matrix[(j*num_local_cols)+k] = sub_matrix[(j*num_local_cols)+k]/sub_matrix[(k*num_local_cols)+k];             

            //sending the column needed for reduction to right process
            int sr_size = num_local_rows-local_iter;
            float *sr_buff = (float *)malloc(sr_size*sizeof(float)); 
            for(i=0;i<num_local_rows-local_iter;i++)
                sr_buff[i]=sub_matrix[((i+local_iter)*num_local_cols)+local_iter];            
            MPI_Cart_shift(cart, 1, 1, &left_rank, &right_rank);
            if(right_rank != MPI_PROC_NULL)
                MPI_Isend(sr_buff, sr_size, MPI_FLOAT,  right_rank, 0, cart, &request_r); //Carrying on with computation as the communication is happening               

            //local reduction of the diagonal submatrix for the local iteration
            for(i = k+1; i < num_local_rows; i++)                
                for(j = k+1; j < num_local_cols; j++)            
                    sub_matrix[(i*num_local_cols)+j] -= sub_matrix[(k*num_local_cols)+j]*sub_matrix[(i*num_local_cols)+k];     

            //waiting for communication to complete     
            MPI_Test(&request_r, &flag_r, &status_r);
            MPI_Test(&request_b, &flag_b, &status_b);
            if(flag_r == 0 && right_rank != MPI_PROC_NULL)
            {                
                MPI_Wait(&request_r, &status_r);
            }
            if(flag_b == 0 && bottom_rank != MPI_PROC_NULL)
            {              
                MPI_Wait(&request_b, &status_b);
            }     
                                 
        }
        else if(coords[0] == iter/local_n && coords[1] >= (iter/local_n)+1)
        {//right processes
            int left_rank, right_rank, top_rank, bottom_rank, flag_r, flag_b;
            MPI_Request request_r, request_b;
            MPI_Status status_r, status_b;

            //receiving the buffer from the left process
            int rl_size = num_local_cols-local_iter;
            float *rl_buff = (float *)malloc(rl_size*sizeof(float)); 
            MPI_Cart_shift(cart, 1, 1, &left_rank, &right_rank);
            if(left_rank != MPI_PROC_NULL)
                MPI_Recv(rl_buff, rl_size, MPI_FLOAT, left_rank, 0, cart, &status);                

            //Forwarding the buffer to the right process
            if(right_rank != MPI_PROC_NULL)
                MPI_Isend(rl_buff, rl_size, MPI_FLOAT, right_rank, 0, cart, &request_r);
            
            //sending the row corresponding to the current iteration to the bottom process
            k = (iter%local_n);
            int sb_size = num_local_cols;
            float *sb_buff = (float *)malloc(sb_size*sizeof(float)); 
            for(i = 0; i < num_local_cols; i++)
                sb_buff[i]=sub_matrix[(k*num_local_cols)+i];                       
            MPI_Cart_shift(cart, 0, 1, &top_rank, &bottom_rank);
            if(bottom_rank != MPI_PROC_NULL)
                MPI_Isend(sb_buff, sb_size, MPI_FLOAT,  bottom_rank, 0, cart, &request_b);

            //local reduction of the submatrix
            for(i = k+1; i < num_local_rows; i++)                
                for(j = 0; j < num_local_cols; j++)    
                       sub_matrix[(i*num_local_cols)+j] -= rl_buff[i-k]*sub_matrix[(k*num_local_cols)+j];         
            
            //waiting for communication to complete
            MPI_Test(&request_r, &flag_r, &status_r);
            MPI_Test(&request_b, &flag_b, &status_b);
            if(flag_r == 0 && right_rank != MPI_PROC_NULL)
            {                
                MPI_Wait(&request_r, &status_r);
            }
            if(flag_b == 0 && bottom_rank != MPI_PROC_NULL)
            {              
                MPI_Wait(&request_b, &status_b);
            }                
        }
        else if(coords[1] == iter/local_n && coords[0] >= (iter/local_n)+1)
        {//bottom process
            int left_rank, right_rank, top_rank, bottom_rank, flag_r, flag_b;
            MPI_Request request_r, request_b;
            MPI_Status status_r, status_b;           

            //receiving the buffer from the top process
            int rt_size = num_local_cols-local_iter;
            float *rt_buff = (float *)malloc(rt_size*sizeof(float)); 
            MPI_Cart_shift(cart, 0, 1, &top_rank, &bottom_rank);
            if(top_rank != MPI_PROC_NULL)
                MPI_Recv(rt_buff, rt_size, MPI_FLOAT, top_rank, 0, cart, &status);           
            
            // Forwarding the buffer to the bottom process
            if(bottom_rank != MPI_PROC_NULL)
                MPI_Isend(rt_buff, rt_size, MPI_FLOAT, bottom_rank, 0, cart, &request_b);

            
            //modifying the column corresponding to the current iteration by dividing with the diagonal element received from the top process
            //this will be stored in lower triangular matrix
            k = iter % local_n;
            for(j = 0; j < num_local_rows; j++)        
                sub_matrix[(j*num_local_cols)+k] = sub_matrix[(j*num_local_cols)+k]/rt_buff[0]; 

            //sending the column corresponding to the local iteration to the right process            
            int sr_size = num_local_rows;
            float *sr_buff = (float *)malloc(sr_size*sizeof(float)); 
            for(i=0; i < num_local_rows; i++)
                sr_buff[i]=sub_matrix[(i*num_local_cols)+k];                       
            MPI_Cart_shift(cart, 1, 1, &left_rank, &right_rank);
            if(right_rank != MPI_PROC_NULL)
                MPI_Isend(sr_buff, sr_size, MPI_FLOAT,  right_rank, 1, cart, &request_r);

            //local reduction of the submatrix
            for(i = 0; i < num_local_rows; i++)                
                for(j = k+1; j < num_local_cols; j++)            
                    sub_matrix[(i*num_local_cols)+j] -= sub_matrix[(i*num_local_cols)+k]*rt_buff[j-k];

            //waiting for communication to complete
            MPI_Test(&request_r, &flag_r, &status_r);   
            MPI_Test(&request_b, &flag_b, &status_b);          
            if(flag_r == 0 && right_rank != MPI_PROC_NULL)
            {                
                MPI_Wait(&request_r, &status_r);
            }
            if(flag_b == 0 && bottom_rank != MPI_PROC_NULL)
            {              
                MPI_Wait(&request_b, &status_b);
            }            
        }
        else if(coords[0] > (iter/local_n) && coords[1] > (iter/local_n))        
        {//processes which are atleast 1 block away from the diagonal processes in both directions
            int left_rank, right_rank, top_rank, bottom_rank, flag_r, flag_b;
            MPI_Request request_r, request_b;
            MPI_Status status_r, status_b;

            k = iter % local_n;
            //receiving the buffer from the top process
            int rt_size = num_local_cols;
            float *rt_buff = (float *)malloc(rt_size*sizeof(float));                                  
            MPI_Cart_shift(cart, 0, 1, &top_rank, &bottom_rank);
            if(top_rank != MPI_PROC_NULL)
               MPI_Recv(rt_buff, rt_size, MPI_FLOAT, top_rank, 0, cart, &status);

            //receiving the buffer from the left process
            int rl_size = num_local_rows;
            float *rl_buff = (float *)malloc(rl_size*sizeof(float));                                 
            MPI_Cart_shift(cart, 1, 1, &left_rank, &right_rank);
            if(left_rank != MPI_PROC_NULL)
                MPI_Recv(rl_buff, rl_size, MPI_FLOAT, left_rank, 1, cart, &status); //tag is set to 1 because it is receiving from both top and left processes            
            
            //Forwarding the buffer to the bottom process
            if(bottom_rank != MPI_PROC_NULL)
                MPI_Isend(rt_buff, rt_size, MPI_FLOAT, bottom_rank, 0, cart, &request_b);

            //Forwarding the buffer to the right process
            if(right_rank != MPI_PROC_NULL)
                MPI_Isend(rl_buff, rl_size, MPI_FLOAT, right_rank, 1, cart, &request_r);

            //local reduction of the submatrix
            for(i = 0; i < num_local_rows; i++)                
                for(j = 0; j < num_local_cols; j++)            
                    sub_matrix[(i*num_local_rows)+j] -= rl_buff[i]*rt_buff[j];
            
            //waiting for communication to complete
            MPI_Test(&request_r, &flag_r, &status_r);   
            MPI_Test(&request_b, &flag_b, &status_b);          
            if(flag_r == 0 && right_rank != MPI_PROC_NULL)
            {                
                MPI_Wait(&request_r, &status_r);
            }
            if(flag_b == 0 && bottom_rank != MPI_PROC_NULL)
            {              
                MPI_Wait(&request_b, &status_b);
            }            
        }
        iter++;
    }    
    double end = MPI_Wtime();

    //timing the algorithm
    printf("time taken with %d processes: %e s\n", p, start-end);

    //writing the results ie., A=LU to a file using MPI_File_write_all
    MPI_Datatype filetypew;
    MPI_File fhw;
    MPI_Type_create_darray(p, rank, 2, g_sizes, distribs, dargs, p_sizes, MPI_ORDER_C, MPI_FLOAT, &filetypew);
    MPI_Type_commit(&filetypew);
    
    MPI_File_open(comm, "LU.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fhw);
    MPI_File_set_view(fhw, 0, MPI_FLOAT, filetypew, "native", MPI_INFO_NULL);
    local_mat_size = num_local_rows * num_local_cols;
    MPI_File_write_all(fhw, sub_matrix, local_mat_size, MPI_FLOAT, &status);
    
    MPI_File_close(&fhw);    
    
    
    MPI_Finalize();
    return 0;
}