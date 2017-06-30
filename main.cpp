#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#define MASTER 0
#define MATRIX_SIZE 2048



void matrixMult();
void createGrid(MPI_Comm *com,MPI_Comm *rowCom,MPI_Comm *colCom, int rank, int *coords,int gridSize);
void initBlocks(int blockSize,double *&A,double *&B, double *&C, double *&T,int *coords);
void deleteBlocks(int blockSize,double *&A,double *&B, double *&C, double *&T);
void blockSendRow(int* coords, int gridSize, int iter, double *&A, double *&T, MPI_Comm &rowCom);
void blockSendCol(int* coords, int gridSize, double *&B, MPI_Comm &colCom);
void multBlocks(double *&T,double *&B,double *&C,int blockSize);

inline int getRank(){int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);return rank;}

inline void printMx(double *&mx,int size){
    int idx;
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            idx = i*size+j;
            printf("%lf ",mx[idx]);
        }
        printf("\n");
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
        matrixMult();
    MPI_Finalize();

    return 0;
}

void matrixMult(){
    int rank,procCnt,gridSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCnt);
    gridSize = (int)sqrt((double)procCnt);
    if(rank==MASTER){
        if(procCnt!= gridSize*gridSize){
            printf("Error.Process count is not perfect square!\n");
            return;
        }
    }

    MPI_Comm gridCom, rowCom,colCom;
    int coords[2];
    createGrid(&gridCom,&rowCom,&colCom,rank,coords,gridSize);

    const int blockSize = MATRIX_SIZE/gridSize;
    double *A,*B,*C,*T,start,end;
    initBlocks(blockSize,A,B,C,T,coords);
    if(rank==MASTER) start=MPI_Wtime();
    for (int i = 0; i < gridSize; i++)
    {
        blockSendRow(coords,gridSize,i,A,T,rowCom);
        MPI_Barrier(MPI_COMM_WORLD);
        multBlocks(T,B,C,blockSize);
        blockSendCol(coords,gridSize,B,colCom);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==MASTER){
        end=MPI_Wtime();
        double delta = end-start;
        printf("Matrix mult size =%d finished in %lfs by %d processes.",MATRIX_SIZE,delta,procCnt);
    }
    //printf("\n\n\nProc[%d]\n",rank);
    //printMx(C,blockSize);

    deleteBlocks(blockSize,A,B,C,T);
    return;
}

void createGrid(MPI_Comm *com,MPI_Comm *rowCom,MPI_Comm *colCom, int rank, int *coords,int gridSize)
{
    int dimSize[2] = {gridSize, gridSize};
    int periodic[2] = {0,0};
    int subdims[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, periodic, 0, com);
    MPI_Cart_coords(*com, rank, 2, coords);
    subdims[0] = 0;
    subdims[1] = 1;
    MPI_Cart_sub(*com, subdims, rowCom);
    subdims[0] = 1;
    subdims[1] = 0;
    MPI_Cart_sub(*com, subdims, colCom);
    MPI_Barrier(MPI_COMM_WORLD);
}


void initBlocks(int blockSize,double *&A,double *&B, double *&C, double *&T,int *coords){
    int sq = blockSize*blockSize;
    A=new double[sq];
    B=new double[sq];
    C=new double[sq];
    T=new double[sq];
    double val;
    int idx;
    memset(C,0,sq*sizeof(double));
    double N =MATRIX_SIZE;
    const double diag = (N-2.0)/N;
    const double nondiag = -2.0/N;
    if(coords[0]==coords[1]){//Это блоки по диагонали
        for(int i=0;i<blockSize;i++){
            for(int j=0;j<blockSize;j++){
                if(i==j) val=diag;
                else val=nondiag;
                idx = i*blockSize+j;
                A[idx]=val;
                B[idx]=val;

            }
        }
    }
    else{
        for(int i=0;i<blockSize;i++){
            for(int j=0;j<blockSize;j++){
                idx = i*blockSize+j;
                A[idx]=nondiag;
                B[idx]=nondiag;

            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

//для каждой строки i (i = 0, …, q – 1) блок Aij одного из процессов пересылается во все процессы
//этой же строки; при этом индекс j пересылаемого блока определяется по формуле j = (i + m) mod q;
void blockSendRow(int *coords, int gridSize, int iter, double *&A, double *&T, MPI_Comm &rowCom) {
    int p = (coords[0] + iter) % gridSize;
    int blockSize = MATRIX_SIZE/gridSize;

    if (coords[1] == p)
        for (int i = 0; i < blockSize*blockSize; i++) T[i] = A[i];
        MPI_Bcast(T, blockSize * blockSize, MPI_DOUBLE, p, rowCom);
        //printf("Block[%d][%d] is sent\n",coords[0],coords[1]);
}


//полученный в результате подобной пересылки блок матрицы A и содержащийся в процессе (i, j)
//блок матрицы B перемножаются, и результат прибавляется к матрице Сij;
void multBlocks(double *&T, double *&B, double *&C, int blockSize) {
    int i,j,k;
//#pragma omp parallel for private(j,i)
    for (k = 0; k< blockSize; k++) {
        for (i = 0; i < blockSize; i++) {
            for (j = 0; j < blockSize; j++)
                C[i*blockSize+j] += T[i*blockSize+k] * B[k*blockSize+j];
        }
    }
//#pragma omp parallel for private(j,k,t)

}
//для каждого столбца j (j = 0, …, q – 1) выполняется циклическая пересылка блоков матрицы B, со-
//держащихся в каждом процессе (i, j) этого столбца, в направлении убывания номеров строк.
void blockSendCol(int *coords, int gridSize, double *&B, MPI_Comm &colCom) {
    MPI_Status status;
    int next,prev,blockSize = MATRIX_SIZE/gridSize;
    next = (coords[0] + 1) % gridSize;
    prev = (coords[0] + gridSize-1) % gridSize;
    MPI_Sendrecv_replace(B, blockSize*blockSize, MPI_DOUBLE, prev, 0, next, 0, colCom, &status);
}

void deleteBlocks(int blockSize, double *&A, double *&B, double *&C, double *&T) {
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] T;
}
