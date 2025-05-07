#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>

using namespace std;

void readMatrix(const string& filename, double*& matrix, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    file >> rows >> cols;
    matrix = new double[rows * cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i * cols + j];
    file.close();
}

void writeMatrix(const string& filename, double*& matrix, int& rows, int& cols) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return;
    }
    file << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            file << fixed << setprecision(2) << matrix[i * cols + j] << " ";
        file << "\n";
    }
    file.close();
}

void multiplyLocal(const double* A, const double* B, double* C, int blockSize) {
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            for (int k = 0; k < blockSize; ++k)
                C[i * blockSize + j] += A[i * blockSize + k] * B[k * blockSize + j];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " matrixA.txt matrixB.txt output.txt" << endl;
        MPI_Finalize();
        return 1;
    }

    double* A = nullptr;
    double* B = nullptr;
    int A_rows = 0, A_cols = 0, B_rows = 0, B_cols = 0;

    if (rank == 0) {
        readMatrix(argv[1], A, A_rows, A_cols);
        readMatrix(argv[2], B, B_rows, B_cols);

        if (A_cols != B_rows) {
            cerr << "Matrix dimension mismatch!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&A_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q = sqrt(size);
    if (q * q != size || A_rows % q != 0 || B_cols % q != 0 || A_cols % q != 0) {
        if (rank == 0)
            cerr << "Number of processes must be perfect square and matrix sizes divisible by sqrt(p)" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int blockSizeA_row = A_rows / q;
    int blockSizeB_col = B_cols / q;
    int blockSizeCommon = A_cols / q;

    int dims[2] = { q, q }, periods[2] = { 0, 0 }, coords[2];
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int row = coords[0], col = coords[1];

   
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, row, col, &row_comm);
    MPI_Comm_split(grid_comm, col, row, &col_comm);

   
    vector<double> A_block(blockSizeA_row * blockSizeCommon);
    vector<double> B_block(blockSizeCommon * blockSizeB_col);
    vector<double> C_block(blockSizeA_row * blockSizeB_col, 0.0);

    
    if (rank == 0) {
        for (int r = 0; r < q; ++r) {
            for (int c = 0; c < q; ++c) {
                int dest_rank;
                int coords_temp[2] = { r, c };
                MPI_Cart_rank(grid_comm, coords_temp, &dest_rank);

                if (dest_rank == 0) {
                    for (int i = 0; i < blockSizeA_row; ++i)
                        for (int j = 0; j < blockSizeCommon; ++j)
                            A_block[i * blockSizeCommon + j] = A[(r * blockSizeA_row + i) * A_cols + (j + c * blockSizeCommon)];

                    for (int i = 0; i < blockSizeCommon; ++i)
                        for (int j = 0; j < blockSizeB_col; ++j)
                            B_block[i * blockSizeB_col + j] = B[(i + r * blockSizeCommon) * B_cols + (c * blockSizeB_col + j)];
                }
                else {
                    vector<double> tempA(blockSizeA_row * blockSizeCommon);
                    vector<double> tempB(blockSizeCommon * blockSizeB_col);

                    for (int i = 0; i < blockSizeA_row; ++i)
                        for (int j = 0; j < blockSizeCommon; ++j)
                            tempA[i * blockSizeCommon + j] = A[(r * blockSizeA_row + i) * A_cols + (j + c * blockSizeCommon)];

                    for (int i = 0; i < blockSizeCommon; ++i)
                        for (int j = 0; j < blockSizeB_col; ++j)
                            tempB[i * blockSizeB_col + j] = B[(i + r * blockSizeCommon) * B_cols + (c * blockSizeB_col + j)];

                    MPI_Send(tempA.data(), tempA.size(), MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(tempB.data(), tempB.size(), MPI_DOUBLE, dest_rank, 1, MPI_COMM_WORLD);
                }
            }
        }
    }
    else {
        MPI_Recv(A_block.data(), A_block.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_block.data(), B_block.size(), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    vector<double> A_temp(blockSizeA_row * blockSizeCommon);
    vector<double> B_temp(blockSizeCommon * blockSizeB_col);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    // Algoritm SUMMA
    for (int k = 0; k < q; ++k) {
        if (col == k) A_temp = A_block;
        MPI_Bcast(A_temp.data(), A_temp.size(), MPI_DOUBLE, k, row_comm);

        if (row == k) B_temp = B_block;
        MPI_Bcast(B_temp.data(), B_temp.size(), MPI_DOUBLE, k, col_comm);

        multiplyLocal(A_temp.data(), B_temp.data(), C_block.data(), blockSizeCommon);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;


    double* C = nullptr;
    if (rank == 0)
        C = new double[A_rows * B_cols];

    for (int r = 0; r < q; ++r) {
        for (int c = 0; c < q; ++c) {
            int source_rank;
            int coords_temp[2] = { r, c };
            MPI_Cart_rank(grid_comm, coords_temp, &source_rank);

            if (rank == 0) {
                if (source_rank == 0) {
                    for (int i = 0; i < blockSizeA_row; ++i)
                        for (int j = 0; j < blockSizeB_col; ++j)
                            C[(r * blockSizeA_row + i) * B_cols + c * blockSizeB_col + j] = C_block[i * blockSizeB_col + j];
                }
                else {
                    vector<double> tempC(blockSizeA_row * blockSizeB_col);
                    MPI_Recv(tempC.data(), tempC.size(), MPI_DOUBLE, source_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int i = 0; i < blockSizeA_row; ++i)
                        for (int j = 0; j < blockSizeB_col; ++j)
                            C[(r * blockSizeA_row + i) * B_cols + c * blockSizeB_col + j] = tempC[i * blockSizeB_col + j];
                }
            }
            else if (rank == source_rank) {
                MPI_Send(C_block.data(), C_block.size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    if (rank == 0) {
        writeMatrix(argv[3], C, A_rows, B_cols);
        cout << "Time taken : " << elapsed.count() << " seconds" << endl;
        delete[] C;
        delete[] A;
        delete[] B;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}
