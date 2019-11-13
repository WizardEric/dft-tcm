/* //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Author: Eric Qin
// We should be using 8 MPI ranks when testing the performance of the code. 

	- // MPI_Send (void * data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator)
	- // MPI_Recv(void * data, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm communicator, MPI_Status* status)
	- https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
	- http://www.cplusplus.com/reference/complex/

COMMANDS
module load gcc/4.9.0 ; module load cmake/3.9.1; module load openmpi
cd .. ; rm -rf build ; mkdir build ; cp Tower256.txt build/ ; cd build ; cmake .. ; make ; mpirun -np 8 ./p32 reverse Tower256.txt Output256.txt
cd .. ; rm -rf build ; mkdir build ; cp Tower1024.txt build/ ; cd build ; cmake .. ; make ; mpirun -np 8 ./p32 reverse Tower1024.txt Output1024.txt
*/ //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
// #include <complex>
#include <valarray>
#include <chrono>
#include <complex.h>
#include <complex.cc>
#include <input_image.cc>
#include <input_image.h>

// const double PI = 3.14159265358979323846;

using namespace std;
// typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

void computeFFT(CArray& block);

int main(int argc, char *argv[]) {
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BASIC MPI COMMANDS
	// Initialize the MPI environment
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process (individual process ID)
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// Print off a hello world message
	//printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gather and Process User Arguments

	Complex * grid, * row, * column;
	int dimensionX, dimensionY;
	auto start = std::chrono::system_clock::now();

	// Have rank 0 processor be the root node
	if (world_rank == 0) {

		// cout << "Final Project with MPI\n";
		// cout << "You have entered " << argc-1 << " arguments:\n";
		// cout << "Direction: " << argv[1] << "\n";
		// cout << "Input Filename: " << argv[2] << "\n";
		// cout << "Output Filename: " << argv[3] << "\n";

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Reading First line of Tower Text File
		std::string configName = argv[2];
		string line;

		int count = 0;
		int spaceID = 0;
		string dimension;
		int dimensionX, dimensionY;
		ifstream readFile;
		readFile.open(configName);

		while (!readFile.eof()) {
			getline(readFile, line);
			if (count == 0) {
				dimension = line;
				int spaceID = line.find(' ');
				dimensionX = stoi(line.substr(0,spaceID));
				dimensionY = stof(line.substr(spaceID+1, line.find('\0')));
			}
			count++;
		}
		readFile.close();

		// cout << dimension << "\n";
		// cout << dimensionX << "\n";
		// cout << dimensionY << "\n";

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Allocate Dynamic Memory and store values within Tower Text File
		Complex * grid = new Complex[dimensionX*dimensionY]();
		Complex * buffergrid = new Complex[dimensionX*dimensionY]();

		count = 0; // line count
		spaceID = 0; // space ID
		int prevspaceID = 0; // previous space ID
		int subID = 0; // substitue ID for grid index
		int tempID = 0; // temp index to determine line value

		
    	InputImage img = InputImage(argv[2]);
		int rows = img.get_height();
		int columns = img.get_width();
   		grid = img.get_image_data();
		dimensionX = columns;
		dimensionY = rows;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Spread to other Processors to Compute using MPI Functions

		Complex * row = new Complex[dimensionX]();
		Complex * column = new Complex[dimensionX]();

		// When the number of processors equal the described amount
		if (world_size == 8) {
			// cout << "Num cores = 8 \n" ;

			int row_divide = dimensionY/8;
			int col_divide = row_divide;
			int fftD = 2;

   	 		// If we are the root process, send our data to everyone
   			for (int i = 0; i < world_size; i++) {
     				if (i != world_rank) {
        				MPI_Send(
						&dimensionX, 
						1, 
						MPI_INT, 
						i, 
						0, 
						MPI_COMM_WORLD);
				}
   			 }

			// 2D FFT
			for (int fft = 0 ; fft < fftD ; fft++) {
				// compute FFT on the first few rows of the grid
				for (int y = 0 ; y < row_divide ; y++) {
					for (int x = 0 ; x < dimensionX ; x++) {
						row[x] = grid[y*dimensionX+x];
					}
					CArray rowblock(row, dimensionX);
					computeFFT(rowblock);

					for (int x = 0 ; x < dimensionX ; x++) {
						buffergrid[y*dimensionX+x] = rowblock[x];
					}
				}

				// Loop through 7/8 of rows to broadcast to the other processors (scatter)
				int core_select = 1;
				int row_divide_counter = 0;
				for (int y = row_divide ; y < dimensionX; y++) {
				//cout << core_select << '\n';
					for (int x = 0 ; x < dimensionX ; x++){
						MPI_Send(
							&grid[y*dimensionX+x], 
							1, 
							MPI_COMPLEX, 
							core_select, 
							y*dimensionX+x, 
							MPI_COMM_WORLD);
		      			}

					row_divide_counter++;

					if (row_divide_counter % row_divide == 0) {
						core_select++;
					}
				}

				// Recieve the FFTs from the slave processors and combine it to buffergrid
				int core_recieve = 1;
				row_divide_counter = 0;
				for (int y = row_divide; y < dimensionX; y++) {
					for (int x = 0; x < dimensionX; x++) {
						MPI_Recv(
							&buffergrid[y*dimensionX+x], 
							1,
							MPI_COMPLEX, 
							core_recieve, 
							y*dimensionX+x, 
							MPI_COMM_WORLD,
					     		MPI_STATUS_IGNORE);
					}

					row_divide_counter++;

					if (row_divide_counter % row_divide == 0) {
						core_recieve++;
					}
				}

				// Transpose Matrix
				for (int y = 0 ; y < dimensionY ; y++) {
					for (int x = 0 ; x < dimensionX ; x++) {
						grid[x+y*dimensionX] = buffergrid[x*dimensionX+y];
					}
				}

				for (int y = 0 ; y < dimensionY ; y++) {
					for (int x = 0 ; x < dimensionX ; x++) {
						buffergrid[x+y*dimensionX] = grid[x+y*dimensionX];
					}
				}
			}
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// To do this serially
		} else {
			// cout << "Num cores not 8, running serially to output correct results \n";

			// FFT on Row
			for (int y = 0 ; y < dimensionY ; y++) {
				for (int x = 0 ; x < dimensionX ; x++) {
					row[x] = grid[y*dimensionX+x];
				}
				CArray rowblock(row, dimensionX);
				computeFFT(rowblock);

				for (int x = 0 ; x < dimensionX ; x++) {
					buffergrid[y*dimensionX+x] = rowblock[x];
				}
			}

			// FFT on Column
			for (int x = 0; x < dimensionX ; x++) {
				for (int y = 0; y < dimensionY; y++) {
					column[y] = buffergrid[y*dimensionX+x];
				}
				CArray colblock(column, dimensionY);
				computeFFT(colblock);

				for (int y = 0 ; y < dimensionY ; y++) {
					buffergrid[y*dimensionX+x] = colblock[y];
				}
			}
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Print Data and Timing into Text File
		// auto end = std::chrono::system_clock::now();
		// std::chrono::duration<double> elapsed = end - start;
		// ofstream myfile;
  		// myfile.open ("timing.csv");
   		// myfile << elapsed.count();
   		// myfile.close();

		
    	img.save_image_data(argv[3], buffergrid, columns, rows);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		
		cout << "Runtime in seconds: " << elapsed.count() << "s";
		// cout << "Program finished\n";

		delete [] grid;
		delete [] buffergrid;
		delete [] row;
		delete [] column;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Other processors node when number of processors match correctly
	} else if (world_size == 8)  {

		// Recieve dimension size
		MPI_Recv(
			&dimensionX, 
			1,
			MPI_INT, 
			0, 
			0, 
			MPI_COMM_WORLD,
             		MPI_STATUS_IGNORE);

		int row_divide = dimensionX/8;
		int fftD = 2;

		// truncated grid memory, each processor creates its own (Buffers for Scatter)
		Complex * gridcopy = new Complex[row_divide*dimensionX]();
		Complex * buffer = new Complex[row_divide*dimensionX]();
		Complex * row = new Complex[dimensionX]();
		Complex * column = new Complex[dimensionX]();

		// 2D FFT
		for (int fft = 0; fft < fftD; fft++) {
			for (int y = 0; y < row_divide; y++) {
				for (int x = 0; x < dimensionX; x++) {
					MPI_Recv(
						&gridcopy[y*dimensionX+x], 
						1,
						MPI_COMPLEX, 
						0, 
						(world_rank*row_divide+y)*dimensionX+x, 
						MPI_COMM_WORLD,
			     			MPI_STATUS_IGNORE);
				}
			}

			// conduct row wise FFT
			for (int y = 0 ; y < row_divide ; y++) {
				for (int x = 0 ; x < dimensionX ; x++) {
					row[x] = gridcopy[y*dimensionX+x];
				}

				CArray rowblock(row, dimensionX);
				computeFFT(rowblock);

				for (int x = 0 ; x < dimensionX ; x++) {
					buffer[y*dimensionX+x] = rowblock[x];
				}
			} 

			// send back to node processor
			for (int y = 0; y < row_divide; y++) {
				for (int x = 0; x < dimensionX; x++) {
					//cout << "sending " << ((world_rank*row_divide+y)*dimensionX+x) << '\n';
					MPI_Send(
						&buffer[y*dimensionX+x], 
						1, 
						MPI_COMPLEX, 
						0, 
						(world_rank*row_divide+y)*dimensionX+x, 
						MPI_COMM_WORLD);
				}
			} 
		}	

		delete [] gridcopy;
		delete [] buffer;
		delete [] row;
		delete [] column;
		
	}

	// Finalize MPI environment
	MPI_Finalize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Cooley- Turkey FFT
// Reference:https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
void computeFFT(CArray& block){ 
	const size_t N = block.size();

	// return logic
	if (N <= 1) {
		return;
	}

	// divide and conquer
	CArray odd_block = block[std::slice(1, N/2, 2)]; // odd divide
	CArray even_block = block[std::slice(0, N/2, 2)]; // even divide
	computeFFT(odd_block); // recursive compute
	computeFFT(even_block); // recursive compute

	for (size_t k = 0; k < N/2; ++k) {
		Complex t = Complex(cos(-2 * PI * k / N), sin(-2 * PI * k / N)) * odd_block[k];
		block[k+N/2] = even_block[k] - t;
		block[k] = even_block[k] + t;
	}
}


