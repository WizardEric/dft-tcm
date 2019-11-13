# 2D DFT, IDFT
- Threads (forward = DFT and reverse = DFT then IDFT): 
```
./p31 [forward/reverse] [inputfile] [outputfile] 
```
- CUDA: 
```
./p33 forward [inputfile] [outputfile] 
```
- MPI: 
```
mpirun -np 8 ./p32 forward [inputfile] [outputfile]
```

Threads: It initializes 7 threads and using the the 8th one as the main thread, distributes the flattened DFT matrix among the 8 of them. Inverse DFT (IDFT) was implemented and the original input matrix of real values can be regained by putting reverse as the second input parameter like so: ./p31 reverse [inputfile] [outputfile]. The regained matrix is complex valued and is not exactly equal to the original due to precision error during calculation. Rounding to 3 significant figures do make them equal.

Open MPI: It utilizes all 8 nodes (the 8 in the input indicates this). The root node reads the input file into a dynamic memory and operates the scatter and gather operations. The root node does FFT on 1/8 of the rows and scatters the remaining 7/8 to the other nodes. The child nodes then does FFT on its row and sends it back to the root node. After appending all of the values into a matrix, we swap the matrix to make the columns as rows and repeat the above procedure to complete the 2D FFT. After the scatter and gather, we transpose the matrix again to get the final output.

CUDA: It computes Fourier transform using DFT. On the first pass we perform row-wise DFT, followed by column-wise DFT on the previous output. Each block has 128 threads and the number of blocks are allocated based on input matrix size. Each thread performs Fourier Transform for a single point in the 2D array.