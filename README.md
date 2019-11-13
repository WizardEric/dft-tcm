# 2D DFT, IDFT
- threads (forward = DFT and reverse = DFT then IDFT): 
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