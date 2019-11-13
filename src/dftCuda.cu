#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "complex.cu"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <input_image.cu>
// #include <input_image.h>
#define BLOCK_SIZE 128

using namespace std;
using namespace chrono;

__global__ void DFTrow(Complex* d_in, int* d_size, Complex* d_in2) {
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int col = gindex%d_size[0];
  int row = gindex/d_size[0];
  float exp=0;
  Complex temp(0, 0);
  for(int i=0; i< *d_size; i++){
    exp = 2*PI*col*i/d_size[0];
    temp.real += d_in[i+row*d_size[0]].real*cos(exp);
    temp.imag -= d_in[i+row*d_size[0]].real*sin(exp);;
  }
//  printf("%.2f\n",cos(exp));
  d_in2[gindex].real = temp.real;
  d_in2[gindex].imag = temp.imag;
}
__global__ void DFTcol(Complex* d_in2, int* d_size, Complex* d_out) {
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int col = gindex%d_size[0];
  int row = gindex/d_size[0];
  float exp=0;
  Complex temp(0, 0);
  for(int i=0; i<d_size[0]; i++){
    int idx = col+i*d_size[0];
    exp = 2*PI*row*i/d_size[0] ;
    temp.real += ( d_in2[idx].real*cos(exp) + d_in2[idx].imag*sin(exp) );
    temp.imag += ( d_in2[idx].imag*cos(exp) - d_in2[idx].real*sin(exp) );
  }
//  printf("%.2f\n",cos(exp));
  d_out[gindex].real = temp.real;
  d_out[gindex].imag = temp.imag;
}

/*
__global__ void defComplex(int* d_a, int* d_b, Complex* d_c) {
        d_c[0].real = d_a[1];
        d_c[1]->imag = *d_b;
        printf("hello_old\n");
}
*/

int main(int argc, char**argv) {
high_resolution_clock::time_point t1 = high_resolution_clock::now();
  InputImage img = InputImage(argv[2]);
  int row = img.get_height();
  int column = img.get_width();
  Complex *in = img.get_image_data();
  // int *ImgData;
  // ifstream configRead;
  // configRead.open(argv[1]); /// will need to change depending on the run type ////
  // int size_x=0, size_y=0, temp=0; // size_x is number of colums, size_y is rows
  // string STRING;
  // int line_num=-1; //first useful data is going to be in line 0
  // if (configRead.is_open()) {
  //   std::cout << "file open" << std::endl;
  //   while(!configRead.eof()) {
  //     getline(configRead, STRING);
  //     stringstream sst(STRING);
  //     int col=0;
  //     while (sst >> temp) {
  //       if(line_num==-1) {
  //         if(col==0){
  //           size_x=temp;
  //         }
  //         else {
  //           size_y=temp;
  //           ImgData = (int*) malloc ((size_x*size_y+1)*sizeof(int));
  //           if(size_x!=size_y) cout << "INFO: width and height read are not equal please check" << endl;
  //         }
  //       }

  //       else {
  //         if(col<size_x) ImgData[col+size_x*line_num]=temp;
  //         else cout << "more data in the row: "<< line_num << endl;
  //       }
  //       col++;
  //     }
  //     line_num++;

  //   }
  // }
  // configRead.close();
  Complex *d_in;
  int *d_size;
  Complex *d_in2, *d_out;
  int size_x = row;
  int N = size_x*size_x;
  Complex *rowFT, *colFT;
  rowFT = (Complex*) malloc ((N+1)*sizeof(Complex));
  colFT = (Complex*) malloc ((N+1)*sizeof(Complex));

  cudaMalloc((void**)&d_in, N*sizeof(Complex));
  cudaMalloc((void**)&d_size, sizeof(int));
  cudaMalloc((void**)&d_in2, N*sizeof(Complex));
  cudaMalloc((void**)&d_out, N*sizeof(Complex));

  cudaMemcpy(d_in, in, N*sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(d_size, &size_x, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, rowFT, N*sizeof(Complex), cudaMemcpyHostToDevice);

  DFTrow<<<((N+BLOCK_SIZE-1)/BLOCK_SIZE),BLOCK_SIZE>>>(d_in, d_size, d_in2);

  cudaMemcpy(rowFT, d_in2, N*sizeof(Complex), cudaMemcpyDeviceToHost);

  DFTcol<<<((N+BLOCK_SIZE-1)/BLOCK_SIZE),BLOCK_SIZE>>>(d_in2, d_size, d_out);

  cudaMemcpy(colFT, d_out, N*sizeof(Complex), cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_in2);
  cudaFree(d_size);
  cudaFree(d_out);
  img.save_image_data(argv[3], colFT, column, row);
  // std::cout << colFT[2];
    // fstream myfile;
    // img.save_image_data(argv[3], colFT, column, row);
    // myfile.open(argv[2], fstream::out);
    // myfile << "";
    // myfile.close();
    // myfile.open( argv[2], fstream::app);
    // for(int y=0; y<size_x; y++){
    //     for(int x=0; x<size_x; x++) {
    //         if((x+1)%size_x != 0) myfile << colFT[x+y*size_x] << ", ";
    //         else myfile << colFT[x+y*size_x] ;
    //     }
    //     myfile << "\n" ;
    // }

    // myfile.close();
  free(rowFT);
  free(colFT);
  high_resolution_clock::time_point  t2 = high_resolution_clock::now();
  auto  duration = duration_cast<microseconds>( t2 - t1 ).count();
  // myfile.open ("timing.txt");
  cout << "Runtime in seconds: " << float(duration)/(1000*1000) << "s";
  return 0;
}

