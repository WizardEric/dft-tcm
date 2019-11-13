#include <string.h> 
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <thread>
#include <complex.h>
#include <complex.cc>
#include <input_image.cc>
#include <input_image.h>
#include <chrono>

// const float PI = 3.14159265358979f;

// String split into floats
std::vector<float> str_flt(std::string s)
{
    std::string delimiter = ",";
    std::vector<float> r;
    s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());
    int pos;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        r.push_back((float) atof(token.c_str()));
        // std::cout << token << std::endl;
        s.erase(0, pos + delimiter.length());
    }    
    // std::cout << "temp" << atof (s.c_str()) << std::endl;
    r.push_back((float) atof(s.c_str()));   
    return r;
}
// String split into integers
std::vector<int> str_int(std::string s)
{
    std::string delimiter = ",";
    std::vector<int> r;
    s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());
    int pos;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        r.push_back(atoi(token.c_str()));
        // std::cout << token << std::endl;
        s.erase(0, pos + delimiter.length());
    }
    r.push_back(atoi(s.c_str()));       
    return r;
}

void dft1d(Complex *in, Complex *out, int *row, int stt_grp, int *each_thread) {
    float ang;
    // Complex sums(0,0);
    // int top, bottom, left, right, back, front;
    for (int stt = stt_grp * (*row); stt < (stt_grp * (*row)) + (*each_thread) * (*row) ; stt+=(*row)){
        for (int idxi = stt; idxi < stt + (*row); idxi++){
            Complex sums(0,0);
            for (int idx = stt; idx < stt + (*row); idx++){
                if (idx  < (*row) * (*row)){
                    ang = 2 * PI * (idx-stt) * (idxi-stt) / (*row);
                    sums = sums + Complex(cos(ang),-1 * sin(ang)) * in[idx];
                }
            }
            out[idxi] = sums;
        }
    }
}

void idft1d(Complex *in, Complex *out, int *row, int stt_grp, int *each_thread) {
    float ang;
    // Complex sums(0,0);
    // int top, bottom, left, right, back, front;
    for (int stt = stt_grp * (*row); stt < (stt_grp * (*row)) + (*each_thread) * (*row) ; stt+=(*row)){
        for (int idxi = stt; idxi < stt + (*row); idxi++){
            Complex sums(0,0);
            for (int idx = stt; idx < stt + (*row); idx++){
                if (idx  < (*row) * (*row)){
                    ang = 2 * PI * (idx-stt) * (idxi-stt) / (*row);
                    sums = sums + Complex(cos(ang)/ (*row), sin(ang)/ (*row)) * in[idx];
                }
            }
            out[idxi] = sums ;
        }
    }
}

int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();
    InputImage img = InputImage(argv[2]);
    std::string str;
    int line_no = 0;
    int no_dim;
    float k;
    std::vector<Complex> grid;
    std::vector<float> tmp;
    int no_timesteps;
    float temp_start;

    // Setup variables
    Complex *in, *out; // host copies of a, b, c
    // TODO: row, column first and second entries of first line?
    int row = img.get_height();
    int column = img.get_width();
    int N = row*column;
    int size = (N) * sizeof(Complex);
    // Alloc space for host copies and setup values
    in = (Complex *)malloc(size);
    out = (Complex *)malloc(size);
    std::ofstream myfile;
    
    // READ FILE ################################
    // TODO: line 1 which is row and which is column
    in = img.get_image_data();

    // std::vector<std::thread> threads;
    int nthreads = 7; // 8-1
    int mt_id = nthreads;
    int each_thread = (int) ceil( ((float)row) / (float)(nthreads+1) );
    int mt_id_thread = row - (nthreads * each_thread);
    
    std::thread threads[11];
    // Launch threads for first dimension
    for (int j = 0; j < nthreads; ++j){
        threads[j] = std::thread(dft1d, in, out, &row, j*each_thread, &each_thread);
    }
    dft1d(in, out, &row, mt_id*each_thread, &mt_id_thread);
    for (int j = 0; j < nthreads; ++j){
        threads[j].join();
    }

    // TRANSPOSING
    Complex temp(0,0);
    for (int i=0; i<row; i++){
        for (int j=i+1; j<column; j++){
            if (i!=j){
                temp = out[j+i*row];
                out[j+i*row] = out[i+j*row];
                out[i+j*row] = temp;
            }
        }
    }
    memcpy(in, out, size);
    // Launch threads for second dimension
    for (int j = 0; j < nthreads; ++j){
        threads[j] = std::thread(dft1d, in, out, &row, j*each_thread, &each_thread);
    }
    dft1d(in, out, &row, mt_id*each_thread, &mt_id_thread);
    for (int j = 0; j < nthreads; ++j){
        threads[j].join();
    }
    // TRANSPOSING
    for (int i=0; i<row; i++){
        for (int j=i+1; j<column; j++){
            if (i!=j){
                temp = out[j+i*row];
                out[j+i*row] = out[i+j*row];
                out[i+j*row] = temp;
            }
        }
    }

    // IDFT ***************************************************
    if (argv[1][0] == 'r'){
        memcpy(in, out, size);
        // Launch threads for first dimension
        for (int j = 0; j < nthreads; ++j){
            threads[j] = std::thread(idft1d, in, out, &row, j*each_thread, &each_thread);
        }
        idft1d(in, out, &row, mt_id*each_thread, &mt_id_thread);
        for (int j = 0; j < nthreads; ++j){
            threads[j].join();
        }

        // TRANSPOSING
        Complex temp(0,0);
        for (int i=0; i<row; i++){
            for (int j=i+1; j<column; j++){
                if (i!=j){
                    temp = out[j+i*row];
                    out[j+i*row] = out[i+j*row];
                    out[i+j*row] = temp;
                }
            }
        }
        memcpy(in, out, size);
        // Launch threads for second dimension
        for (int j = 0; j < nthreads; ++j){
            threads[j] = std::thread(idft1d, in, out, &row, j*each_thread, &each_thread);
        }
        idft1d(in, out, &row, mt_id*each_thread, &mt_id_thread);
        for (int j = 0; j < nthreads; ++j){
            threads[j].join();
        }
        // TRANSPOSING
        for (int i=0; i<row; i++){
            for (int j=i+1; j<column; j++){
                if (i!=j){
                    temp = out[j+i*row];
                    out[j+i*row] = out[i+j*row];
                    out[i+j*row] = temp;
                }
            }
        }
    }

    // Write to files
    img.save_image_data(argv[3], out, column, row);
    // // std::ofstream myfile;
    // myfile.open (argv[2]);
    // // myfile << mt_id_thread << "  " << each_thread;
    // for (int y = 0; y < row; y++){
    //     for (int x = 0; x < row; x++){
    //         myfile << out[x + row*y];
    //         if (x < row-1){
    //             myfile << " ";
    //         }       
    //     }
    //     if (y < row-1){
    //         myfile << "\n";
    //     }
    // } 
    // myfile.close();
        
    // Cleanup
    free(in); free(out);

    

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Runtime in seconds: " << elapsed.count() << "s";
    // myfile.open ("timing.txt");
    // myfile << "Runtime in seconds: " << elapsed.count() << "s";
    // myfile.close();

    return 0;
}