cmake_minimum_required(VERSION 3.9.1)
PROJECT(final)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

find_package(MPI REQUIRED)
if (MPI_FOUND)
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
	 message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

find_package(CUDA QUIET REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX)

INCLUDE_DIRECTORIES( 
    src
)

set(SOURCE_FILES
        src/complex.h
        src/complex.cc
        src/input_image.h
        src/input_image.cc
        src/input_image.cu
	src/complex.cu
	src/dftCuda.cu
	src/p32.cpp
	src/dft_threads.cc
        )

ADD_EXECUTABLE(
	p31
	src/dft_threads.cc
	)

ADD_EXECUTABLE(
	p32
	src/p32.cpp
	)

CUDA_ADD_EXECUTABLE(
	p33
	src/dftCuda.cu
)

target_link_libraries(p32 ${MPI_LIBRARIES})


