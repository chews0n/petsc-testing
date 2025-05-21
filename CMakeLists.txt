cmake_minimum_required(VERSION 3.15)
project(petsc-testing CXX)

set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE(PkgConfig REQUIRED)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

if (NOT DEFINED PETSC_LIB_PATH)
    set(PETSC_LIB_PATH "/opt/anl/petsc")
endif()
link_directories("${PETSC_LIB_PATH}/lib")
SET(MKLROOT "$ENV{MKLROOT}/lib")
link_directories("${MKLROOT}")
set(MKL_LINK_LIBS "-l:libmkl_intel_lp64.a -l:libmkl_sequential.a -l:libmkl_core.a -l:libmkl_blacs_intelmpi_lp64.a")


include_directories(SYSTEM "${PETSC_LIB_PATH}/include")


add_executable(petsc-testing main.cpp)


target_link_libraries(petsc-testing -l:libpetsc.a -l:libmpi.a -l:libmpifort.a -l:libHYPRE.a)

target_link_libraries(petsc-testing ${MKL_LINK_LIBS} ${MKL_LINK_LIBS} ${MKL_LINK_LIBS} gfortran)