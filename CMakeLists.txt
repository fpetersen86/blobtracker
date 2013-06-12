project(blobtracker)
cmake_minimum_required(VERSION 2.8)
find_package(Qt4 REQUIRED)
find_package(CUDA)
set (AUTOMOC true)

include_directories(${QT_INCLUDES} ${CMAKE_CURRENT_BINARY_DIR})

set(blobtracker_UIS settings.ui)
qt4_wrap_ui(blobtracker_UIS_H ${blobtracker_UIS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_10,code=sm_10)
set(blobtracker_SRCS webcamtest.cpp camera.cpp main.cpp)
set(blobtracker_cuda_SRCS camarray.cu)
qt4_automoc(${blobtracker_SRCS} ${blobtracker_cuda_SRCS})
cuda_add_executable(blobtracker ${blobtracker_SRCS} ${blobtracker_cuda_SRCS} ${blobtracker_UIS_H})
#cuda_add_executable(blobtracker ${blobtracker_SRCS} ${blobtracker_cuda_O})
target_link_libraries(blobtracker ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
install(TARGETS blobtracker RUNTIME DESTINATION bin)



#cuda_compile(BLOBTRACKER_CUDA_O ${blobtracker_cuda_SRCS})
#cuda_compile(DRIVER_O driver.cu OPTIONS -G)
#cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})
