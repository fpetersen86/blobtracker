project(blobtracker)
cmake_minimum_required(VERSION 2.8)
find_package(Qt4 REQUIRED)
find_package(CUDA)

include_directories(${QT_INCLUDES} ${CMAKE_CURRENT_BINARY_DIR})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
set(blobtracker_SRCS webcamtest.cpp camarray.cpp camera.cpp main.cpp)
qt4_automoc(${blobtracker_SRCS})
add_executable(blobtracker ${blobtracker_SRCS})
target_link_libraries(blobtracker ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
install(TARGETS blobtracker RUNTIME DESTINATION bin)



#cuda_compile(HELPER_O helper.cu)
#cuda_compile(DRIVER_O driver.cu OPTIONS -G)

#cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})
