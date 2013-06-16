#ifndef UTILITY_ENVIRONMENT_H
#define UTILITY_ENVIRONMENT_H
/*********************************************************************************\
 *  A better-than-nothing environment and utility functions for GPGPU exercises  *
 *  (function names start with "ue_" to show that these are not part of CUDA)    *
 *                                                                               *
 *  no changes to this code (or the rest of the infrastructure) should be        *
 *  necessary to solve the exercise                                              *
 *                                                                               *
 *  based on example code of the 5.0 SDK Copyright 1993-2012 NVIDIA Corporation  *
\*********************************************************************************/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <algorithm>

using namespace std;

#ifdef _WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#define srand48(seed) srand(seed)
#define drand48() double(rand())/double(RAND_MAX)
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif


/***************************************************\
 *  General CUDA error handler                     *
 *  Prints error message, if something went wrong  *
 *  use this after every CUDA call to locate bugs  *
\***************************************************/
void handleCUDAerror( int line )
{
	cudaError_t err = cudaGetLastError();
	if ( err == cudaSuccess )
		return;

	printf("CUDA error message: %s\n"
		   "line %d\n", cudaGetErrorString(err), line );

	int devID;
    cudaGetDevice( &devID );
    cudaDeviceProp props;
    cudaGetDeviceProperties( &props, devID );
    printf("CUDA device %d: \"%s\" with compute capability %d.%d\n",
           devID, props.name, props.major, props.minor);
}


/*********************************************************\
 *  probe default CUDA device                            *
 *  prints device name and basic capabilities to stdout  *
\*********************************************************/
void ue_checkGPU(void)
{
	int numberOfDevices = 0;
	if(cudaSuccess != cudaGetDeviceCount(&numberOfDevices))
	{
		printf("No CUDA device found\n");
		exit(EXIT_FAILURE);
	}

	int deviceNumber = 0;
	if(cudaSuccess != cudaGetDevice(&deviceNumber))
	{
		printf("No CUDA device found\n");
		exit(EXIT_FAILURE);
	}

	cudaDeviceProp deviceProperties;
	if (cudaSuccess != cudaGetDeviceProperties(&deviceProperties, deviceNumber))
	{
		printf("Could not get device properties\n");
		exit(EXIT_FAILURE);
	}

	printf("using CUDA device number %d (of %d device(s) available): ", deviceNumber, numberOfDevices);
	printf("%s\n", &(deviceProperties.name));
	printf("compute capability: %0d.%0d\n", deviceProperties.major, deviceProperties.minor);
	printf("clock frequency: %d MHz, device memory: %ld MBytes\n", deviceProperties.clockRate/1000,
		deviceProperties.totalGlobalMem/(1024*1024));
	printf("device limits: %d threads\n  (%d, %d, %d) blockSize\n  (%d, %d, %d) gridSize\n", deviceProperties.maxThreadsPerBlock,
		deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2],
		deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
	printf("\n");
}


/******************************************************************\
 *  simple time measurement                                       *
 *  cannot be nested, because there is only one global timer      *
 *  has some overhead, so measure milliseconds, not microseconds  *
\******************************************************************/
cudaEvent_t start_UE_obscureName, stop_UE_obscureName;  // names obscured to avoid clash with main code

// starts time measurement
void ue_startTimer(void)
{
	cudaEventCreate(&start_UE_obscureName);
	cudaEventRecord(start_UE_obscureName, 0);
	cudaEventSynchronize(start_UE_obscureName);
}

// returns time in milliseconds since last call to ue_startTimer()
float ue_stopTimer(void)
{
	cudaEventCreate(&stop_UE_obscureName);
	cudaEventRecord(stop_UE_obscureName, 0);
	cudaEventSynchronize(stop_UE_obscureName);

	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start_UE_obscureName, stop_UE_obscureName);

	cudaEventDestroy(start_UE_obscureName);      // we are done measuring time
	cudaEventDestroy(stop_UE_obscureName);

	return elapsedTime;
}


/**************************************\
 *  write image as portable grey map  *
\**************************************/
 int ue_dumpImageToFile(const char *fileName, const int *data, const int imgWidth, const int imgHeight)
{
	FILE* outFile = fopen(fileName, "wb");
	if (outFile == NULL)
	{
		printf("could not open output image file\n");
		return -1;
	}

	fprintf(outFile, "P5\n%d %d\n255\n", imgWidth, imgHeight);  // simple header for binary "portable grey map" file

	// write all "pixels"
	for (int i = 0; i < imgWidth*imgHeight; i++)
    {
	    putc(data[i], outFile);
	}

	fclose(outFile);
	return 0;
}


/***********************************************\
 *  element-wise comparison of two buffers     *
 *  bufferSize is in total number of elements  *
\***********************************************/
unsigned long ue_countDifferingArrayElements(const int* bufferA, const int* bufferB, unsigned long bufferSize)
{
	unsigned long numDiff = 0;
    for (unsigned long i = 0; i < bufferSize; i++)
    {
        numDiff += (bufferA[i] != bufferB[i]);
    }

	return numDiff;
}

#endif  // defined(UTILITY_ENVIRONMENT_H)
