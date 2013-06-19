#include "camarray.h"
#include <stdio.h>
#include "webcamtest.h"
#include "global.h"

#ifdef CUDA
#include "utility_environment.h"
#include <cuda.h>


//GPU Kernel
__global__ void lensCorrection(char *image, char *output, int width, int height, int width2, int height2, float strength, float zoom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width2 + x;                              // index within linear array
    
    x += (width-width2)/2;
	y += (height-height2)/2;
	
	int halfWidth = width / 2;
	int halfHeight = height / 2;
	float correctionRadius = sqrtf(width * width + height * height) / strength;
	int newX = x - halfWidth;
	int newY = y - halfHeight;

	float distance = sqrtf(newX * newX + newY * newY);
	float r = distance / correctionRadius;
	
	float theta;
	if(r != 0)
	{
		theta = atanf(r)/r;
	} else {
		theta = 1;
	}
	
	int sourceX = halfWidth + theta * newX * zoom;
	int sourceY = halfHeight + theta * newY * zoom;
	
	output[elemID] = image[sourceY*width + sourceX];
}

__global__ void lensCorrection2(char *image, char *output, int width, int height, int width2, int height2, float strength, float zoom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;                              // index within linear array
    
	output[elemID] = image[elemID];
}

#endif


CamArray::CamArray(webcamtest* p) : QThread(p)
{
	w = p; // All hail the mighty alphabet ;)

}

#ifdef CUDA
void CamArray::initPlatform() {
	//host buffers
	cudaMallocHost(&h_a, bufferSize);
	cudaMallocHost(&h_b, bufferSize2);
	
	//device buffers
	cudaMalloc((void**) &d_a, bufferSize);
	cudaMalloc((void**) &d_b, bufferSize2);
}
#else //NOCUDA
void CamArray::initPlatform() {
	h_a = reinterpret_cast<char*>(malloc(bufferSize));
	h_b = reinterpret_cast<char*>(malloc(bufferSize2));
}
#endif


void CamArray::run()
{
	stopped = false;
	
	//initialise cams
	QDir d("/dev/");
	d.setFilter(QDir::System);
	d.setNameFilters(QStringList("*video*"));
	QStringList camList = d.entryList();
	numCams = camList.size();
	QString c;
	QSemaphore *sem = new QSemaphore(numCams);
	sem->acquire(numCams);
	
	bufferSize = xSize * ySize * numCams * sizeof(char);
	bufferSize2 = xSize2 * ySize2 * numCams * sizeof(char);
	
	initPlatform();
	
	for(int i = 0; i < numCams; i++)
	{
		c = camList.at(i);
		cams[i] = new Camera(d.absoluteFilePath(c).toStdString().c_str(), i, sem, h_a);
	}
	
	//start capturing
	for(int i = 0; i < numCams; i++)
	{
		cams[i]->start();
	}
	
	mainloop();
}

#ifdef CUDA
void CamArray::mainloopCUDA()
{
	dim3 cudaBlockSize(16,16);  // image is subdivided into rectangular tiles for parallelism - this variable controls tile size
	dim3 cudaGridSize(xSize2/cudaBlockSize.x, ySize2/cudaBlockSize.y);
	
	while(!stopped)
	{
		sem->acquire(numCams);
		cudaMemcpy( d_a, h_a, bufferSize, cudaMemcpyHostToDevice );
		handleCUDAerror(__LINE__);
		
		lensCorrection<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize, xSize2, ySize2, lcStrength, lcZoom);
		handleCUDAerror(__LINE__);
		
		cudaMemcpy( h_b, d_b, bufferSize2, cudaMemcpyDeviceToHost );
		handleCUDAerror(__LINE__);
		
		output();
	}
}

#endif

#ifdef NOCUDA
void CamArray::mainloopCPU()
{
	while(!stopped)
	{
		sem->acquire(numCams);
		
		int width = xSize;
		int height = ySize;
		int width2 = xSize2;
		int height2 = ySize2;
		float strength = lcStrength;
		float zoom = lcZoom;
		
		for (int y = 0; y < ySize2; y++)
		{
			for (int x = 0; x < xSize2; x++)
			{
				int myX = x;
				int myY = y;
				int elemID = myY*width2 + myX;                              // index within linear array

				myX += (width-width2)/2;
				myY += (height-height2)/2;
				
				int halfWidth = width / 2;
				int halfHeight = height / 2;
				float correctionRadius = sqrt(width * width + height * height) / strength;
				int newX = myX - halfWidth;
				int newY = myY - halfHeight;

				float distance = sqrt(newX * newX + newY * newY);
				float r = distance / correctionRadius;
				
				float theta;
				if(r != 0)
				{
					theta = atan(r)/r;
				} else {
					theta = 1;
				}
				
				int sourceX = halfWidth + theta * newX * zoom;
				int sourceY = halfHeight + theta * newY * zoom;
				
				h_b[elemID] = h_a[sourceY*width + sourceX];
				//qDebug("elemID: %d   X: %d myX: %d sourceX: %d   Y: %d myY: %d sourceY: %d", elemID, x, myX, sourceX, y, myY, sourceY);
			}
		}
		output();
	}
}
#endif

void CamArray::output()
{
	for (int y = 0; y < ySize; y++)
	{
		for (int x = 0; x < xSize; x++)
		{
			int val = h_a[y*xSize+x];
			w->i.setPixel(x,y, qRgb(val, val, val));
		}
	}
	
	for (int y = 0; y < ySize2; y++)
	{
		for (int x = 0; x < xSize2; x++)
		{
			int val = h_b[y*xSize2+x];
			w->i.setPixel(x,y+ySize, qRgb(val, val, val));
		}
	}
	w->update();
	//qDebug("available: %d", sem->available());
}

void CamArray::stop()
{
	stopped = true;
}


void CamArray::loadFile(QString filenName)
{
	QImage fileImage(filenName);

	for (int y = 0; y < fileImage.height(); y++)
	{
		for (int x = 0; x < fileImage.width(); x++)
		{
			h_a[y*fileImage.width()+x] = qGray(fileImage.pixel(x,y));
		}
	}
}

CamArray::~CamArray()
{
	for (int i = 0; i < numCams; i++)
	{
		cams[i]->stop();
	}
	for (int i = 0; i < numCams; i++)
	{
		cams[i]->wait();
		delete cams[i];
	}
	qDebug() << "CamArray stopped";
	
	// free memory buffers
#ifndef NOCUDA
	cudaFree(d_a);
	handleCUDAerror(__LINE__);
	cudaFree(d_b);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_a);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_b);
	handleCUDAerror(__LINE__);
#else
	free(h_a);
	free(h_b);
#endif
	qDebug("Memory deallocated successfully\n");
}

