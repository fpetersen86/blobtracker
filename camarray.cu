#include "camarray.h"
#include <stdio.h>



//GPU Kernel
__global__ void findBlobs(char *image, float *output, int width, int height, float strength, float zoom)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;         // coordinates within 2d array follow from block index and thread index within block
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int elemID = y*width + x;                              // index within linear array

	// compute cells needed for update (neighbors + central element)
	int borderFlag = (x > 0);                              // boolean values enable border handling without thread divergence
	float leftNeighb = input[elemID - borderFlag];
	borderFlag = (x < (width - 1));
	float rightNeighb = input[elemID + borderFlag];
	borderFlag = -(y > 0);									// unary minus turns boolean value into boolean bitwise mask
	float topNeighb = input[elemID - (borderFlag & width)];	
	borderFlag = -(y < (height - 1));
	float bottomNeighb = input[elemID + (borderFlag & width)];
	float currElement = input[elemID];
	
	int halfWidth = width / 2;
	int halfHeight = height / 2;
	float correctionRadius = sqrtf(width * width + height * height) / strength;
	int newX = x - halfWidth;
	int newY = y - halfHeight;

	float distance = sqrtf(newX * newX + newY * newY);
	float r = distance / correctionRadius;
	
	float theta = 1;
	if(r != 0)
	{
		theta = atan(r)/r;
	}
	
	float sourceX = halfWidth + theta * newX * zoom;
	float sourceY = halfHeight + theta * newY * zoom;
	
	output[elemID] = image[sourceY*width + sourceX];
	
	//output[elemID] = currElement + mu * ( (leftNeighb-currElement) + (rightNeighb-currElement) + (bottomNeighb-currElement) + (topNeighb-currElement) );
}



CamArray::CamArray(webcamtest* p)
{
	int xSize = 320;
	int ySize = 240;
	
	//initialise cams
	QDir d("/dev/");
	d.setFilter(QDir::System);
	d.setNameFilters(QStringList("*video*"));
	QStringList camList = d.entryList();
	numCams = camList.size();
	QString c;
	QSemaphore *sem = new QSemaphore(numCams);
	sem->acquire(numCams);
	
	
	int bufferSize = xSize * ySize * numCams * sizeof(char);
	//host buffers
	char* h_a = (char *) malloc(bufferSize);
	
	//device buffers
	char* d_a;
	cudaMalloc((void**) &d_a, bufferSize);
	
	
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
	
	
	while(true)
	{
		sem->acquire(numCams);
		cudaMemcpy( d_a, h_a, bufferSize, cudaMemcpyHostToDevice );
		//qDebug("available: %d", sem->available());
	}
	
}


CamArray::~CamArray()
{
	for (int i = 0; i < numCams; i++)
		delete cams[i];

}
