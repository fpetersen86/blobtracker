#include "camarray.h"
#include <stdio.h>

CamArray::CamArray(webcamtest* p)
{
	
	//initialise cams
	QDir d("/dev/");
	d.setFilter(QDir::System);
	d.setNameFilters(QStringList("*video*"));
	QStringList camList = d.entryList();
	numCams = camList.size();
	QString c;
	QSemaphore *sem = new QSemaphore(numCams);
	sem->acquire(numCams);
	
	
	int bufferSize = 320 * 240 * numCams * sizeof(char);
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

