#include "camarray.h"

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
	for(int i = 0; i < numCams; i++)
	{
		c = camList.at(i);
		cams[i] = new Camera(d.absoluteFilePath(c).toStdString().c_str(), i, sem);
	}
	
	//start capturing
	for(int i = 0; i < numCams; i++)
	{
		cams[i]->start();
	}
	
	
	while(true)
	{
		sem->acquire(numCams);
		//qDebug("available: %d", sem->available());
	}
	
}


CamArray::~CamArray()
{
	for (int i = 0; i < numCams; i++)
		delete cams[i];

}

