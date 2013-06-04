#include "camarray.h"

CamArray::CamArray(webcamtest* p)
{
	QDir d("/dev/");
	d.setFilter(QDir::System);
	d.setNameFilters(QStringList("*video*"));
	QStringList camList = d.entryList();

	QString c;
	for(int i = 0; i < camList.size(); i++)
	{
		c = camList.at(i);
		cams[i] = new Camera(d.absoluteFilePath(c).toStdString().c_str());
	}
	numCams = camList.size();
	cams[0]->w = p;
	//cams[0]->capture();
	//cams[0]->loop();
	
}

CamArray::~CamArray()
{
	for (int i = 0; i < numCams; i++)
		delete cams[i];

}

