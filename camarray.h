#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "camera.h"
#include <Qt/QtCore>
//#include "webcamtest.h"

class webcamtest;
class Camera;

class CamArray : public QThread
{
friend class webcamtest;

public:
    CamArray(webcamtest* p);
    virtual ~CamArray();
	Camera *cams[64]; // 64 == max# of cameras in Linux
	int numCams;
	webcamtest *w;
	void run();
	bool stopped;
	void stop();
	void loadFile(QString filenName);
	
private:
	QSemaphore *sem;
	//host buffers
	char* h_a, *h_b;
	//device buffers
	char* d_a, *d_b;
	
	float lcStrength;
	float lcZoom;
	void initBuffers();
	//void initCUDA();
	void mainloop();
	//void mainloopCUDA();
	void output();
	
	int bufferSize;
	int bufferSize2;
	

	
};

#endif // CAMARRAY_H
