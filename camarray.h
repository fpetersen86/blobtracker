#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "camera.h"
#include <Qt/QtCore>
#include "webcamtest.h"

class webcamtest;
class Camera;

class CamArray : public QThread
{

public:
    CamArray(webcamtest* p);
    virtual ~CamArray();
	Camera *cams[64]; // 64 == max# of cameras in Linux
	int numCams;
	webcamtest *w;
	void run();
	bool stopped;
	void stop();
	
private:
	QSemaphore *sem;
	//host buffers
	char* h_a, *h_b;
	//device buffers
	char* d_a, *d_b;
};

#endif // CAMARRAY_H
