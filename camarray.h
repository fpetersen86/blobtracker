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
    CamArray(webcamtest* p, int testimages = 0);
    virtual ~CamArray();
	Camera *cams[64]; // 64 == max# of cameras in Linux
	int numCams;
	bool imgTest;
	bool calibrating;	
	webcamtest *w;
	void run();
	bool stopped;
	void stop();
	void loadFiles();
	
private:
	QSemaphore *sem;
	//host buffers
	char *h_a, *h_b, *h_c;
	camSettings *h_s;
	//device buffers
	char *d_a, *d_b, *d_c;
	camSettings *d_s;
	
	int canvX;
	int canvY;
	float lcStrength;
	float lcZoom;
	void initBuffers();
	//void initCUDA();
	void mainloop();
	//void mainloopCUDA();
	void findblob();
	void output();
	int threshold;
	
	int bufferImgSize;
	int bufferSettings;
};

#endif // CAMARRAY_H
