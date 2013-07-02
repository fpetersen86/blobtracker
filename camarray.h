#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "camera.h"
#include <Qt/QtCore>
//#include "webcamtest.h"

class webcamtest;
class Camera;

struct Blob
{
	int x;
	int y;
	int w;
	int h;
};

class CamArray : public QThread
{
friend class webcamtest;

public:
    CamArray(webcamtest* p, int testimages = 0);
    virtual ~CamArray();
	Camera *cams[64]; // 64 == max# of cameras in Linux
	int numCams;
	bool imgTest;
	int viewmode;
	webcamtest *w;
	void run();
	bool stopped;
	void stop();
	void loadFiles();
	
private:
	QSemaphore *sem;
	//host buffers
	unsigned char *h_a, *h_b, *h_c, *h_d;
	camSettings *h_s;
	//device buffers
	unsigned char *d_a, *d_b, *d_c, *d_d;
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
    bool white(int x, int y);
	int threshold;
	
	int bufferImgSize;
	int bufferSettings;
	int bufferStitchedImg;
};

#endif // CAMARRAY_H
