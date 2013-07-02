#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "global.h"
#include "camera.h"
#include <Qt/QtCore>
//#include "webcamtest.h"

class webcamtest;
class Camera;

struct Blob
{
	int x;
	int y;
	int x2;
	int y2;
	int maxDepth;
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
	enum Direction {left, down, right};
	enum FieldState {no = 0, yes = 1, visited = 2};
	
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
    bool white(int x, int y);
	int threshold;
	
	int bufferImgSize;
	int bufferSettings;
	FieldState blobMap[xSize/blobstep][ySize/blobstep];
	QList<Blob*> blobs;
protected:
    int isBlob(int x, int y, Blob* bob, int depth);
};

#endif // CAMARRAY_H
