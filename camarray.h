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
	int id;
	int x;
	int y;
	int x2;
	int y2;
	int maxDepth;
	QColor color;
};


struct yRange
{
	int y1;
	int y2;
};

struct xyRange
{
	int x1;
	int x2;
	int y1;
	int y2;
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
	int min(int a, int b) {
		if(a < b)
			return a;
		return b;
	};
	int max(int a, int b) {
		if(a > b)
			return a;
		return b;
	};
	
private:
	QSemaphore *sem;
	//host buffers
	unsigned char *h_a, *h_b, *h_c, *h_d;
	camSettings *h_s;
	bool *h_blobMap;
	//device buffers
	unsigned char *d_a, *d_b, *d_c, *d_d;
	camSettings *d_s;
	bool *d_blobMap;
	yRange *d_yRanges;
	xyRange *d_xyRanges;
	xyRange *h_xyRanges;
	
	int canvX;
	int canvY;
	int canvOffX;
	int canvOffX2;
	int canvOffY;
	int canvOffY2;
	float lcStrength;
	float lcZoom;
	void initBuffers();
	//void initCUDA();
	void mainloop();
	//void mainloopCUDA();
	void output();
    bool white(int x, int y);
	int threshold;
	
	int bufferImgSize;
	int bufferSettings;
	int bufferStitchedImg;
	int xyRangeSize;
	
	
	//FieldState blobMap[xSize/blobstep][ySize/blobstep];
	QList<Blob*> blobs;
	QList<Blob*> blobs2;
protected:
	void findblob();
    void isBlob(int x, int y, Blob* bob, int depth);
	void trackBlobs();
	void findBlobs_3();
	bool mergeBlobs(Blob *bob, Blob* notBob);
};

#endif // CAMARRAY_H
