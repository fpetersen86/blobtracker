#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "global.h"
#include "camera.h"
#include <Qt/QtCore>
//#include "webcamtest.h"

class webcamtest;
class Camera;

struct Blob			// A blob, defined as a rectangle.
{
	int id;			// Id of the blob. Not really used.
	int x;			// left edge
	int y;			// upper edge
	int x2;			// right edge
	int y2;			// bottom edge
	int maxDepth;	// unimportant
	QColor color;	// color of the rectangle around the blob
};


struct yRange		// used to save per-line ranges (part of blob) of white pixels
{
	int y1;			// start
	int y2;			// end
};

struct xyRange		// used to save 2-dimensional ranges (part of blob) of white pixels
{
	int x1;			// left edge
	int x2;			// right edge
	int y1;			// upper edge
	int y2;			// bottom edge
};

/*-------------------------------------------------------------------------------------------/

	This class handles all Camera objects and all the GPU-code.

/-------------------------------------------------------------------------------------------*/


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
