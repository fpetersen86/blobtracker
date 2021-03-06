#ifndef CAMERA_H
#define CAMERA_H

#include <stdlib.h>
#include <Qt/QtGui>
#include <Qt/QtCore>

//#include "webcamtest.h"

//class webcamtest;

/*-------------------------------------------------------------------------------------------/

	Struct for camera-specific settings. It is an isolated struct because at one point 
	we forward it to the GPU.

/-------------------------------------------------------------------------------------------*/

struct camSettings
{
	float angle;
	int xOffset;
	int yOffset;
};

/*-------------------------------------------------------------------------------------------/

	This class handles setup and input of one camera.

/-------------------------------------------------------------------------------------------*/

class Camera : public QThread
{
	Q_OBJECT
	struct buffer
	{
        void   *start;
        size_t  length;
	};

public:
    Camera(const char *device, const int id, QSemaphore *sem, unsigned char* myBuffer, camSettings *cset);
    virtual ~Camera();
	
	void run();
	void stop();
	void loop();
    void doOurStuff(void* bufStart, unsigned int size, int index);
	struct buffer *buffers;
	int getID() {return id;};

private:
	QSemaphore *sem;
	unsigned char* myBuffer;
	int id;
	void setParameters();
	int fd;
	bool stopped;
	int video_set_format(int dev, unsigned int w, unsigned int h);
	int video_set_framerate(int dev);
	camSettings *settings;
	
public slots:
	void setXOffset(int xOff) {settings->xOffset = xOff;};
	void setYOffset(int yOff) {settings->yOffset = yOff;};
	void setAngle(double angle) {settings->angle = angle;};

	
	
};

#endif // CAMERA_H
