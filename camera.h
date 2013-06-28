#ifndef CAMERA_H
#define CAMERA_H

#include <stdlib.h>
#include <Qt/QtGui>
#include <Qt/QtCore>

//#include "webcamtest.h"

//class webcamtest;

class Camera : public QThread
{
	struct buffer
	{
        void   *start;
        size_t  length;
	};

public:
    Camera(const char *device, const int id, QSemaphore *sem, char* myBuffer);
    virtual ~Camera();
	
	void run();
	void stop();
	void loop();
    void doOurStuff(void* bufStart, unsigned int size, int index);
	struct buffer *buffers;
	float angle;

private:
	QSemaphore *sem;
	char* myBuffer;
	int id;
	void setParameters();
	int fd;
	bool stopped;
	int video_set_format(int dev, unsigned int w, unsigned int h);
	int video_set_framerate(int dev);
};

#endif // CAMERA_H
