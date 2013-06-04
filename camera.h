#ifndef CAMERA_H
#define CAMERA_H

#include <stdlib.h>
#include <Qt/QtGui>
#include "webcamtest.h"

class webcamtest;

class Camera
{
	struct buffer {
        void   *start;
        size_t  length;
	};

public:
    Camera(const char* device);
    virtual ~Camera();
		
	void capture();
	void stop();
	void loop();
    void doOurStuff(void* bufStart, unsigned int size, int index);
	struct buffer *buffers;
	webcamtest *w;

private:
	void setParameters();
	int fd;
	bool stopped;
};

#endif // CAMERA_H
