#ifndef CAMERA_H
#define CAMERA_H

#include <stdlib.h>
#include <Qt/QtGui>
#include <Qt/QtCore>

#include "webcamtest.h"

class webcamtest;

class Camera : public QThread
{
	struct buffer
	{
        void   *start;
        size_t  length;
	};

public:
    Camera(const char *device, const int id, QSemaphore *sem);
    virtual ~Camera();
	
	void run();
	void stop();
	void loop();
    void doOurStuff(void* bufStart, unsigned int size, int index);
	struct buffer *buffers;
	webcamtest *w;

private:
	QSemaphore *sem;
	int id;
	void setParameters();
	int fd;
	bool stopped;
};

#endif // CAMERA_H
