#include "camera.h"
#include <stdlib.h>
#include <sys/ioctl.h> 
#include <linux/videodev2.h> 
#include <sys/mman.h> // mmap
#include <string.h> // memset
#include <fcntl.h> // open
#include <unistd.h> // close
#include <errno.h>

#define CLEAR(x) memset(&(x), 0, sizeof(x))
const int num_buffers = 1;

const int xSize = 320;
const int ySize = 240;
const int framerate = 125;

Camera::Camera(const char *device, const int id, QSemaphore *sem, char* myBuffer) : QThread(NULL)
{
	this->id = id;
	this->sem = sem;
	this->myBuffer = myBuffer;
	fd = open(device, O_RDWR /* required */ | O_NONBLOCK, 0);
	qDebug("Camera initialised: dev %s, id %d, fd %d", device, id, fd);
	
	// set format
	struct v4l2_format fmt;
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_G_FMT, &fmt);
	
	fmt.fmt.pix.width = xSize;
	fmt.fmt.pix.height = ySize;
	//fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
	ioctl(fd, VIDIOC_S_FMT, &fmt);
	
	// set framerate
	v4l2_streamparm strp;
	ioctl(fd, VIDIOC_G_PARM, &strp);
	//qDebug("Time = %d/%d", strp.parm.capture.timeperframe.numerator, strp.parm.capture.timeperframe.denominator);
	strp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	strp.parm.capture.timeperframe.numerator = 1;
	strp.parm.capture.timeperframe.denominator = framerate;
	//qDebug("Time = %d/%d", strp.parm.capture.timeperframe.numerator, strp.parm.capture.timeperframe.denominator);
	//qDebug() << "ioctl " << ioctl(fd, VIDIOC_S_PARM, &strp) << " errno " << errno;
	
	
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = num_buffers;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	ioctl(fd, VIDIOC_REQBUFS, &req);

	buffers = (buffer*)(calloc(req.count, sizeof(*buffers)));

	for (int i = 0; i < req.count; ++i)
	{
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory      = V4L2_MEMORY_MMAP;
		buf.index       = i;

		ioctl(fd, VIDIOC_QUERYBUF, &buf);

		buffers[i].length = buf.length;
		buffers[i].start =
			mmap(NULL /* start anywhere */,
				buf.length,
				PROT_READ | PROT_WRITE /* required */,
				MAP_SHARED /* recommended */,
				fd, buf.m.offset);

	}
}

Camera::~Camera()
{
	stop();
	unsigned int i;
	for (i = 0; i < num_buffers; ++i)
		munmap(buffers[i].start, buffers[i].length);
	free(buffers);
	close(fd);
	qDebug("Camera stopped: id %d, fd %d", id, fd);
}

void Camera::run()
{
	unsigned int i;
	enum v4l2_buf_type type;

	for (i = 0; i < num_buffers; ++i)
	{
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (-1 == ioctl(fd, VIDIOC_QBUF, &buf))
			printf("VIDIOC_QBUF");
	}
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == ioctl(fd, VIDIOC_STREAMON, &type)){
		printf("VIDIOC_STREAMON");
	}
	stopped = false;
// 	qDebug() << "run id: " << id;
	loop();
}

void Camera::stop()
{
	stopped = true;
}

void Camera::loop()
{
	//setbuf(stdout, NULL);
	//setbuf(stderr, NULL);
	fd_set fds;
	struct timeval tv;
	int r;

	FD_ZERO(&fds);
	FD_SET(fd, &fds);

	while(!stopped)
	{
		/* Timeout. */
		tv.tv_sec = 2;
		tv.tv_usec = 0;

		r = select(fd + 1, &fds, NULL, NULL, &tv);

		if (r > 0) 
		{
			struct v4l2_buffer buf;

			CLEAR(buf);

			buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			buf.memory = V4L2_MEMORY_MMAP;
			ioctl(fd, VIDIOC_DQBUF, &buf);

			//qDebug() << "bufindex " << buf.index;
			doOurStuff(buffers[buf.index].start, buf.bytesused, buf.index);

			if (-1 == ioctl(fd, VIDIOC_QBUF, &buf))
					printf("VIDIOC_QBUF");
		}
		else
			fprintf(stderr, "select timeout\n");
	}
	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ioctl(fd, VIDIOC_STREAMOFF, &type);
}

void Camera::doOurStuff(void* bufStart, __u32 size, int index)
{
	unsigned int pos = 0, offset = id*xSize*ySize, val;
	char * buf = (char*) bufStart;
	
	for (int y = 0; y < ySize; y++)
	{
		for (int x = 0; x < xSize; x++)
		{
			val = buf[pos];
			//w->i.setPixel(x,y, qRgb(val, val, val));
			myBuffer[offset + x] = val;
			pos+=2;
		}
		offset += xSize;
	}
	sem->release(1);
}