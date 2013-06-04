#ifndef CAMARRAY_H
#define CAMARRAY_H

#include "camera.h"
#include <Qt/QtCore>
#include "webcamtest.h"

class webcamtest;
class Camera;

class CamArray
{

public:
    CamArray(webcamtest* p);
    virtual ~CamArray();
	Camera *cams[64]; // 64 = max# of cameras in Linux
	int numCams;
};

#endif // CAMARRAY_H
