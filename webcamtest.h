#ifndef webcamtest_H
#define webcamtest_H

#include <QtGui/QMainWindow>
#include "camarray.h"

class CamArray;

class webcamtest : public QMainWindow
{
Q_OBJECT
public:
    webcamtest();
    virtual ~webcamtest();
	CamArray * ca;
	QImage i;
	QFuture<void> future;
	
protected:
	virtual void paintEvent(QPaintEvent* e);

};

#endif // webcamtest_H