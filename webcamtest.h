#ifndef webcamtest_H
#define webcamtest_H

#include <QtGui/QMainWindow>
#include "camarray.h"
#include "ui_settings.h"

//class CamArray;

class webcamtest : public QMainWindow
{
Q_OBJECT
public:
    webcamtest();
    virtual ~webcamtest();
	CamArray * ca;
	QImage i;
	QFuture<void> future;
	Ui_settings *ui;
	
	
private:
	int winX, winY;
	int xGrid;
	int yGrid;
	int imageWidth;
	int imageHeight;
	
protected:
	virtual void paintEvent(QPaintEvent* e);
	
public slots:
	void setLcStrenght(float f) {ca->lcStrength = f;};
	void setLcZoom(float f) {ca->lcZoom = f;};
	void setXGrid(int i) {xGrid = i;};
	void setYGrid(int i) {yGrid = i;};
    void thi(int);

};

#endif // webcamtest_H
