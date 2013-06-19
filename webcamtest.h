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
	QWidget *w;
	void resizeImage(int num);
	
	
private:
	int winX, winY;
	int xGrid;
	int yGrid;
	int imageWidth;
	int imageHeight;
	QColor myColor;
	
protected:
	virtual void paintEvent(QPaintEvent* e);
	
public slots:
	void setLcStrenght(double d) {ca->lcStrength = d;};
	void setLcZoom(double d) {ca->lcZoom = d;};
	void setXGrid(int i) {xGrid = i;};
	void setYGrid(int i) {yGrid = i;};
	void setColor();

};

#endif // webcamtest_H
