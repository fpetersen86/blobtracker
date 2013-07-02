#ifndef webcamtest_H
#define webcamtest_H

#include <QtGui/QMainWindow>
#include "camarray.h"
#include "ui_settings.h"
#include "ui_camsettings.h"

//class CamArray;

class CamSettingsUi : public QWidget
{
Q_OBJECT
	Ui_camSettings *ui;
public:
    CamSettingsUi(Camera *cam, QWidget* parent = 0);
    virtual ~CamSettingsUi() {};
};

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
	bool imgTestMode;
    void resizeMe();
	
protected:
	virtual void paintEvent(QPaintEvent* e);
	
public slots:
	void setLcStrenght(double d) {ca->lcStrength = d;};
	void setLcZoom(double d) {ca->lcZoom = d;};
	void setXGrid(int i) {xGrid = i;};
	void setYGrid(int i) {yGrid = i;};
	void setColor();
	void setThreshold(int i) {ca->threshold = i;};
	void setViewMode(int b) {ca->viewmode = b;resizeMe();};
	void setCanvX(int cx) {ca->canvX = cx; resizeMe();};
	void setCanvY(int cy) {ca->canvY = cy; resizeMe();};
	void setCanvX() {ca->canvX = ui->canvXSpinBox->value(); resizeMe();};
	void setCanvY() {ca->canvY = ui->canvYSpinBox->value(); resizeMe();};
	void setCanvOffX(int i) {ca->canvOffX = i;};
	void setCanvOffX2(int i) {ca->canvOffX2 = i;};
	void setCanvOffY(int i) {ca->canvOffY = i;};
	void setCanvOffY2(int i) {ca->canvOffY2 = i;};

};

#endif // webcamtest_H
