#include "webcamtest.h"

#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QAction>

webcamtest::webcamtest()
{
	winX = 320;
	winY = 480;
	
	resize(winX,winY);
	ca = new CamArray(this);
	i = QImage(winX, winY, QImage::Format_RGB32);
	ca->start();
}

webcamtest::~webcamtest()
{
	ca->stop();
	ca->wait();
	
}

void webcamtest::paintEvent(QPaintEvent* e)
{
	QPainter painter(this);
	painter.drawImage(QRect(0,0,winX,winY),i);
}


#include "webcamtest.moc"
