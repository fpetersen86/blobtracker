#include "webcamtest.h"

#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QAction>

webcamtest::webcamtest()
{
	resize(320,240);
	ca = new CamArray(this);
	i = QImage(320, 240, QImage::Format_RGB32);
//future = QtConcurrent::run(ca->cams[0], &Camera::capture);
}

webcamtest::~webcamtest()
{
	future.cancel();
	ca->cams[0]->stop();
	
}

// void webcamtest::paintEvent(QPaintEvent* e)
// {
// 	QPainter painter(this);
// 	painter.drawImage(QRect(0,0,320,240),i);
// }


#include "webcamtest.moc"
