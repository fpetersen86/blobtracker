#include "webcamtest.h"

#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QAction>

webcamtest::webcamtest()
{
	winX = 720;
	winY = 480;
	imageWidth = 320;
	imageHeight= 240;
	
	resize(winX,winY);
	ca = new CamArray(this);
	i = QImage(winX, winY, QImage::Format_RGB32);
	ca->start();
	QWidget w = new QWidget(this);
	w.setGeometry(imageWidth, 0, winX-imageWidth, winY);
	ui->setupUi(w);
	
	connect(ui->lcStrengthSpinBox, SIGNAL(valueChanged(double)),
			this, SLOT(setLcStrenght(float)));
	connect(ui->lcZoomSpinBox, SIGNAL(valueChanged(double)),
			this, SLOT(setLcZoom(float)));
	connect(ui->gridXSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setXGrid(int)));
	connect(ui->GridYSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setYGrid(int)));
	ui->lcStrengthSpinBox->setValue(5.0);
	ui->lcZoomSpinBox->setValue(2.0);
	ui->gridXSpinBox->setValue(0);
	ui->gridYSpinBox->setValue(0);
	
	
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
	int gridWidht, gridHeight;
	if (xGrid)
	{
		gridWidht = imageWidth / xGrid;
		for (int x = 0; x < imageWidth; x+= gridWidht)
			painter.drawLine(x, 0, x, winY);
	}
		
	if (yGrid) 
	{
		gridHeight = imageHeight / yGrid;
		for (int y = 0; y < winY; y+= gridWidht)
			painter.drawLine(0, y, imageWidth, y);
	}
	
	QMainWindow::paintEvent(e);
}


#include "webcamtest.moc"
