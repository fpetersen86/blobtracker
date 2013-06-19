#include "webcamtest.h"
#ifdef NOCUDA // W.T.F.???????!!!!! neccessary for proper compilation
#include "camarray.cu"
#endif
#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QAction>
#include "global.h"

webcamtest::webcamtest()
{
	winX = 720;
	winY = 480;
	imageWidth = 320;
	imageHeight= 240;
	xGrid = 0;
	yGrid = 0;
	myColor = QColor("#62b5ff");
	
	ca = new CamArray(this);
	ca->start();
	QWidget *w = new QWidget(NULL);
	ui=new Ui_settings();
	ui->setupUi(w);
	
	connect(ui->lcStrengthSpinBox, SIGNAL(valueChanged(double)),
			this, SLOT(setLcStrenght(double)));
	connect(ui->lcZoomSpinBox, SIGNAL(valueChanged(double)),
			this, SLOT(setLcZoom(double)));
	connect(ui->gridXSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setXGrid(int)));
	connect(ui->gridYSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setYGrid(int)));
	connect(ui->colEdit, SIGNAL(editingFinished()),
			this, SLOT(setColor()));
	ui->lcStrengthSpinBox->setValue(5.0);
	ui->lcZoomSpinBox->setValue(2.0);
	ui->gridXSpinBox->setValue(0);
	ui->gridXSpinBox->setMaximum(imageWidth/2);
	ui->gridYSpinBox->setValue(0);
	ui->gridYSpinBox->setMaximum(imageHeight/2);
	ui->colEdit->setText("62b5ff");
	qDebug() << xGrid << "Q" << imageWidth;
	//setAttribute(Qt::WA_DeleteOnClose, true);
	setAttribute(Qt::WA_QuitOnClose, true);
	w->setAttribute(Qt::WA_QuitOnClose, false);
	//w->setAttribute(Qt::WA_DeleteOnClose, true);
	
	w->show();
	
}

void webcamtest::resizeImage(int num)
{
	int x = xSize * num;
	int y = ySize * 2;
	i = QImage(x, y, QImage::Format_RGB32);
	resize(x,y);
}


webcamtest::~webcamtest()
{
	//w->close();
	ca->stop();
	ca->wait();
}

void webcamtest::paintEvent(QPaintEvent* e)
{
	QMainWindow::paintEvent(e);
	QPainter painter(this);
	
	painter.drawImage(i.rect(),i);
	
	//QColor myColor(98,181,255);
	QPen myPen;
	myPen.setColor(myColor);
	myPen.setWidth(1);
	myPen.setCosmetic(true);
	QBrush myBrush; 
	myBrush.setColor(myColor);
	painter.setPen(myPen);
	painter.setBrush(myBrush);
	
	int gridWidht, gridHeight;
	if (xGrid)
	{
		gridWidht = imageWidth / (xGrid+1);
		for (int x = 0; x < imageWidth; x+= gridWidht)
			painter.drawLine(x, winY/2, x, winY);
	}
		
	if (yGrid) 
	{
		gridHeight = imageHeight / (yGrid+1);
		for (int y = winY/2; y < winY; y+= gridHeight)
			painter.drawLine(0, y, imageWidth, y);
	}
}

void webcamtest::setColor()
{
	myColor.setNamedColor("#" + ui->colEdit->displayText());
}



#include "webcamtest.moc"
