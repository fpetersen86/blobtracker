#include "webcamtest.h"
#ifdef NOCUDA // W.T.F.???????!!!!! neccessary for proper compilation
#include "camarray.cu"
#endif
#include <QtGui/QLabel>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QAction>
#include "global.h"

/*-------------------------------------------------------------------------------------------/

constructor
This constructs the GUI, loads some settings and finaly starts the CamArray.

parameters:
	none
	
/-------------------------------------------------------------------------------------------*/

webcamtest::webcamtest()
{
	QCoreApplication::setApplicationName("Blobtracker");
	QCoreApplication::setOrganizationName("Stuff");

	QSettings s;
	imgTestMode = QApplication::argc() > 1;
	winX = 720;
	winY = 480;
	imageWidth = 320;
	imageHeight= 240;
	xGrid = 0;
	yGrid = 0;
	myColor = QColor("#62b5ff");
	
	ca = new CamArray(this, QApplication::arguments().size() - 1);
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
	connect(ui->threshSlider, SIGNAL(valueChanged(int)),
			this, SLOT(setThreshold(int)));
	connect(ui->viewModeComboBox, SIGNAL(currentIndexChanged(int)),
			this, SLOT(setViewMode(int)));
	connect(ui->canvXSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvX(int)));
	connect(ui->canvYSpinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvY(int)));
	connect(ui->canvOffX1spinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvOffX(int)));
	connect(ui->canvOffX2spinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvOffX2(int)));
	connect(ui->canvOffY1spinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvOffY(int)));
	connect(ui->canvOffY2spinBox, SIGNAL(valueChanged(int)),
			this, SLOT(setCanvOffY2(int)));
	
	ui->viewModeComboBox->setCurrentIndex(defaultMode);
	
	if(imgTestMode) 
	{
		connect(ui->lcStrengthSpinBox, SIGNAL(valueChanged(double)),
				this, SLOT(update()));
		connect(ui->lcZoomSpinBox, SIGNAL(valueChanged(double)),
				this, SLOT(update()));
		connect(ui->gridXSpinBox, SIGNAL(valueChanged(int)),
				this, SLOT(update()));
		connect(ui->gridYSpinBox, SIGNAL(valueChanged(int)),
				this, SLOT(update()));
		connect(ui->colEdit, SIGNAL(editingFinished()),
				this, SLOT(update()));
		connect(ui->threshSlider, SIGNAL(valueChanged(int)),
				this, SLOT(update()));
		resizeImage(QApplication::argc() - 1);
		ca->loadFiles();
	}
	
	ui->lcStrengthSpinBox->setValue(s.value("lcStrength", 5.0).toDouble());
	ui->lcZoomSpinBox->setValue(s.value("lcZoom", 2.0).toDouble());
	ui->gridXSpinBox->setValue(0);
	ui->gridXSpinBox->setMaximum(imageWidth/2);
	ui->gridYSpinBox->setValue(0);
	ui->gridYSpinBox->setMaximum(imageHeight/2);
	ui->colEdit->setText(s.value("gridColor", "62b5ff").toString());
	ui->canvXSpinBox->setValue(s.value("canvX", 640).toInt());
	ui->canvYSpinBox->setValue(s.value("canvY", 480).toInt());
	ui->threshSlider->setValue(s.value("threshold").toInt());
	ui->threshBox->setValue(s.value("threshold").toInt());
	ui->canvOffX1spinBox->setValue(s.value("canvOffX1", 0).toInt());
	ui->canvOffX2spinBox->setValue(s.value("canvOffX2", 20).toInt());
	ui->canvOffY1spinBox->setValue(s.value("canvOffY1", 0).toInt());
	ui->canvOffY2spinBox->setValue(s.value("canvOffY2", 20).toInt());
	
	for(int i = 0; i < ca->numCams; i++)
	{
		ui->verticalLayout->addWidget(new CamSettingsUi(ca->cams[i], this));
	}
	//setAttribute(Qt::WA_DeleteOnClose, true);
	setAttribute(Qt::WA_QuitOnClose, true);
	w->setAttribute(Qt::WA_QuitOnClose, false);
	//w->setAttribute(Qt::WA_DeleteOnClose, true);
	
	ca->start();
	w->show();
}

/*-------------------------------------------------------------------------------------------/
This function takes care of resizing the Output Window

parameters:
	int num				Number of cameras.
output:
	none
/-------------------------------------------------------------------------------------------*/

void webcamtest::resizeImage(int num)
{
	int x = xSize * num;
	int y = ySize * imgRows;
	i = QImage(x, y, QImage::Format_RGB32);
	resize(x,y);
}

/*-------------------------------------------------------------------------------------------/
This function also takes care of resizing the Output Window

parameters:
	none
output:
	none
/-------------------------------------------------------------------------------------------*/

void webcamtest::resizeMe()
{
	if (ca->viewmode)
	{
		i = QImage(ca->canvX, ca->canvX, QImage::Format_RGB32);
		resize(ca->canvX,ca->canvY);
	}
	else
		resizeImage(ca->numCams);
}

/*-------------------------------------------------------------------------------------------/

deconstructor
it does, what a deconstructor shall do. Stop the CamArray, save Settings.

/-------------------------------------------------------------------------------------------*/

webcamtest::~webcamtest()
{
	QSettings s;
	s.setValue("lcStrength", ui->lcStrengthSpinBox->value());
	s.setValue("lcZoom", ui->lcZoomSpinBox->value());
	s.setValue("gridColor", ui->colEdit->text());
	s.setValue("canvX", ui->canvXSpinBox->value());
	s.setValue("canvY", ui->canvYSpinBox->value());
	s.setValue("threshold", ui->threshSlider->value());
	s.setValue("canvOffX1", ui->canvOffX1spinBox->value());
	s.setValue("canvOffX2", ui->canvOffX2spinBox->value());
	s.setValue("canvOffY1", ui->canvOffY1spinBox->value());
	s.setValue("canvOffY2", ui->canvOffY2spinBox->value());
	//w->close();
	ca->stop();
	ca->wait();
}

/*-------------------------------------------------------------------------------------------/

Paintevent
Paints the output window.


parameters:
	QPaintEvent* e		This is given to us by Qt and contains information that we just forward.
	
output:
	none
/-------------------------------------------------------------------------------------------*/

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
		for (int x = 0; x < imageWidth * ca->numCams; x+= gridWidht)
			painter.drawLine(x, winY/2, x, winY);
	}
		
	if (yGrid) 
	{
		gridHeight = imageHeight / (yGrid+1);
		for (int y = winY/2; y < winY; y+= gridHeight)
			painter.drawLine(0, y, imageWidth*ca->numCams, y);
	}
	if (ca->viewmode == 1)
	{
		painter.drawLine(ca->canvOffX, 0, ca->canvOffX, ca->canvY);
		painter.drawLine(ca->canvOffX2, 0, ca->canvOffX2, ca->canvY);
		painter.drawLine(0, ca->canvOffY, ca->canvX, ca->canvOffY);
		painter.drawLine(0, ca->canvOffY2, ca->canvX, ca->canvOffY2);
	}
	if (ca->viewmode == 2)
	{
		myPen.setWidth(2);
		foreach(Blob *bob, ca->blobs2)
		{
			//qDebug() << "Blobs " << ca->blobs.count();
			QRect r(bob->x * blobstep, bob->y*blobstep,
					(bob->x2 - bob->x)*blobstep, (bob->y2 - bob->y)*blobstep);
			//qDebug() << bob->color;
			myPen.setColor(bob->color);
			myBrush.setColor(bob->color);
			painter.setPen(myPen);
			painter.setBrush(myBrush);
			painter.drawRect(r);
			//qDebug() << "drew " << r;
			//delete bob;
		}
	}
}

void webcamtest::setColor()
{
	myColor.setNamedColor("#" + ui->colEdit->displayText());
}

/*-------------------------------------------------------------------------------------------/

Constructor of CamSettingsUi
This constructs the GUI part for camera-specific settings.

parameters:
	Camera* cam			Which Camera object to handle.
	QWidget* parent		parent Widget (webcamtest)

/-------------------------------------------------------------------------------------------*/
CamSettingsUi::CamSettingsUi(Camera* cam, QWidget* parent): QWidget(parent)
{
	ui = new Ui_camSettings();
	ui->setupUi(this);
	ui->angleSpinBox->setMaximum(4*PI);
	ui->angleSpinBox->setMinimum(-4*PI);
	ui->angleSpinBox->setDecimals(6);
	ui->angleSpinBox->setSingleStep(PI/360);
	
	connect(ui->angleSpinBox, SIGNAL(valueChanged(double)),cam, SLOT(setAngle(double)));
	connect(ui->xOffSpinBox, SIGNAL(valueChanged(int)), cam, SLOT(setXOffset(int)));
	connect(ui->yOffSpinBox, SIGNAL(valueChanged(int)), cam, SLOT(setYOffset(int)));
	
	QSettings s;
	s.beginGroup("cam::"+QString::number(cam->getID()));
	ui->angleSpinBox->setValue(s.value("angle", 0.0).toDouble());
	ui->xOffSpinBox->setValue(s.value("xoffset", xSize * cam->getID()).toInt());
	ui->yOffSpinBox->setValue(s.value("yoffset", 0).toInt());
	s.endGroup();
	
	show();
}


#include "webcamtest.moc"
