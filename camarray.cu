#include "camarray.h"
#include <stdio.h>
#include "webcamtest.h"
#include "global.h"

#ifdef CUDA
#include "utility_environment.h"
#include <cuda.h>



/*-------------------------------------------------------------------------------------------/

lens correction kernel
This kernel corrects distorted images due to uncorrected camera lenses.

parameters:
	unsigned char *image	all input camera images consecutively in one buffer
	unsigned char *output	buffer for the corrected images
	int width				width of one camera image
	int height				height of one camera image
	float strength			correction strength -> how much shall each image be corrected
	float zoom				each image gets zoomed by this amount, so that only useful data remains

/-------------------------------------------------------------------------------------------*/

__global__ void lensCorrection(unsigned char *image,
							   unsigned char *output,
							   int width,
							   int height,
							   float strength,
							   float zoom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; 
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;
    int offset = blockIdx.z * width * height;
	
	int halfWidth = width / 2;
	int halfHeight = height / 2;
	float correctionRadius = sqrtf(width * width + height * height) / strength;
	int newX = x - halfWidth;
	int newY = y - halfHeight;

	float distance = sqrtf(newX * newX + newY * newY);
	float r = distance / correctionRadius;
	
	float theta;
	if(r != 0)
	{
		theta = atanf(r)/r;
	} else {
		theta = 1;
	}
	
	int sourceX = halfWidth + theta * newX * zoom;
	int sourceY = halfHeight + theta * newY * zoom;
	
	output[elemID + offset] = image[sourceY*width + sourceX + offset];
}


/*-------------------------------------------------------------------------------------------/

median kernel
This kernel reduces noise

parameters:
	unsigned char *input	all input camera images consecutively in one buffer
	unsigned char *output	buffer for the rotated images
	int width				width of one camera image
	int height				height of one camera image

/-------------------------------------------------------------------------------------------*/

__global__ void median(unsigned char *input,
					   unsigned char *output,
					   int width,
					   int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;
    int offset = blockIdx.z * width * height;

    //border handling stolen from assignment 4 ;)
	int borderFlag = (x > 0);
	unsigned char leftNeighb = input[offset + elemID - borderFlag];
	borderFlag = (x < (width - 1));
	unsigned char rightNeighb = input[offset + elemID + borderFlag];
	borderFlag = -(y > 0);
	unsigned char topNeighb = input[offset + elemID - (borderFlag & width)];	
	borderFlag = -(y < (height - 1));
	unsigned char bottomNeighb = input[offset + elemID + (borderFlag & width)];
	unsigned char currElement = input[offset + elemID];
	
	output[elemID + offset] = (currElement + leftNeighb + rightNeighb + topNeighb + bottomNeighb) / 5;
}


/*-------------------------------------------------------------------------------------------/

lens rotation kernel
This kernel rotates each camera if necessary. It is necessary if they are not mounted perfectly.

parameters:
	unsigned char *image	all input camera images consecutively in one buffer
	unsigned char *output	buffer for the corrected images
	camSettings *settings	a struct containing settings for each camera
	int width				width of one camera image
	int height				height of one camera image

/-------------------------------------------------------------------------------------------*/

__global__ void rotate(unsigned char *image,
					   unsigned char *output,
					   camSettings *settings,
					   int width,
					   int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;
    int offset = blockIdx.z * width * height;
	
	
	float sin_, cos_;
	int ydiff, xdiff;
	int xCenter = width/2;
	int yCenter = height/2;
	
	sin_ = sin(settings[blockIdx.z].angle);
	cos_ = cos(settings[blockIdx.z].angle);
	ydiff = yCenter - y;
	xdiff = xCenter - x;
	
	int myX = xCenter + (-xdiff * cos_ - ydiff * sin_);
	int myY = yCenter + (-ydiff * cos_ + xdiff * sin_);
	
	if (myY < 0 || myY >= height || myX < 0 || myX >= width) // ekliges borderhandling :(
	{
		//output[offset + elemID] = 255;
		return;
	}
    
	output[offset + elemID] = image[offset + myY*width + myX];
}


/*-------------------------------------------------------------------------------------------/

stitching kernel
This kernel merges all the input images into one large binary image

parameters:
	unsigned char *image	all input camera images consecutively in one buffer
	unsigned char *output	buffer for the corrected images
	camSettings *settings	a struct containing settings for each camera
	bool *blobMap			a bool map that matches white pixels to true
	int width				width of the final image
	int height				height of the final image
	int camWidth			width of one camera image
	int camHeight			height of one camera image
	int numCams				number of cameras used
	int threshold			the threshold for the binary conversion
	int ignoX1				all pixels left of this value are ignored -> always set to black
	int ignoX2				all pixels right of this value are ignored -> always set to black
	int ignoY1				all pixels above of this value are ignored -> always set to black
	int ignoY2				all pixels below of this value are ignored -> always set to black

/-------------------------------------------------------------------------------------------*/

__global__ void stitch(unsigned char *image,
					   unsigned char *output,
					   camSettings *settings,
					   bool *blobMap,
					   int width,
					   int height,
					   int camWidth,
					   int camHeight,
					   int numCams,
					   int threshold,
					   int ignoX1,
					   int ignoX2,
					   int ignoY1,
					   int ignoY2
  					)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;
	
	if(x < ignoX1 || x > ignoX2 || y < ignoY1 || y > ignoY2)
	{
		output[elemID] = 0;
		blobMap[elemID] = false;
	} else {
		if(elemID > width * height) return;
		
		int val = 0;
		int usedCams = 0;
		
		for (int i = 0; i < numCams; i++)
		{
			int xOffset = settings[i].xOffset;
			int yOffset = settings[i].yOffset;
			int offset = i * camWidth * camHeight;
			
			if (x > xOffset && x < (xOffset + camWidth) && y > yOffset && y < (yOffset + camHeight)){
				val += image[offset - xOffset + x + (-yOffset + y)*camWidth];
				//val += 40;
				usedCams += 1;
			}
		}
		if(usedCams == 0){
			usedCams++;
		}
		
		if (val / usedCams <= threshold)
		{
			output[elemID] = 255;
			blobMap[elemID] = true;
		} else {
			output[elemID] = 0;
			blobMap[elemID] = false;
		}
	}
}

/*-------------------------------------------------------------------------------------------/

blob-detection kernel 1
This kernel looks at each column and saves each range of white pixels as a yRange (see camarray.h) to the output buffer

parameters:
	unsigned char *input	the binary input image
	yRange *output			buffer for all found ranges
	int width				width of the image
	int height				height of the image

/-------------------------------------------------------------------------------------------*/

__global__ void findBlobs_1(unsigned char *input,
							yRange *output,
							int width,
							int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(x > width) return;
	
	bool b = false;
	int start = 0;
	int end = 0;

	int counter = x * height / 2;
	for(int i = 0; i < height; i++)
	{
		if(input[i*width+x] != 0)
		{
			if(!b)
			{
				b = true;
				start = i;
			} else {
				end = i;
			}
			
		} else {
			if(b)
			{
				b = false;
				output[counter].y1 = start;
				output[counter].y2 = end;
				counter++;
			}
		}
	}
	if(b)
	{
		b = false;
		output[counter].y1 = start;
		output[counter].y2 = end;
		counter++;
	}
	output[counter].y2 = 0;
}


/*-------------------------------------------------------------------------------------------/

blob-detection kernel 2
This kernel compares the yRanges of two columns and merges those that touch each other

parameters:
	yRange *input			all yRanges found by findBlobs_1
	xyRange *output			buffer for all found xyRanges (see camarray.h)
	int width				width of the image
	int height				height of the image

/-------------------------------------------------------------------------------------------*/

__global__ void findBlobs_2(yRange *input,
							xyRange *output,
							int width,
							int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x > width / 2) return;
	
	int countLeft = x * height;
	int countRight = x * height + height / 2;
	
	int counter = x * height;
	int yMin;
	bool merged = false;
	
	while(true)
	{
// 		if (input[countRight].y2 || input[countRight].y2)
// 			printf("%d, y: %d - %d \t\t  R_ID: y: %d - %d\n", x, input[countLeft].y1 , input[countLeft].y2 , input[countRight].y1 , input[countRight].y2 );
// 		break;
	
		if(input[countRight].y2 == input[countLeft].y2 == 0)
			break;
		if(	input[countLeft].y1 <= input[countRight].y2 && input[countLeft].y1 >= input[countRight].y1 || 
			input[countLeft].y2 >= input[countRight].y1 && input[countLeft].y2 <= input[countRight].y2 ||
			input[countLeft].y2 >= input[countRight].y2 && input[countLeft].y1 <= input[countRight].y1 ||
			input[countLeft].y2 <= input[countRight].y2 && input[countLeft].y1 >= input[countRight].y1)
		{
			merged = true;
			yMin = fminf(input[countLeft].y1, input[countRight].y1);
			output[counter].x1 = x*2;
			output[counter].x2 = x*2+1;
			
			if (input[countLeft].y2 > input[countRight+1].y1 && input[countRight+1].y2 != 0)
			{
				input[countRight+1].y1 = yMin;
				countRight++;
				continue;
			}
			if (input[countRight].y2 > input[countLeft+1].y1 && input[countLeft+1].y2 != 0)
			{
				input[countLeft+1].y1 = yMin;
				countLeft++;
				continue;
			}
			merged = false;
			countLeft++;
			countRight++;
			printf("id: %d, x: %d - %d, y: %d - %d\n", x, output[counter].x1 , output[counter].x2 , output[counter].y1 , output[counter].y2 );
			counter++;
		}
		else if( input[countLeft].y2 > input[countRight].y2 || !input[countRight].y2)
		{
			output[counter].y1 = input[countRight].y1;
			output[counter].y2 = input[countRight].y2;
			if (output[counter].y2 == 0)
				printf("bah1");
			if (!merged)
			{
				output[counter].x1 = output[counter].x2 = x*2+1;
			}
			countRight++;
			printf("id: %d, x: %d - %d, y: %d - %d\n", x, output[counter].x1 , output[counter].x2 , output[counter].y1 , output[counter].y2 );
			counter++;
			merged = false;
		}
		else if (!input[countLeft].y2)
		{
			output[counter].y1 = input[countLeft].y1;
			output[counter].y2 = input[countLeft].y2;
			if (output[counter].y2 == 0)
				printf("bah2");
			if (!merged)
			{
				output[counter].x1 = output[counter].x2 = x*2;
			}
			countLeft++;
			printf("id: %d, x: %d - %d, y: %d - %d\n", x, output[counter].x1 , output[counter].x2 , output[counter].y1 , output[counter].y2 );
			counter++;
			merged = false;
		}
	}
	output[counter].y2 = 0;
	if (counter - x * height && 1 == 0)
	{ 
		printf("thread %d, counter= %d, last= %d\n", x, counter-(x * height), output[counter - x * height - 2].y2);
		
	}
}

#endif


/*-------------------------------------------------------------------------------------------/

blob-detection function 3
This function merges all xyRanges found by findBlobs_2 that touch each other, so that we get the complete blobs

parameters:
	none. It accesses the host buffer h_xyRanges directly

/-------------------------------------------------------------------------------------------*/

void CamArray::findBlobs_3()
{
	QList<Blob*> bb;
	Blob *bob;
	
	for (int i=0; i < canvX; i++)
	{
		for (int j=0; j < 1; j++)
		{
			if (h_xyRanges[i*canvY+j].y2 != 0)
			{
				qDebug(" x %d %d %d", i, j, h_xyRanges[i*canvY+j].y2);
			}
		}
	}
			
	
	for (int i=0; i < canvX; i++)
	{
		int j = 0;
		while(1)
		{
			if (h_xyRanges[i*canvY+j].y2 == 0)
			{
				break;
			}
			bob = new Blob;
			bob->x = h_xyRanges[i*canvY+j].x1;
			bob->x2 = h_xyRanges[i*canvY+j].x2;
			bob->y = h_xyRanges[i*canvY+j].y1;
			bob->y2 = h_xyRanges[i*canvY+j].y2;
			bob->maxDepth = 50;
			bb.append(bob);
			j++;
// 			qDebug() << "bob";
		}
	}
	
	for (int i=0; i < bb.size(); i++)
	{
		for (int j=i+1; j < bb.size();)
		{
			if (mergeBlobs(bb.at(i), bb.at(j)))
			{
				Blob *a = bb.at(i), *b = bb.at(j);
				a->x = min(a->x, b->x);
				a->x2 = max(a->x2, b->x2);
				a->y = min(a->y, b->y);
				a->y2 = max(a->y2, b->y2);
				bb.removeAt(j);
				continue;
			}
			j++;
		}
	}
	
	blobs = bb;
}


/*-------------------------------------------------------------------------------------------/

blob-merging function
Helper function of findBlobs_3. It checks, if two blobs can be merged

parameters:
	Blob *blob1				The first blob to be merged
	Blob* blob2				The other blob to be merged
	
output:
	true 					if they can be merged
	false					if not
/-------------------------------------------------------------------------------------------*/

bool CamArray::mergeBlobs(Blob *blob1, Blob* blob2)
{
	QRect a(blob1->x, blob1->y, blob1->x2-blob1->x, blob1->y2 - blob1->y);
	QRect b(blob2->x -1, blob2->y-1, blob2->x2-blob2->x+1, blob2->y2 - blob2->y+1);
	return a.intersects(b);
}



/*-------------------------------------------------------------------------------------------/

constructor
it initializes all cameras that are found in the system as a Camera object

parameters:
	none
	
/-------------------------------------------------------------------------------------------*/


CamArray::CamArray(webcamtest* p, int testimages) : QThread(p)
{
	QSettings s;
	canvX = s.value("canvX", 640).toInt();
	canvY = s.value("canvY", 480).toInt();
	
	viewmode = false;
	w = p; // All hail the mighty alphabet ;)
	//initialise cams
	QStringList camList;
	QDir d("/dev/");
	if (!testimages)
	{
		imgTest = false;
		d.setFilter(QDir::System);
		d.setNameFilters(QStringList("*video*"));
		camList = d.entryList();
		numCams = camList.size();
	}
	else
	{
		imgTest = true;
		numCams = testimages;
	}
	
	p->resizeImage(numCams);
	QString c;
	sem = new QSemaphore(numCams);
	sem->acquire(numCams);
	
	bufferImgSize = xSize * ySize * numCams * sizeof(char);
	bufferSettings = numCams * sizeof(camSettings);
	qDebug("buffer   x: %d   y: %d", canvX, canvY);
	bufferStitchedImg = canvX * canvY * sizeof(char);
	
	initBuffers();
	
	if (!imgTest)
	{
		for(int i = 0; i < numCams; i++)
		{
			c = camList.at(i);
			cams[i] = new Camera(d.absoluteFilePath(c).toStdString().c_str(), i, sem, h_a, &(h_s[i]));
			//h_s[i].angle = 1.0;
			h_s[i].xOffset = 0;
			h_s[i].yOffset = 0;
		}
	}
}

#ifdef CUDA
void CamArray::initBuffers() {
	int fieldStateSize = canvY * canvX * sizeof(bool);
	int yRangeSize = canvY * canvX * sizeof(yRange) / 2;
	xyRangeSize = canvY * canvX * sizeof(xyRange);
	
	//host buffers
	cudaMallocHost(&h_a, bufferImgSize, 0x04);
	cudaMallocHost(&h_b, bufferImgSize);
	cudaMallocHost(&h_c, bufferImgSize);
	cudaMallocHost(&h_d, bufferStitchedImg);
	cudaMallocHost(&h_s, bufferSettings, 0x04);
	cudaMallocHost(&h_blobMap, fieldStateSize);
	cudaMallocHost(&h_xyRanges, xyRangeSize);
	
	
	//device buffers
	cudaMalloc((void**) &d_a, bufferImgSize);
	cudaMalloc((void**) &d_b, bufferImgSize);
	cudaMalloc((void**) &d_c, bufferImgSize);
	qDebug("x %d, y %d, = %d", canvX, canvY, bufferStitchedImg);
	cudaMalloc((void**) &d_d, bufferStitchedImg);
	cudaMalloc((void**) &d_s, bufferSettings);
	cudaMalloc((void**) &d_blobMap, fieldStateSize);
	cudaMalloc((void**) &d_yRanges, yRangeSize);
	cudaMalloc((void**) &d_xyRanges, xyRangeSize);
}
#else //NOCUDA
void CamArray::initBuffers() {
	h_a = reinterpret_cast<unsigned char*>(malloc(bufferImgSize));
	h_b = reinterpret_cast<unsigned char*>(malloc(bufferImgSize));
	h_c = reinterpret_cast<unsigned char*>(malloc(bufferImgSize));
	h_d = reinterpret_cast<unsigned char*>(malloc(bufferStitchedImg));
	h_s = reinterpret_cast<camSettings*>(malloc(bufferSettings));
}
#endif


/*-------------------------------------------------------------------------------------------/

run function
this code is executed when CamArray is started as a thread
It does some setups and starts the main loop of the thread.

parameters:
	none
	
/-------------------------------------------------------------------------------------------*/

void CamArray::run()
{
	stopped = false;
	
	//start capturing
	if (!imgTest)
		for(int i = 0; i < numCams; i++)
		{
			cams[i]->start();
		}
	
	mainloop();
}

#ifdef CUDA


/*-------------------------------------------------------------------------------------------/

mainloop of the program
The mainloop waits for all cameras to capture one frame and then copies the images to the gpu and launches the kernels and the blobtracking function. It optionally copies the output image back to the host, so that the results can be shown on screen.

parameters:
	none
	
output:
	it optionally shows the resulting image and blobs
/-------------------------------------------------------------------------------------------*/

void CamArray::mainloop()
{
	dim3 cudaBlockSize(16,16);  // image is subdivided into rectangular tiles for parallelism - this variable controls tile size
	dim3 cudaGridSize(xSize2/cudaBlockSize.x, ySize2/cudaBlockSize.y, numCams);
	
	dim3 cudaBlockSize2(16,16);  // image is subdivided into rectangular tiles for parallelism - this variable controls tile size
	dim3 cudaGridSize2(canvX/cudaBlockSize2.x, canvY/cudaBlockSize2.y);
	
	dim3 cudaBlockSize3(canvX);
	dim3 cudaGridSize3(1);
	
	dim3 cudaBlockSize4(canvX/2);
	dim3 cudaGridSize4(1);
	
	qDebug("x %d   y %d", canvX, canvY);
	qDebug("grid: %d %d   block: %d %d", cudaGridSize2.x, cudaGridSize2.y, cudaBlockSize2.x, cudaBlockSize2.y);
	
// 	cudaMemcpy( d_s, h_s, bufferSettings, cudaMemcpyHostToDevice );
// 	handleCUDAerror(__LINE__);
	
	while(!stopped)
	{
		sem->acquire(numCams);
// 		qDebug() << "-------------- collected --------------";
		
		cudaMemcpy( d_a, h_a, bufferImgSize, cudaMemcpyHostToDevice );
		handleCUDAerror(__LINE__);
		
// 		lensCorrection<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize, lcStrength, lcZoom);
// 		handleCUDAerror(__LINE__);
		
		median<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize);
		handleCUDAerror(__LINE__);
		
		cudaMemcpy( d_s, h_s, bufferSettings, cudaMemcpyHostToDevice );
		handleCUDAerror(__LINE__);
		
		rotate<<<cudaGridSize, cudaBlockSize>>>(d_b, d_c, d_s, xSize, ySize);
		handleCUDAerror(__LINE__);
		
		if(viewmode == 0 || viewmode == 1){
			cudaMemcpy( h_b, d_b, bufferImgSize, cudaMemcpyDeviceToHost );
			handleCUDAerror(__LINE__);
			
			cudaMemcpy( h_c, d_c, bufferImgSize, cudaMemcpyDeviceToHost );
			handleCUDAerror(__LINE__);
		}
		
		if(viewmode == 2){
			//qDebug("x1: %d   X2: %d   y1: %d   y2: %d", canvOffX, canvOffX2, canvOffY, canvOffY);
			stitch<<<cudaGridSize2, cudaBlockSize2>>>(d_c, d_d, d_s, d_blobMap, canvX, canvY, xSize, ySize, numCams, threshold, canvOffX, canvOffX2, canvOffY, canvOffY2);
			handleCUDAerror(__LINE__);
			
			cudaMemcpy( h_d, d_d, bufferStitchedImg, cudaMemcpyDeviceToHost );
			handleCUDAerror(__LINE__);
			cudaMemcpy( h_blobMap, d_blobMap, bufferStitchedImg, cudaMemcpyDeviceToHost );
			handleCUDAerror(__LINE__);
			
			
		
			findBlobs_1<<<cudaGridSize3, cudaBlockSize3>>>(d_d, d_yRanges, canvX, canvY);
			handleCUDAerror(__LINE__);
			
			findBlobs_2<<<cudaGridSize4, cudaBlockSize4>>>(d_yRanges, d_xyRanges, canvX, canvY);
			handleCUDAerror(__LINE__);
			
			cudaMemcpy( h_xyRanges, d_xyRanges, xyRangeSize, cudaMemcpyDeviceToHost );
			handleCUDAerror(__LINE__);
			
			findBlobs_3();
// 			findblob();
			trackBlobs();
			
		}
// 		qDebug() << "-------------- cuda ready --------------";
 		output();
   		if (myBreak)
			break;
	}
}

#endif

#ifdef NOCUDA
/*-------------------------------------------------------------------------------------------/
cpu implementation of the mainloop and the kernels
/-------------------------------------------------------------------------------------------*/

void CamArray::mainloop()
{
	while(!stopped)
	{
		if (!imgTest)
			sem->acquire(numCams);
		else 
			msleep(8);
		
		int width = xSize;
		int height = ySize;
		int width2 = xSize2;
		int height2 = ySize2;
		float strength = lcStrength;
		float zoom = lcZoom;
		int myX, myY;
		
		int offset = 0;
		// lensCorrection
		for( int n = 0; n < numCams; n++)
		{
			for (int y = 0; y < ySize2; y++)
			{
				for (int x = 0; x < xSize2; x++)
				{
					myX = x;
					myY = y;
					int elemID = myY*width2 + myX;                              // index within linear array

					myX += (width-width2)/2;
					myY += (height-height2)/2;
					
					int halfWidth = width / 2;
					int halfHeight = height / 2;
					float correctionRadius = sqrt(width * width + height * height) / strength;
					int newX = myX - halfWidth;
					int newY = myY - halfHeight;

					float distance = sqrt(newX * newX + newY * newY);
					float r = distance / correctionRadius;
					
					float theta;
					if(r != 0)
					{
						theta = atan(r)/r;
					} else {
						theta = 1;
					}
					
					int sourceX = halfWidth + theta * newX * zoom;
					int sourceY = halfHeight + theta * newY * zoom;
					
					h_b[offset + elemID] = h_a[offset + sourceY*width + sourceX];
					//qDebug("elemID: %d   X: %d myX: %d sourceX: %d   Y: %d myY: %d sourceY: %d", elemID, x, myX, sourceX, y, myY, sourceY);
				}
			}
			offset   += xSize*ySize;
		}
		
		// rotate
		float sin_, cos_;
		offset = 0;
		int ydiff, xdiff;
		int xCenter = width/2, yCenter = height/2;
		
		for( int n = 0; n < numCams; n++)
		{
			sin_ = sin(h_s[n].angle);
			cos_ = cos(h_s[n].angle);
			for (int y = 0; y < ySize2; y++)
			{
				ydiff = yCenter - y;
				for (int x = 0; x < xSize2; x++)
				{
					xdiff = xCenter - x;
					myX = xCenter + (-xdiff * cos_ - ydiff * sin_);
					if (myX < 0 || myX >= width)
						continue;
					myY = yCenter + (-ydiff * cos_ + xdiff * sin_);
					if (myY < 0 || myY >= height)
						continue;
					h_c[offset + y*width + x] = h_b[offset + myY*width + myX];
				}
			}
			offset += ySize*xSize;
		}
		
		//stitch
		for (int y = 0; y < canvY; y++)
		{
			ydiff = yCenter - y;
			for (int x = 0; x < canvX; x++)
			{
				int val = 0;
				int usedCams = 0;
				
				for (int i = 0; i < numCams; i++)
				{
					int xOffset = h_s[i].xOffset;
					int yOffset = h_s[i].yOffset;
					int offset = i * xSize * ySize;
					
					if (x > xOffset && x <= (xOffset + xSize) && y > yOffset && y <= (yOffset + ySize)){
						val += h_c[offset - xOffset + x + (-yOffset + y)*xSize];
						//val += 40;
						usedCams++;
					}
				}
// 				if(usedCams > 1){
// 					qDebug("val: %d   cams: %d \n", val, usedCams);
// 				}
				
				if(usedCams == 0){
					usedCams++;
				}
				
				h_d[y*canvX + x] = val / usedCams;
			}
		}
		
		findblob();
		output();
		trackBlobs();
	}
}
#endif
/*-------------------------------------------------------------------------------------------/

cpu blob-detection function
It iterates over the blobMap and checks if it detects a white pixel. If it finds one it starts the recursive function isBlob() for that pixel. After that it puts the found blob (if it has the right size -> not too big/small) in the blob list.

parameters:
	none
	
output:
	none
/-------------------------------------------------------------------------------------------*/
void CamArray::findblob()
{
	Blob * bob;
	int b = 0;
	int w = 0;
	
	for (int y = 1; y < canvY / blobstep - 1; y++)
	{
		for (int x = 1; x < canvX /blobstep - 1; x++)
		{
			if ( h_blobMap[y*canvX+x] != true)
			{
				b++;
				continue;
			}
			w++;
			bob = new Blob;
			bob->x = bob->x2 = x;
			bob->y = bob->y2 = y;
			bob->maxDepth = 0;
			isBlob(x,y,bob, 0);
			
			if (bob->x2-bob->x > 15 &&
				bob->y2-bob->y > 15 &&
				bob->x2-bob->x < 70 &&
				bob->y2-bob->y < 70)
			{
				blobs.append(bob);
// 				qDebug() << "Blob at X:" << bob->x*blobstep << "-" << bob->x2*blobstep
//  						 << "  Y:" << bob->y*blobstep << "-" << bob->y2*blobstep
//  						 << " depth: " << bob->maxDepth;
			}
			else
			{
				delete bob;
			}
		}
	}
// 	qDebug("blobs: %d   b: %d   w: %d", blobs.count(), b, w);
}

/*-------------------------------------------------------------------------------------------/

cpu blob-expansion function
this function starts at one pixel coordinate and from there it looks at the neighboring pixels if they are also part of the blob. if they are it merges them into the blob. Each visited pixel is set to black, so that findblob() doesn't look at those again.

parameters:
	int x				starting x coordinate
	int y				starting y coordinate
	Blob * bob			the blob that gets expanded
	int depth			current depth
	
output:
	none
/-------------------------------------------------------------------------------------------*/

void CamArray::isBlob(int x, int y, Blob * bob, int depth)
{
// 	qDebug() << "isBlob " << x << " " << y;
	if (depth > 10000)
	{
		return;
// 		qDebug() << "panic!!!\nBlob at X: " << x*blobstep 
// 					   << "  Y: " << y*blobstep
// 					   << " depth: " << bob->maxDepth << "value is " << blobMap[y*canvX+x] ;
	}
	switch (h_blobMap[y*canvX+x])
	{
		case false:
			return;
		case true:
			h_blobMap[y*canvX+x] = false;
	}
	if (x < bob->x)
		bob->x = x;
	if (x > bob->x2)
		bob->x2 = x;
	if (y < bob->y)
		bob->y = y;
	if (y > bob->y2)
		bob->y2 = y;
	if (depth > bob->maxDepth)
		bob->maxDepth = depth;
	
	
	
	if (x > 0)
		isBlob(x-1,y, bob, depth+1);
	if (x < canvX/blobstep -1)
		isBlob(x+1,y, bob, depth+1);
	if (y > 0)
		isBlob(x,y-1, bob, depth+1);
	if (y < canvY/blobstep -1)
		isBlob(x,y+1, bob, depth+1);
}


/*-------------------------------------------------------------------------------------------/

cpu blob-tracking function
this function compares the found blobs with the blobs from the frame before and tries to match the new ones to the old ones. If a blob is considered to be the same as one of the old blobs it's id and color are set to the old one's.

parameters:
	none
	
output:
	none
/-------------------------------------------------------------------------------------------*/

void CamArray::trackBlobs()
{
	foreach(Blob *b, blobs)
	{
		int i = 0;
		int x = (b->x2 + b->x)/2;
		int y = (b->y2 + b->y)/2;
		bool colored = false;
		
		foreach(Blob* b2, blobs2)
		{
			int x2 = (b2->x2 + b2->x)/2;
			int y2 = (b2->y2 + b2->y)/2;
			if( sqrtf((x-x2)*(x-x2) + (y-y2)*(y-y2)) < maxDistance )
			{
				b->color = b2->color;
				colored = true;
				b->id = b2->id;
				//blobs.removeAt(i);
				break;
			}
			i++;
		}
		
		if(!colored)
		{
			int c1 = qrand() % 256;
			int c2 = qrand() % 105 + 151;
			
			switch(qrand() % 6)
			{
				case 0:
					b->color = QColor(0, c1, c2);
					break;
				case 1:
					b->color = QColor(0, c2, c1);
					break;
				case 2:
					b->color = QColor(c1, 0, c2);
					break;
				case 3:
					b->color = QColor(c2, 0, c1);
					break;
				case 4:
					b->color = QColor(c1, c2, 0);
					break;
				case 5:
					b->color = QColor(c2, c1, 0);
					break;
			}
		}
	}
	
 	qDeleteAll(blobs2.begin(), blobs2.end());
	blobs2.clear();
 	
 	while(!blobs.empty())
 	{
		blobs2.append(blobs.takeFirst());
 	}
 	blobs.clear();
}


/*-------------------------------------------------------------------------------------------/

image painting function
this is a rather inefficient way to paint the camera images and kernel output to screen. It has three modes in which it displays different things.

parameters:
	none
	
output:
	none
/-------------------------------------------------------------------------------------------*/

void CamArray::output()
{
	int offset = 0, xOffset = 0, xOffset2 = 0;
	switch (viewmode)
	{
		case 0:
			for( int n = 0; n < numCams; n++)
			{
				for (int y = 0; y < ySize; y++)
				{
					for (int x = 0; x < xSize; x++)
					{
						int val = h_a[offset+y*xSize+x];
						w->i.setPixel(xOffset+x,y, qRgb(val, val, val));
					}
				}

				for (int y = 0; y < ySize2; y++)
				{
					for (int x = 0; x < xSize2; x++)
					{
						int val = h_b[offset+y*xSize2+x];
						w->i.setPixel(xOffset2+x,y+ySize, qRgb(val, val, val));
					}
				}
				for (int y = 0; y < ySize2; y++)
				{
					for (int x = 0; x < xSize2; x++)
					{
						int val = h_c[offset+y*xSize2+x];
						w->i.setPixel(xOffset2+x,y+ySize*2, qRgb(val, val, val));
					}
				}
				for (int y = 0; y < ySize2; y++)
				{
					for (int x = 0; x < xSize2; x++)
					{
						unsigned char val = h_c[offset+y*xSize2+x];
						if (val <= threshold)
							val = 0;
						else
							val = 255;
						w->i.setPixel(xOffset2+x,y+ySize*3, qRgb(val, val, val));
					}
				}
				xOffset  += xSize;
				xOffset2 += xSize2;
				offset   += xSize*ySize;
			}
			
			break;
		case 1:
			int yOffset, xMax, yMax, x, y;
			w->i.fill(Qt::black);
			
			for( int n = 0; n < numCams; n++)
			{
				xOffset2 = h_s[n].xOffset;
				yOffset = h_s[n].yOffset;
				//yMax = yOffset+ySize > canvY ? canvY - yOffset : ySize;
				for (y = 0; y < ySize; y++)
				{
					if (y+yOffset < 0 || y+yOffset >= canvY )
						continue;
					//xMax = xOffset+xSize > canvX ? canvX - xOffset : xSize; //xMax = xSize;
					for (x = 0; x < xSize; x++)
					{
						if (x+xOffset2 < 0 || x+xOffset2 >= canvX)
							continue;
						int val = h_c[offset+y*xSize+x];
						if(val)
							w->i.setPixel(x+xOffset2,y+yOffset, qRgb(val, val, val));
					}
				}
				xOffset  += xSize;
				offset   += xSize*ySize;
			}
			break;
		case 2:
			for (y = 0; y < canvY; y++)
			{
				for (x = 0; x < canvX; x++)
				{
					int val = h_d[y*canvX+x];
						w->i.setPixel(x,y, qRgb(val, val, val));
				}
			}
			break;
	}
	w->update();
	//qDebug("available: %d", sem->available());
}

void CamArray::stop()
{
	stopped = true;
}


/*-------------------------------------------------------------------------------------------/

file loading function
Instead of video capture this function can also load images from disc. It was only used for early testing and it is not known whether it still works or not.

parameters:
	none
	
return:
	none
/-------------------------------------------------------------------------------------------*/

void CamArray::loadFiles()
{
	QImage fileImage;
	int offset = 0;
	for (int j = 1; j < QApplication::arguments().size(); j++)
	{
		fileImage = QImage(QApplication::arguments().at(j)).scaled(xSize, ySize,
												Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		qDebug() << QApplication::arguments().at(j) << " xsize " << fileImage.width() ;
		for (int y = 0; y < ySize; y++)
		{
			for (int x = 0; x < xSize; x++)
			{
				h_a[offset+y*xSize+x] = qGray(fileImage.pixel(x,y));
			}
		}
		offset += (ySize * xSize);
	}
	w->update();
}


/*-------------------------------------------------------------------------------------------/

deconstructor
it does, what a deconstructor shall do. unload everything.

/-------------------------------------------------------------------------------------------*/

CamArray::~CamArray()
{
	for (int i = 0; i < numCams; i++)
	{
		cams[i]->stop();
	}
	for (int i = 0; i < numCams; i++)
	{
		cams[i]->wait();
		delete cams[i];
	}
	qDebug() << "CamArray stopped";
	
	// free memory buffers
#ifndef NOCUDA
	cudaFree(d_a);
	handleCUDAerror(__LINE__);
	cudaFree(d_b);
	handleCUDAerror(__LINE__);
	cudaFree(d_c);
	handleCUDAerror(__LINE__);
	cudaFree(d_d);
	handleCUDAerror(__LINE__);
	cudaFree(d_s);
	handleCUDAerror(__LINE__);
	cudaFree(d_blobMap);
	handleCUDAerror(__LINE__);
	cudaFree(d_yRanges);
	handleCUDAerror(__LINE__);
	cudaFree(d_xyRanges);
	handleCUDAerror(__LINE__);
	
	cudaFreeHost(h_a);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_b);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_c);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_d);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_s);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_blobMap);
	handleCUDAerror(__LINE__);
#else
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_d);
	free(h_s);
	free(h_blobMap);
#endif
	qDebug("Memory deallocated successfully\n");
}


