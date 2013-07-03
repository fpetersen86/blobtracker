#include "camarray.h"
#include <stdio.h>
#include "webcamtest.h"
#include "global.h"

#ifdef CUDA
#include "utility_environment.h"
#include <cuda.h>





__global__ void lensCorrection(unsigned char *image,
							   unsigned char *output,
							   int width,
							   int height,
							   int width2,
							   int height2,
							   float strength,
							   float zoom)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width2 + x;                             // index within linear array
    int offset = blockIdx.z * width * height;
    
    x += (width-width2)/2;
	y += (height-height2)/2;
	
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



__global__ void rotate(unsigned char *image,
					   unsigned char *output,
					   camSettings *settings,
					   int width,
					   int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;                             // index within linear array
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



__global__ void median(unsigned char *input,
					   unsigned char *output,
					   int width,
					   int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;                             // index within linear array
    int offset = blockIdx.z * width * height;

	// compute cells needed for update (neighbors + central element)
	int borderFlag = (x > 0);                              // boolean values enable border handling without thread divergence
	unsigned char leftNeighb = input[offset + elemID - borderFlag];
	borderFlag = (x < (width - 1));
	unsigned char rightNeighb = input[offset + elemID + borderFlag];
	borderFlag = -(y > 0);									// unary minus turns boolean value into boolean bitwise mask
	unsigned char topNeighb = input[offset + elemID - (borderFlag & width)];	
	borderFlag = -(y < (height - 1));
	unsigned char bottomNeighb = input[offset + elemID + (borderFlag & width)];
	unsigned char currElement = input[offset + elemID];
	
	output[elemID + offset] = (currElement + leftNeighb + rightNeighb + topNeighb + bottomNeighb) / 5;
}



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
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;                             // index within linear array
	
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


__global__ void findBlobs_1(unsigned char *input,
							yRange *output,
							int width,
							int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x > width) return;
	
	bool b = false;
	int start;
	int end;
	
	int counter = x * height / 2;
	for(int i = 0; i < height; i++)
	{
		if(input[i] == 255)
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

__global__ void findBlobs_2(yRange *input,
							xyRange *output,
							int width,
							int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x > width) return;
	
	int countLeft = x * height;
	int countRight = x * height + height / 2;
	
	int counter = x * height;
	int yMin;
	bool merged = false;
	
	while(true)
	{
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
			counter++;
		}
		else if( input[countLeft].y2 > input[countRight].y2 || !input[countRight].y2)
		{
			output[counter].y1 = input[countRight].y1;
			output[counter].y2 = input[countRight].y2;
			if (!merged)
			{
				output[counter].x1 = output[counter].x2 = x*2+1;
			}
			countRight++;
			counter++;
			merged = false;
		}
		else if (!input[countLeft].y2)
		{
			output[counter].y1 = input[countLeft].y1;
			output[counter].y2 = input[countLeft].y2;
			if (!merged)
			{
				output[counter].x1 = output[counter].x2 = x*2;
			}
			countLeft++;
			counter++;
			merged = false;
		}
	}
	output[counter].y2 = 0;
}

#endif


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
	//threshold = 20;
	
	initBuffers();
	
	if (!imgTest)
		for(int i = 0; i < numCams; i++)
		{
			c = camList.at(i);
			cams[i] = new Camera(d.absoluteFilePath(c).toStdString().c_str(), i, sem, h_a, &(h_s[i]));
			//h_s[i].angle = 1.0;
			h_s[i].xOffset = 0;
			h_s[i].yOffset = 0;
		}

	
}

#ifdef CUDA
void CamArray::initBuffers() {
	int fieldStateSize = canvY * canvX * sizeof(bool);
	int yRangeSize = canvY * canvX * sizeof(yRange) / 2;
	int xyRangeSize = canvY * canvX * sizeof(xyRange);
	
	//host buffers
	cudaMallocHost(&h_a, bufferImgSize, 0x04);
	cudaMallocHost(&h_b, bufferImgSize);
	cudaMallocHost(&h_c, bufferImgSize);
	cudaMallocHost(&h_d, bufferStitchedImg);
	cudaMallocHost(&h_s, bufferSettings, 0x04);
	cudaMallocHost(&h_blobMap, fieldStateSize);
	
	
	//device buffers
	cudaMalloc((void**) &d_a, bufferImgSize);
	cudaMalloc((void**) &d_b, bufferImgSize);
	cudaMalloc((void**) &d_c, bufferImgSize);
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
void CamArray::mainloop()
{
	dim3 cudaBlockSize(16,16);  // image is subdivided into rectangular tiles for parallelism - this variable controls tile size
	dim3 cudaGridSize(xSize2/cudaBlockSize.x, ySize2/cudaBlockSize.y, numCams);
	
	dim3 cudaBlockSize2(16,16);  // image is subdivided into rectangular tiles for parallelism - this variable controls tile size
	dim3 cudaGridSize2(canvX/cudaBlockSize2.x, canvY/cudaBlockSize2.y);
	
	dim3 cudaBlockSize3(canvX);
	dim3 cudaGridSize3(1);
	
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
		
// 		lensCorrection<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize, xSize2, ySize2, lcStrength, lcZoom);
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
			
			findBlobs_2<<<cudaGridSize3, cudaBlockSize3>>>(d_yRanges, d_xyRanges, canvX, canvY);
			handleCUDAerror(__LINE__);
			
			findblob();
			trackBlobs();
		}
// 		qDebug() << "-------------- cuda ready --------------";
 		output();
	}
}

#endif

#ifdef NOCUDA
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


int CamArray::isBlob(int x, int y, Blob * bob, int depth)
{
// 	qDebug() << "isBlob " << x << " " << y;
	if (depth > 10000)
	{
		return 1;
// 		qDebug() << "panic!!!\nBlob at X: " << x*blobstep 
// 					   << "  Y: " << y*blobstep
// 					   << " depth: " << bob->maxDepth << "value is " << blobMap[y*canvX+x] ;
	}
	switch (h_blobMap[y*canvX+x])
	{
		case false:
			return 0;
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
	return 1;
}


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


