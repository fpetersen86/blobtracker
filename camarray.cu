#include "camarray.h"
#include <stdio.h>
#include "webcamtest.h"
#include "global.h"

#ifdef CUDA
#include "utility_environment.h"
#include <cuda.h>


//GPU Kernel
__global__ void find_features(float *feature_values, int* feature_positions, const int width, const int height, const int eta, const int beta_2) 
{
    const int x = blockIdx.x * beta_2;
    const int y = blockIdx.y*beta_2;    
    const int offset = (blockIdx.x + blockIdx.y*gridDim.x)*eta;
    const int double_offset = offset*2;
    int min_pos;
    float min_value = 1.0f;
    float value; 
    
    for(int i = 0; i<eta; i++)
    {        
        feature_positions[double_offset + i*2] = x+i;
        feature_positions[double_offset + i*2 + 1] = y;
        value = feature_values[offset + i] = tex2D(float1_tex, x+i, y).x;
             
        if(value <= min_value)
        {        
            min_value = value;
            min_pos = i;
        }
    }    
    
    if( y <= height - beta_2 && x< width - beta_2)
    {            
        for(int j = 0; j<beta_2; j++)
            for(int i = 0; i<beta_2; i++)
            {        
                value = tex2D(float1_tex, float(x+i) + 0.5f, float(y+j) + 0.5f).x;
                if(value > min_value)
                {
                    feature_positions[double_offset + min_pos*2] = x+i;
                    feature_positions[double_offset + min_pos*2 + 1] = y+j;
                    feature_values[offset + min_pos] = value;
                    min_pos = find_min_pos(feature_values, offset, eta);
                    min_value = feature_values[offset + min_pos];
                }
            }

        for(int i = 0; i<eta; i++)
        {
            if(feature_values[offset + i] == 0.0f)
            {
                feature_positions[double_offset + i*2] = 0;
                feature_positions[double_offset + i*2 + 1] = 0;
            }
        }
    }
}



__global__ void lensCorrection(char *image,
							   char *output,
							   int width,
							   int height,
							   int width2,
							   int height2,
							   float strength,
							   float zoom
							  )
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

__global__ void rotate(char *image,
					   char *output,
					   camSettings *settings,
					   int width,
					   int height
					  )
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
		return;
    
	output[offset + elemID] = image[offset + myY*width + myX];
}

__global__ void median(char *input, char *output, int width, int height)
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
	
	output[elemID + offset] = (currElement + leftNeighb + rightNeighb + leftNeighb + bottomNeighb) / 5;
}


__global__ void stitch(char *image, char *output, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;         // coordinates within 2d array follow from block index and thread index within block
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int elemID = y*width + x;                             // index within linear array
    
    if(elemID > width * height) return;
    
	output[elemID] = image[elemID];
}


#endif


CamArray::CamArray(webcamtest* p, int testimages) : QThread(p)
{
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
	//host buffers
	cudaMallocHost(&h_a, bufferImgSize, 0x04);
	cudaMallocHost(&h_b, bufferImgSize);
	cudaMallocHost(&h_c, bufferImgSize);
	cudaMallocHost(&h_d, bufferStitchedImg);
	cudaMallocHost(&h_s, bufferSettings, 0x04);
	
	//device buffers
	cudaMalloc((void**) &d_a, bufferImgSize);
	cudaMalloc((void**) &d_b, bufferImgSize);
	cudaMalloc((void**) &d_c, bufferImgSize);
	cudaMalloc((void**) &d_d, bufferStitchedImg);
	cudaMalloc((void**) &d_s, bufferSettings);
}
#else //NOCUDA
void CamArray::initBuffers() {
	h_a = reinterpret_cast<char*>(malloc(bufferImgSize));
	h_b = reinterpret_cast<char*>(malloc(bufferImgSize));
	h_c = reinterpret_cast<char*>(malloc(bufferImgSize));
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
	dim3 cudaGridSize2(canvX/cudaBlockSize.x, canvY/cudaBlockSize.y);
	
	qDebug("x %d   y %d", canvX, canvY);
	qDebug("grid: %d %d   block: %d %d", cudaGridSize2.x, cudaGridSize2.y, cudaBlockSize2.x, cudaBlockSize2.y);
	
// 	cudaMemcpy( d_s, h_s, bufferSettings, cudaMemcpyHostToDevice );
// 	handleCUDAerror(__LINE__);
	
	while(!stopped)
	{
		if (!imgTest){
			sem->acquire(numCams);
		}
			else
		{
			qDebug() << "frame dropped";
			msleep(8);
		}
		cudaMemcpy( d_a, h_a, bufferImgSize, cudaMemcpyHostToDevice );
		handleCUDAerror(__LINE__);
		
		lensCorrection<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize, xSize2, ySize2, lcStrength, lcZoom);
		handleCUDAerror(__LINE__);
		
// 		median<<<cudaGridSize, cudaBlockSize>>>(d_a, d_b, xSize, ySize);
// 		handleCUDAerror(__LINE__);
		
		cudaMemcpy( d_s, h_s, bufferSettings, cudaMemcpyHostToDevice );
		handleCUDAerror(__LINE__);
		
		rotate<<<cudaGridSize, cudaBlockSize>>>(d_b, d_c, d_s, xSize, ySize);
		handleCUDAerror(__LINE__);
		
		stitch<<<cudaGridSize2, cudaBlockSize2>>>(d_c, d_d, canvX, canvY);
		handleCUDAerror(__LINE__);
		
// 		cudaMemcpy( h_b, d_b, bufferImgSize, cudaMemcpyDeviceToHost );
// 		handleCUDAerror(__LINE__);
// 		
// 		cudaMemcpy( h_c, d_c, bufferImgSize, cudaMemcpyDeviceToHost );
// 		handleCUDAerror(__LINE__);
		
		cudaMemcpy( h_d, d_d, bufferStitchedImg, cudaMemcpyDeviceToHost );
		handleCUDAerror(__LINE__);
		
		//test code - what does it do?
		int beta_2 = 8; //???
		int eta 20; //???
		float* d_feature_values;
		int *d_feature_positions;
		float* h_feature_values;
		int *h_feature_positions;
		int data_width = (canvX/beta_2);
		int data_height = (canvY/beta_2);
		int feature_nb = data_width*data_height*eta;

		dim3 extraction_block (1, 1, 1);
		dim3 extraction_grid (_width/beta_2, _height/beta_2, 1);

		cudaMalloc((void**)&d_feature_positions, 2*feature_nb*sizeof(int));
		cudaMalloc((void**)&d_feature_values, feature_nb*sizeof(float));
		
		cudaMallocHost((void**)&h_feature_positions, 2*feature_nb*sizeof(int));
		cudaMallocHost((void**)&h_feature_values, feature_nb*sizeof(float));
		
		find_features<<<cudaGridSize2,cudaBlockSize2>>>(d_feature_values, d_feature_positions, canvX, canvY, eta, beta_2);
		
		cudaMemcpy( h_feature_positions, d_feature_positions, 2*feature_nb*sizeof(int), cudaMemcpyDeviceToHost );
		cudaMemcpy( h_feature_values, d_feature_values, feature_nb*sizeof(float), cudaMemcpyDeviceToHost );
		
		for (int i = 0; i < feature_nb; i++)
		{
			qDebug() << "blob " << i << " @ x " << h_feature_positions[i*2] << " y " 
					 << h_feature_positions[i*2+1] << " with value = " << h_feature_values[i];
		}
		cudaFree(d_feature_positions);
		cudaFree(d_feature_values);
		cudaFreeHost(h_feature_positions);
		cudaFreeHost(h_feature_values);
		//end test
		
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
		
		
		output();
	}
}
#endif
inline bool CamArray::white(int x, int y) 
{
	return h_c[y*xSize+x] > threshold;
}

void CamArray::findblob()
{
	bool maybeBlobs[xSize/blobstep][ySize/blobstep];
	int offset;
	for (int y = 0; y < ySize / blobstep; y++)
		for (int x = 0; x < xSize / blobstep; x++)
			maybeBlobs[x][y] = white(x*blobstep,y*blobstep);
	
	for (int y = 1; y < ySize - 1; y++)
		for (int x = 1; x < xSize - 1; x++)
		{
			if (!white(x*blobstep, y*blobstep))
				continue;

		}
	
		
		
		
		
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
					int val = h_d[y*xSize+x];
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
	cudaFree(d_s);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_a);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_b);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_c);
	handleCUDAerror(__LINE__);
	cudaFreeHost(h_s);
	handleCUDAerror(__LINE__);
#else
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_s);
#endif
	qDebug("Memory deallocated successfully\n");
}

