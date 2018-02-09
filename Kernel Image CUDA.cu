/*
 ============================================================================
 Name        : Kernel.cu
 Author      : Samuele
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#define TILE_WIDTH 16
#define RESULTS_NUM 7
#define KERNEL_SIZE 3
#define INPUT_TILE_WIDTH TILE_WIDTH + KERNEL_SIZE - 1

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include "PPM.h"
#include "kernels.h"




static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



__constant__ float deviceKernels[3*3 * RESULTS_NUM];


__global__ void kernelConvolution(float* img, float** output ,const int imageWidth, const int imageHeight, const int imageChannels){
	__shared__ float imgTile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH]; //condivisa a livello di blocco


	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;


	if((row < imageHeight) && (col < imageWidth)){
		int xIndex;
		int yIndex;
		int kIndex;

		int tileRow = row % TILE_WIDTH;
		int tileCol = col % TILE_WIDTH;

		float element;
		float sum[RESULTS_NUM];
		for(int h = 0; h < RESULTS_NUM; h++)
			sum[h] = 0;

		//printf("row: %i, col: %i,tileCell: %i, tileRow: %i, tileCol: %i\n",row,col,tileCell,tileRow,tileCol);
		for(int c = 0; c < imageChannels; c++){

			/* caricamento del'input tile in shared memory */
			element = img[(row * imageWidth + col) * imageChannels + c];




			//controllo i lati dei tile di bordo
			if(row == 0){ //bordo immagine superiore

				if((col == 0)||(tileCol == 0))//angolo immagine superiore sinistro
					imgTile[tileRow][tileCol] = element;

				if((col == imageWidth - 1)||(tileCol == TILE_WIDTH - 1))//angolo immagine superiore destro
					imgTile[tileRow][tileCol+2] = element;

				imgTile[tileRow][tileCol+1] = element;
			}
			else if(row == imageHeight - 1){ //bordo immagine inferiore

				if((col == 0)||(tileCol == 0))//angolo immagine inferiore sinistro
					imgTile[tileRow+2][tileCol] = element;

				if((col == imageWidth - 1)||(tileCol == TILE_WIDTH - 1))//angolo immagine inferiore destro
					imgTile[tileRow+2][tileCol+2] = element;

				imgTile[tileRow+2][tileCol+1] = element;
			}

			if(col == 0){//bordo immagine sinistro
				if(tileRow == 0)
					imgTile[tileRow][tileCol] = element;
				if(tileRow == TILE_WIDTH -1)
					imgTile[tileRow+2][tileCol] = element;
				imgTile[tileRow+1][tileCol] = element;
			}
			else if(col == imageWidth - 1){//bordo immagine destro
				if(tileRow == 0)
					imgTile[tileRow][tileCol+2] = element;
				if(tileRow == TILE_WIDTH -1)
					imgTile[tileRow+2][tileCol+2] = element;
				imgTile[tileRow+1][tileCol+2] = element;
			}

			//controllo i lati dei tile centrali
			if((tileRow == 0) && (row != 0)){ //lato superiore

				if(tileCol == 0) // angolo superiore sinistro
					imgTile[tileRow][tileCol]=img[((row-1) * imageWidth + (col-1)) * imageChannels + c];

				if(tileCol == TILE_WIDTH - 1) //angolo superiore destro
					imgTile[tileRow][tileCol+2]=img[((row-1) * imageWidth + (col+1)) * imageChannels + c];

				imgTile[tileRow][tileCol+1]=img[((row-1) * imageWidth + col) * imageChannels + c];
			}


			if((tileRow == TILE_WIDTH - 1) && (row != imageHeight - 1)){ //lato inferiore

				if((tileCol == 0))//angolo inferiore sinistro
					imgTile[tileRow+2][tileCol]=img[((row+1) * imageWidth + (col-1)) * imageChannels + c];

				if((tileCol == TILE_WIDTH - 1)) //angolo inferiore destro
					imgTile[tileRow+2][tileCol+2]=img[((row+1) * imageWidth + (col+1)) * imageChannels + c];

				imgTile[tileRow+2][tileCol+1]=img[((row+1) * imageWidth + col) * imageChannels + c];
			}

			if((tileCol == 0)&&(col > 0)) //lato sinistro
				imgTile[tileRow+1][tileCol]=img[((row) * imageWidth + (col-1)) * imageChannels + c];

			if((tileCol == TILE_WIDTH - 1)&&(col < imageWidth)) //lato destro
				imgTile[tileRow+1][tileCol+2]=img[((row) * imageWidth + (col+1)) * imageChannels + c];



			imgTile[tileRow+1][tileCol+1] = element;
			__syncthreads();


			/*calcolo dell'output*/
			for(int i = 0; i < KERNEL_SIZE; i++){
				yIndex = ((row - 1 + i) < 0) ? 0 : ((row - 1 + i) >= imageHeight) ? imageHeight - 1 : row - 1 + i;
				for(int j = 0; j < KERNEL_SIZE; j++){
					xIndex = ((col - 1 + j) < 0) ? 0 : ((col - 1 + j) >= imageWidth) ? imageWidth - 1 : col - 1 + j;
					kIndex = (2 - i)*3 + (2 -j);

					for(int h = 0; h < RESULTS_NUM; h++)
						//sum[h] += img[(yIndex * imageWidth + xIndex) * imageChannels + c] * deviceKernels[kIndex + 3*3*h];
						sum[h] += imgTile[tileRow + i][tileCol + j] * deviceKernels[kIndex + 3*3*h];
				}
			}
			for(int x = 0; x < RESULTS_NUM; x++){
				//output[x][(row * imageWidth + col) * imageChannels + c] = imgTile[tileRow][tileCol];
				output[x][(row * imageWidth + col) * imageChannels + c] = sum[x];
				sum[x] = 0;
			}
			__syncthreads();
		}
	}

}


int main(int argc,  char** argv){

    Image_t* inputImage = PPM_import(argv[1]);

    const int imageWidth = Image_getWidth(inputImage);
    const int imageHeight = Image_getHeight(inputImage);
    const int imageChannels = Image_getChannels(inputImage);
    const int imageDataSize = sizeof(float)*imageWidth*imageHeight*imageChannels;
    int i;

    float *hostInput;
    float *deviceInput;
    float **deviceOutput;
    float *hostOutput[RESULTS_NUM];
    float *hostKernels[RESULTS_NUM];

    hostInput = Image_getData(inputImage);

    //creo le immagini di output
    Image_t* outputImages[RESULTS_NUM];

    std::cout << imageWidth << "x" << imageHeight << "-" << imageChannels << std::endl;

    //copio in constant memory i dati delle matrici kernel
    hostKernels[0] = identity;
    hostKernels[1] = edge1;
    hostKernels[2] = edge2;
    hostKernels[3] = edge3;
    hostKernels[4] = sharpen;
    hostKernels[5] = boxblur;
    hostKernels[6] = gaussianblur3;

    //alloco la memoria (global) per contenere i dati dell'immagine e ci copio i dati
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceInput,imageDataSize));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceInput,hostInput,imageDataSize,cudaMemcpyHostToDevice));

    float *devicePointers[RESULTS_NUM]; //array di puntatori a locazioni di memoria device
    for(i = 0; i < RESULTS_NUM; i++){
    	outputImages[i] = Image_new(imageWidth,imageHeight,imageChannels);
    	hostOutput[i] = Image_getData(outputImages[i]);
    	CUDA_CHECK_RETURN(cudaMalloc((void**)&devicePointers[i],imageDataSize)); //alloco le varie locazioni di memoria device per le immagini e salvo il puntatore nell'array
    	CUDA_CHECK_RETURN(cudaMemcpy(devicePointers[i],hostOutput[i],imageDataSize,cudaMemcpyHostToDevice)); //trasferisco i dati immagini nelle locazioni appena create
    	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceKernels, hostKernels[i],sizeof(float)*3*3,sizeof(float)*3*3*i)); //trasferisco i dati dei kernel in constant memory
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceOutput,sizeof(float*) * RESULTS_NUM)); //alloco memoria device per ospitare l'array di locazioni di memoria device
    CUDA_CHECK_RETURN(cudaMemcpy(deviceOutput,devicePointers,sizeof(float*) * RESULTS_NUM,cudaMemcpyHostToDevice));

    dim3 gridDim(ceil((float)imageWidth / TILE_WIDTH),ceil((float)imageHeight / TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH);
    printf("%i,%i\n",(imageWidth / TILE_WIDTH) + 1,(imageHeight / TILE_WIDTH) + 1);
    kernelConvolution<<<gridDim,blockDim>>>(deviceInput,deviceOutput,imageWidth,imageHeight,imageChannels);

    cudaDeviceSynchronize();

    //copio indietro i risultati e imposto i dati immagine
    for(i = 0; i < RESULTS_NUM; i++){
    	CUDA_CHECK_RETURN(cudaMemcpy(hostOutput[i],devicePointers[i],imageDataSize,cudaMemcpyDeviceToHost));
    	Image_setData(outputImages[i],hostOutput[i]);
    }


    //esporto le immagini
	PPM_export("processed/identity.ppm",outputImages[0]);
    PPM_export("processed/edge1.ppm",outputImages[1]);
    PPM_export("processed/edge2.ppm",outputImages[2]);
    PPM_export("processed/edge3.ppm",outputImages[3]);
    PPM_export("processed/sharpen.ppm",outputImages[4]);
    PPM_export("processed/boxblur.ppm",outputImages[5]);
    PPM_export("processed/gaussianblur3.ppm",outputImages[6]);

    //libero la memoria
    CUDA_CHECK_RETURN(cudaFree(deviceInput));
    for(int i = 0; i < RESULTS_NUM; i++)
    	CUDA_CHECK_RETURN(cudaFree(devicePointers[i]));
    CUDA_CHECK_RETURN(cudaFree(deviceOutput));









    /*
	float *hostIdentityOutput;
	float *hostEdge1Output;
	float *hostEdge2Output;
	float *hostEdge3Output;
	float *hostSharpenOutput;
	float *hostBoxblurOutput;
	float *hostGaussianblur3Output;

	float *deviceIdentityOutput;
	float *deviceEdge1Output;
	float *deviceEdge2Output;
	float *deviceEdge3Output;
	float *deviceSharpenOutput;
	float *deviceBoxblurOutput;
	float *deviceGaussianblur3Output;

	Image_t* identityOutput = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* edge1Output = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* edge2Output = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* edge3Output = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* sharpenOutput = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* boxblurOutput = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* gaussianblur3Output = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* gaussianblur5Output = Image_new(imageWidth,imageHeight,imageChannels);
	Image_t* unsharpOutput = Image_new(imageWidth,imageHeight,imageChannels);

	hostIdentityOutput = Image_getData(identityOutput);
	hostEdge1Output = Image_getData(edge1Output);
	hostEdge2Output = Image_getData(edge2Output);
	hostEdge3Output = Image_getData(edge3Output);
	hostSharpenOutput = Image_getData(sharpenOutput);
	hostBoxblurOutput = Image_getData(boxblurOutput);
	hostGaussianblur3Output = Image_getData(gaussianblur3Output);

    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceIdentity,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge1,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge2,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge3,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceSharpen,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceBoxblur,sizeof(float)*3*3));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceGaussianblur3,sizeof(float)*3*3));

    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceIdentityOutput,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge1Output,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge2Output,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceEdge3Output,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceSharpenOutput,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceBoxblurOutput,imageDataSize));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceGaussianblur3Output,imageDataSize));

    CUDA_CHECK_RETURN(cudaMemcpy(deviceIdentityOutput,hostIdentityOutput,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceEdge1Output,hostEdge1Output,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceEdge2Output,hostEdge2Output,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceEdge3Output,hostEdge3Output,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceSharpenOutput,hostSharpenOutput,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceBoxblurOutput,hostBoxblurOutput,imageDataSize,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceGaussianblur3Output,hostGaussianblur3Output,imageDataSize,cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemcpy(hostIdentityOutput,deviceIdentityOutput,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostEdge1Output,deviceEdge1Output,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostEdge2Output,deviceEdge2Output,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostEdge3Output,deviceEdge3Output,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostSharpenOutput,deviceSharpenOutput,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostBoxblurOutput,deviceBoxblurOutput,imageDataSize,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(hostGaussianblur3Output,deviceGaussianblur3Output,imageDataSize,cudaMemcpyDeviceToHost));

    Image_setData(identityOutput,hostIdentityOutput);
    Image_setData(edge1Output,hostEdge1Output);
    Image_setData(edge2Output,hostEdge2Output);
    Image_setData(edge3Output,hostEdge3Output);
    Image_setData(sharpenOutput,hostSharpenOutput);
    Image_setData(boxblurOutput,hostBoxblurOutput);
    Image_setData(gaussianblur3Output,hostGaussianblur3Output);

   	PPM_export("processed/identity.ppm",identityOutput);
    PPM_export("processed/edge1.ppm",edge1Output);
    PPM_export("processed/edge2.ppm",edge2Output);
    PPM_export("processed/edge3.ppm",edge3Output);
    PPM_export("processed/sharpen.ppm",sharpenOutput);
    PPM_export("processed/boxblur.ppm",boxblurOutput);
    PPM_export("processed/gaussianblur3.ppm",gaussianblur3Output);

    CUDA_CHECK_RETURN(cudaFree(deviceIdentity));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge1));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge2));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge3));
    CUDA_CHECK_RETURN(cudaFree(deviceSharpen));
    CUDA_CHECK_RETURN(cudaFree(deviceBoxblur));
    CUDA_CHECK_RETURN(cudaFree(deviceGaussianblur3));

    CUDA_CHECK_RETURN(cudaFree(deviceIdentityOutput));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge1Output));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge2Output));
    CUDA_CHECK_RETURN(cudaFree(deviceEdge3Output));
    CUDA_CHECK_RETURN(cudaFree(deviceSharpenOutput));
    CUDA_CHECK_RETURN(cudaFree(deviceBoxblurOutput));
    CUDA_CHECK_RETURN(cudaFree(deviceGaussianblur3Output));
*/



	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

