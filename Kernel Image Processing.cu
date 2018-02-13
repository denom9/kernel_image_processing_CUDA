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
#define MAX_KERNEL 7


#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "PPM.h"
#include "kernels.h"




static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)



__constant__ float deviceKernel[7*7];



__global__ void kernelConvolution(float* img, float* output ,const int imageWidth, const int imageHeight, const int imageChannels, const int KERNEL_SIZE){

	const int INPUT_TILE_WIDTH = TILE_WIDTH + MAX_KERNEL - 1;
	__shared__ float imgTile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH]; //condivisa a livello di blocco

	//printf("%i-%i\n",blockDim.x,blockDim.y);
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int border = KERNEL_SIZE / 2;

	int kIndex;
	float sum = 0;

	//in quale INPUT_TILE sono (e quindi anche in quale TILE)
	int h = blockIdx.x;
	int w = blockIdx.y;

	//dove sono all'interno del mio INPUT_TILE
	int tileRow = threadIdx.y;
	int tileCol = threadIdx.x;
	//int tileChannel = threadIdx.z;

	//in quale rispettivo pixel di immagine devo lavorare
	int wOff = 	(row <= border)? 0 : tileRow - border;
	int hOff = 	(col <= border)? 0 : tileCol - border;

	int imgRow = ((w * TILE_WIDTH + wOff) < imageHeight)? w * TILE_WIDTH + wOff : imageHeight - 1;
	int imgCol = ((h * TILE_WIDTH + hOff) < imageWidth)? h * TILE_WIDTH + hOff : imageWidth - 1;


	for(int c = 0; c < imageChannels; c++){
		imgTile[tileRow][tileCol] = img[(imgRow * imageWidth + imgCol)*imageChannels + c];


		__syncthreads();


		if((tileRow >= border)&&(tileRow < TILE_WIDTH + border)&&(tileCol >= border)&&(tileCol < TILE_WIDTH + border)&&((w * TILE_WIDTH + wOff)<imageHeight)&&((h * TILE_WIDTH + hOff)<imageWidth)){
			for(int i = 0; i < KERNEL_SIZE; i++){
				for(int j = 0; j < KERNEL_SIZE; j++){
					kIndex = (KERNEL_SIZE-1 - i) * KERNEL_SIZE + (KERNEL_SIZE-1 -j);
					sum += imgTile[tileRow + i - border][tileCol + j - border] * deviceKernel[kIndex];
				}
			}
			output[(imgRow * imageWidth + imgCol) * imageChannels + c] = sum;//imgTile[tileRow-1][tileCol-1];
			sum = 0;
		}

		__syncthreads();

	}

}


int main(int argc,  char** argv){

    Image_t* inputImage = PPM_import(argv[1]);

    const int KERNEL_SIZE = (*argv[2] != '3' && *argv[2] != '5' && *argv[2] != '7')? 3 : (int)*argv[2] - '0';
    const int imageWidth = Image_getWidth(inputImage);
    const int imageHeight = Image_getHeight(inputImage);
    const int imageChannels = Image_getChannels(inputImage);
    const int imageDataSize = sizeof(float)*imageWidth*imageHeight*imageChannels;

    Image_t* outputImage = Image_new(imageWidth,imageHeight,imageChannels);

    float *hostInput = Image_getData(inputImage);;
    float *hostOutput = Image_getData(outputImage);
    float *deviceInput;
    float *deviceOutput;

    float* hostKernel = (*argv[2] == '5')? kernel5 : (*argv[2] == '7')? kernel7 : kernel3;



    //alloco la memoria (global) per contenere i dati dell'immagine e ci copio i dati
    CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceInput,imageDataSize));
    CUDA_CHECK_RETURN(cudaMemcpy(deviceInput,hostInput,imageDataSize,cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceOutput,imageDataSize)); //alloco le varie locazioni di memoria device per le immagini e salvo il puntatore nell'array
	CUDA_CHECK_RETURN(cudaMemcpy(deviceOutput,hostOutput,imageDataSize,cudaMemcpyHostToDevice)); //trasferisco i dati immagini nelle locazioni appena create

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceKernel, hostKernel,sizeof(float)*KERNEL_SIZE*KERNEL_SIZE)); //trasferisco i dati dei kernel in constant memory


    dim3 gridDim(ceil((float)imageWidth / TILE_WIDTH),ceil((float)imageHeight / TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH + KERNEL_SIZE - 1,TILE_WIDTH + KERNEL_SIZE - 1);
    //printf("%i,%i\n%i,%i\n",gridDim.x,gridDim.y,blockDim.x,blockDim.y);

    auto start = std::chrono::system_clock::now();

    kernelConvolution<<<gridDim,blockDim>>>(deviceInput,deviceOutput,imageWidth,imageHeight,imageChannels,KERNEL_SIZE);

    cudaDeviceSynchronize();

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed = end-start;
    std::cout << elapsed.count();

    //copio indietro i risultati e imposto i dati immagine
	CUDA_CHECK_RETURN(cudaMemcpy(hostOutput,deviceOutput,imageDataSize,cudaMemcpyDeviceToHost));

	Image_setData(outputImage,hostOutput);

    PPM_export("/home/samuele/Documenti/Università/Parallel Computing/Progetto finale/Kernel image/outputs/output_CUDA.ppm",outputImage);

    //libero la memoria
    CUDA_CHECK_RETURN(cudaFree(deviceInput));
    CUDA_CHECK_RETURN(cudaFree(deviceOutput));


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

