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
//#include <chrono>
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
		int kIndex;
		int a,b;
		int border = KERNEL_SIZE / 2;
		int rowImgBorder,colImgBorder,rowTileBorder,colTileBorder,rowInputBorder,colInputBorder;


		int tileRow = row % TILE_WIDTH;
		int tileCol = col % TILE_WIDTH;

		float element;
		float sum[RESULTS_NUM];
		for(int h = 0; h < RESULTS_NUM; h++)
			sum[h] = 0;

		/* offset per gestire i bordi immagine, bordi tile e bordi dell'input */
		rowInputBorder = (tileRow == 0)? border : 0;
		colInputBorder = (tileCol == 0)? border : 0;

		rowTileBorder = (tileRow == TILE_WIDTH - 1)? border : 0;
		colTileBorder = (tileCol == TILE_WIDTH - 1)? border : 0;

		rowImgBorder = (row == imageHeight - 1)? border : 0;
		colImgBorder = (col == imageWidth - 1)? border : 0;



		//printf("row: %i, col: %i,tileCell: %i, tileRow: %i, tileCol: %i\n",row,col,tileCell,tileRow,tileCol);
		for(int c = 0; c < imageChannels; c++){

			/* caricamento del'input tile in shared memory */
			element = img[(row * imageWidth + col) * imageChannels + c];


			/*
			 * devo gestire tre casi principali:
			 *	- bordo immagine: per i pixel necessari per la convoluzione che "vanno fuori" dall'immagine vengono usati i corrispondenti pixel di bordo
			 *	- bordo del tile non di bordo immagine: per i pixel necessari alla convoluzione bisogna prendere pixel che apparterrebbero ad altri tile adiacenti
			 *	- cella non di bordo: ho tutti i pixel necessari alla convoluzione
			 */

			if((row == 0)||(row == imageHeight-1)||(col == 0)||(col == imageWidth-1)){ //bordi immagine

				if(((row == 0)||(row == imageHeight-1))&&((col == 0)||(col == imageWidth-1))){ //angoli immagine
					for(a = 0; a < border+1; a++){
						for(b = 0; b < border+1; b++)
							imgTile[tileRow+a+rowImgBorder][tileCol+b+colImgBorder]=element;
					}
				}

				else if(((row == 0)||(row == imageHeight-1))&&((tileCol == 0)||(tileCol == TILE_WIDTH-1))){ //angoli intermedi superiori e inferiori sinistri
					for(a = 0; a < border+1; a++){
						for(b = 0; b < border+1; b++){
							if(((a < border)&&(row==0))||((a > 0)&&(row==imageHeight-1)))
								imgTile[tileRow+a+rowImgBorder][tileCol+b+colTileBorder]=element;
							else
								imgTile[tileRow+a+rowImgBorder][tileCol+b+colTileBorder]=img[((row-rowInputBorder+a) * imageWidth + (col-colInputBorder+b)) * imageChannels + c];
						}
					}
				}

				else if(((col == 0)||(col == imageWidth-1))&&((tileRow == 0)||(tileRow == TILE_WIDTH-1))){ //angoli intermedi destri e sinistri
					for(a = 0; a < border+1; a++){
						for(b = 0; b < border+1; b++){
							if(((b < border)&&(col==0))||((b > 0)&&(col==imageWidth-1)))
								imgTile[tileRow+a+rowTileBorder][tileCol+b+colImgBorder]=element;
							else
								imgTile[tileRow+a+rowTileBorder][tileCol+b+colImgBorder]=img[((row-rowInputBorder+a) * imageWidth + (col-colInputBorder+b)) * imageChannels + c];
						}
					}
				}

				else{ //lati immagine
					if((row == 0)||(row == imageHeight-1)){ //lati superiore e inferiore
						for(a = 0; a < border+1; a++)
							imgTile[tileRow+a+rowImgBorder][tileCol+border]=element;
					}
					else if((col == 0)||(col == imageWidth-1)){ //lati sinistro e destro
						for(a = 0; a < border+1; a++)
							imgTile[tileRow+border][tileCol+a+colImgBorder]=element;
					}
				}
			}


			else if((tileRow == 0)||(tileRow == TILE_WIDTH-1)||(tileCol == 0)||(tileCol == TILE_WIDTH-1)){ //bordi tile interni

				if(((tileRow == 0)||(tileRow == TILE_WIDTH-1))&&((tileCol == 0)||(tileCol == TILE_WIDTH-1))){ //angoli tile interni
					for(a = 0; a < border+1; a++){
						for(b = 0; b < border+1; b++)
							imgTile[tileRow+a+rowTileBorder][tileCol+b+colTileBorder]=img[((row-rowInputBorder+a) * imageWidth + (col-colInputBorder+b)) * imageChannels + c];
					}

				}
				else{ //bordi tile interni
					if((tileRow == 0)||(tileRow == TILE_WIDTH-1)){
						for(a = 0; a < border+1; a++)
							imgTile[tileRow+a+rowTileBorder][tileCol+border]=img[((row-rowInputBorder+a) * imageWidth + col) * imageChannels + c];
					}
					else if((tileCol == 0)||(tileCol == TILE_WIDTH-1)){
						for(a = 0; a < border+1; a++)
							imgTile[tileRow+border][tileCol+a+colTileBorder]=img[(row * imageWidth + (col-colInputBorder+a)) * imageChannels + c];
					}
				}

			}


			else //cella non di bordo
				imgTile[tileRow+border][tileCol+border] = element;






			__syncthreads();


			/*calcolo dell'output*/
			for(int i = 0; i < KERNEL_SIZE; i++){
				//yIndex = ((row - 1 + i) < 0) ? 0 : ((row - 1 + i) >= imageHeight) ? imageHeight - 1 : row - 1 + i;
				for(int j = 0; j < KERNEL_SIZE; j++){
					//xIndex = ((col - 1 + j) < 0) ? 0 : ((col - 1 + j) >= imageWidth) ? imageWidth - 1 : col - 1 + j;
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
    //printf("%i,%i\n",(imageWidth / TILE_WIDTH) + 1,(imageHeight / TILE_WIDTH) + 1);

    //auto start = std::chrono::system_clock::now();

    kernelConvolution<<<gridDim,blockDim>>>(deviceInput,deviceOutput,imageWidth,imageHeight,imageChannels);

    cudaDeviceSynchronize();

    //copio indietro i risultati e imposto i dati immagine
    for(i = 0; i < RESULTS_NUM; i++){
    	CUDA_CHECK_RETURN(cudaMemcpy(hostOutput[i],devicePointers[i],imageDataSize,cudaMemcpyDeviceToHost));
    	Image_setData(outputImages[i],hostOutput[i]);
    }


    //esporto le immagini
	//PPM_export("processed/identity.ppm",outputImages[0]);
    //PPM_export("processed/edge1.ppm",outputImages[1]);
   // PPM_export("processed/edge2.ppm",outputImages[2]);
    //PPM_export("processed/edge3.ppm",outputImages[3]);
    //PPM_export("processed/sharpen.ppm",outputImages[4]);
    PPM_export("processed/boxblur.ppm",outputImages[5]);
    //PPM_export("processed/gaussianblur3.ppm",outputImages[6]);

    //libero la memoria
    CUDA_CHECK_RETURN(cudaFree(deviceInput));
    for(int i = 0; i < RESULTS_NUM; i++)
    	CUDA_CHECK_RETURN(cudaFree(devicePointers[i]));
    CUDA_CHECK_RETURN(cudaFree(deviceOutput));





    //auto end = std::chrono::system_clock::now();
    //std::chrono::duration<double> elapsed = end-start;

    //printf("Elapsed time:%d\n",elapsed);





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

