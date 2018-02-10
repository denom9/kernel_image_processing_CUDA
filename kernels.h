//
// Created by samuele on 12/01/18.
//

#ifndef KERNEL_IMAGE_KERNELS_H
#define KERNEL_IMAGE_KERNELS_H


int kernelSize = 3;



float identity[3*3] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };

float edge1[3*3] = {
        1, 0, -1,
        0, 0, 0,
        -1, 0, 1,
};

float edge2[3*3] = {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
};

float edge3[3*3] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1,
};

float sharpen[3*3] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0,
};

/*
float boxblur[3*3] = {
        (float)1/9, (float)1/9, (float)1/9,
        (float)1/9, (float)1/9, (float)1/9,
        (float)1/9, (float)1/9, (float)1/9,
};
*/
float kernel3[3*3] = {
        (float)1/9, (float)1/9, (float)1/9,
        (float)1/9, (float)1/9, (float)1/9,
        (float)1/9, (float)1/9, (float)1/9,
};
float kernel5[5*5] = {
        (float)1/25,(float)1/25,(float)1/25,(float)1/25,(float)1/25,
        (float)1/25,(float)1/25,(float)1/25,(float)1/25,(float)1/25,
        (float)1/25,(float)1/25,(float)1/25,(float)1/25,(float)1/25,
        (float)1/25,(float)1/25,(float)1/25,(float)1/25,(float)1/25,
        (float)1/25,(float)1/25,(float)1/25,(float)1/25,(float)1/25
};
float kernel7[7*7] = {
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,
        (float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49,(float)1/49
};

float gaussianblur3[3*3] = {
        0.0625*1, 0.0625*2, 0.0625*1,
        0.0625*2, 0.0625*4, 0.0625*2,
        0.0625*1, 0.0625*2, 0.0625*1,
};


#endif //KERNEL_IMAGE_KERNELS_H
