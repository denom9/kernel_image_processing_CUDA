//
// Created by samuele on 12/01/18.
//

#ifndef KERNEL_IMAGE_KERNELS_H
#define KERNEL_IMAGE_KERNELS_H


float identity[3*3] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
    };
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




#endif //KERNEL_IMAGE_KERNELS_H
