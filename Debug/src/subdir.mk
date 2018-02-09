################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/Kernel\ Image\ CUDA.cu 

CPP_SRCS += \
../src/Image.cpp \
../src/PPM.cpp 

OBJS += \
./src/Image.o \
./src/Kernel\ Image\ CUDA.o \
./src/PPM.o 

CU_DEPS += \
./src/Kernel\ Image\ CUDA.d 

CPP_DEPS += \
./src/Image.d \
./src/PPM.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/Kernel\ Image\ CUDA.o: ../src/Kernel\ Image\ CUDA.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


