/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <fstream>

#include "pointpillars_main.h"
//#include "cuda_runtime.h"

//#include "./params.h"
//#include "./pointpillar.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

pointpillars_main::pointpillars_main(std::string model_file, std::string save_path)
{
	std::cout << "start to construct pointpillars_main "<< std::endl;
	mModelFile = model_file;
	mSaveDir = save_path;
	std::cout << "end construct pointpillars_main "<< std::endl;
}

void pointpillars_main::Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int pointpillars_main::loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

void pointpillars_main::SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << " ";
          ofs << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

void pointpillars_main::doinit()
{
    if (mModelFile.length() == 0) {
		return;
    }
    Getinfo();
    checkCudaErrors(cudaEventCreate(&mStart));
    checkCudaErrors(cudaEventCreate(&mStop));
    checkCudaErrors(cudaStreamCreate(&mStream));
    mPointPillar = new PointPillar(mModelFile, mStream);
//mPointPillar = &pointpillar;
}

void pointpillars_main::doinfer(char* data_input)
{

	float elapsedTime = 0.0f;
	mNmsPred.clear();
        //Params params_;

	mNmsPred.reserve(100);


	std::string dataFile = data_input;

	std::stringstream ss;


	int n_zero = 6;
	std::string _str = ss.str();
	std::string index_str = std::string(n_zero - _str.length(), '0') + _str;
	dataFile += index_str;
	dataFile +=".bin";

	std::cout << "<<<<<<<<<<<" <<std::endl;
	std::cout << "load file: "<< dataFile <<std::endl;

	//load points cloud
	unsigned int length = 0;
	void *data = NULL;
	std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
	loadData(dataFile.data(), &data, &length);
	buffer.reset((char *)data);

	float* points = (float*)buffer.get();
	size_t points_size = length/sizeof(float)/4;

	std::cout << "find points num: "<< points_size <<std::endl;

	float *points_data = nullptr;
	unsigned int points_data_size = points_size * 4 * sizeof(float);
	checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
	checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
	checkCudaErrors(cudaDeviceSynchronize());

	cudaEventRecord(mStart, mStream);

	mPointPillar->doinfer(points_data, points_size, mNmsPred);
	cudaEventRecord(mStop, mStream);
	cudaEventSynchronize(mStop);
	cudaEventElapsedTime(&elapsedTime, mStart, mStop);
	std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

	checkCudaErrors(cudaFree(points_data));

	std::cout<<"Bndbox objs: "<< mNmsPred.size()<<std::endl;
	std::string save_file_name = mSaveDir + index_str + ".txt";
	SaveBoxPred(mNmsPred, save_file_name);


	std::cout << ">>>>>>>>>>>" <<std::endl;

}

pointpillars_main::~pointpillars_main() {
  checkCudaErrors(cudaEventDestroy(mStart));
  checkCudaErrors(cudaEventDestroy(mStop));
  checkCudaErrors(cudaStreamDestroy(mStream));
  delete mPointPillar;  
}
