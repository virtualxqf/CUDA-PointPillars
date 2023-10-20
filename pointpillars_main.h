#ifndef POINT_PILLARS_MAIN_  
#define POINT_PILLARS_MAIN_  
//#include "pointpillars_main.h"
#include "cuda_runtime.h"

#include "./params.h"
#include "./pointpillar.h"

class pointpillars_main
{
    public:
        pointpillars_main(std::string model_file, std::string save_path);
        void doinit();
        void doinfer(char* data_input);	
        ~pointpillars_main();
    private:
        void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name);
        int loadData(const char *file, void **data, unsigned int *length);
        void Getinfo(void);
	std::string mModelFile;
	std::string mDataInput;
	PointPillar *mPointPillar;
	std::string mSaveDir;
	cudaEvent_t mStart, mStop;
	cudaStream_t mStream = NULL;
	std::vector<Bndbox> mNmsPred;
};
#endif

