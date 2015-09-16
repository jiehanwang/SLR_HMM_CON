#pragma once
#include "Hmm.h"
#include "Recognition.h"
#include <opencv2\opencv.hpp>
#include "S_FeaExtraction.h"
#include "globalDefine.h"
#include <fstream>
using namespace std;
using namespace cv;

class ContinuousHMM
{
public:
	ContinuousHMM(void);
	~ContinuousHMM(void);

public:
	CHMM* m_pDhmm_test;
	CRecognition *m_pRecog;
	S_CFeaExtraction myFeaExtraction;
	CHandSegment_HMM handSegmentVideo;

	CString modelPath;

	CvPoint headPoint, lPoint2, rPoint2;
	vector<Posture>         vPosture;         //Record hand postures from each frames. 
	vector<SLR_ST_Skeleton> vSkeleton;        //Record skeletons from each frames. 

	vector<int> frameSelect;
	bool frameSelected;
	int usefulFrameSize;
	int heightLimit;
	char resWord[500];

	CString groundtruthFile;
	vector<vector<int>> groundTruthAll;

	void loadModel(CString path);
	void run(double **feature, int frameNum, char* result);
	void frameSelect_inMatch(int heightLimit, int leftY, int rightY);
	void readIndata(SLR_ST_Skeleton skeletonCurrent, Mat depthCurrent, IplImage* frameCurrent,int framID);
	void recognize(char* result);
	void patchRun(vector<SLR_ST_Skeleton> skeleton, vector<Mat> depth, vector<IplImage*> color, char* resWord);
	void ReadGallery(CString path);
	static bool comp2(scoreAndIndex dis_1, scoreAndIndex dis_2);
	void patchRun_continuous(SLR_ST_Skeleton vSkeletonData, Mat vDepthData, IplImage* vColorData, int framID, int rankIndex[], int &rankLength);
	void patchRun_release(void);
	void patchRun_initial(void);
	void patchRun_continuous_offline(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, 
		vector<IplImage*> vColorData,int frameNum, char* result);
	Mat retrieveGrayDepth(Mat depthMat);
	int editDistance(vector<int> recognizeList, vector<int> groundtruthList);
	void readstr(FILE *f,char *string);
	void loadGroundTruthAll(vector<vector<int>> &groundTruthAll);
	vector<int> addLanguageModel(int rankIndex[], int rankLength);
};

