#include "StdAfx.h"
#include "ContinuousHMM.h"


ContinuousHMM::ContinuousHMM(void)
{
	m_pDhmm_test = new CHMM;

	if (skinHandseg)
	{
		ReadGallery("..\\model\\HmmData_signFromSentence_370sign_withoutP0805.dat");
	}
	else
	{
		ReadGallery("..\\model\\HmmData_signFromSentence_370sign_withoutP0805.dat");
	}
	
	m_pRecog = new CRecognition;
	m_pRecog->GetHmmModel(m_pDhmm_test);

	handSegmentVideo.init();
	usefulFrameSize = 0;
}


ContinuousHMM::~ContinuousHMM(void)
{
}


void ContinuousHMM::loadModel(CString path)
{
	m_pDhmm_test->Init(path);
}


void ContinuousHMM::run(double **feature, int frameNum, char* result)
{
	//m_pRecog->GetHmmModel(m_pDhmm_test);
	//double* pCadidateProb = new double [m_pRecog->m_pDhmm->m_nTotalHmmWord];
	//int nWordNum, nRecogNum;
	//m_pRecog->TestOneWordFromMemory(feature,frameNum,pCadidateProb,nWordNum,nRecogNum);

	//char resWord[500];
	m_pRecog->ContinueTestOneSen_Whj(feature, frameNum, result);


	//cout<<m_pRecog->m_pDhmm->m_pHmmWordIndex[nRecogNum]->Word<<endl;

// 	int sentenseNum = m_pRecog->m_pDhmm->m_nTotalHmmWord;
// 	scoreAndIndex* SI;
// 	SI = new scoreAndIndex[sentenseNum];
// 	for (int i=0; i<sentenseNum; i++)
// 	{
// 		SI[i].score = *(pCadidateProb + i);
// 		string temp(m_pRecog->m_pDhmm->m_pHmmWordIndex[i]->Word);
// 		string temp2(temp,1,4);
// 		SI[i].index = atoi(temp2.c_str());
// 		//cout<<SI[i].index<<" "<<SI[i].score<<endl;
// 	}
// 	sort(SI,SI+sentenseNum,comp2);
// 
// 	for (int i=0; i<5; i++)
// 	{
// 		rank[i] = SI[i].index;
// 		score[i] = SI[i].score;
// 		//cout<<"Top "<<i<<": "<<rank[i]<<" "<<score[i]<<endl;
// 	}
}

bool ContinuousHMM::comp2(scoreAndIndex dis_1, scoreAndIndex dis_2)
{
	return dis_1.score > dis_2.score;
}

void ContinuousHMM::frameSelect_inMatch(int heightLimit, int leftY, int rightY)
{
	int heightThisLimit = min(leftY,rightY);
	if (heightThisLimit < heightLimit)
	{
		frameSelect.push_back(1);
		usefulFrameSize++;
	}
	else
	{
		frameSelect.push_back(0);
	}
}


void ContinuousHMM::readIndata(SLR_ST_Skeleton skeletonCurrent, Mat depthCurrent, IplImage* frameCurrent,int framID)
{
	if (framID == 0)
	{
		headPoint.x = skeletonCurrent._2dPoint[3].x;
		headPoint.y = skeletonCurrent._2dPoint[3].y;
		bool bHeadFound = handSegmentVideo.headDetectionVIPLSDK(
			frameCurrent,
			depthCurrent,
			headPoint);

		if(bHeadFound)
		{
			handSegmentVideo.colorClusterCv(handSegmentVideo.m_pHeadImage,3);
			handSegmentVideo.getFaceNeckRegion(frameCurrent,depthCurrent);
			handSegmentVideo.copyDepthMat(depthCurrent.clone());
		}
		else
		{
			cout<<"Face is not detected!!!"<<endl;
		}
	}

	//Frame selection is closed in the on-line continuous SLR. 
	//Therefore, the following code is commented. 
	//if (frameSelect[framID] == 1)   
	{
		vSkeleton.push_back(skeletonCurrent);

		Posture posture;
		lPoint2.x = skeletonCurrent._2dPoint[7].x;
		lPoint2.y = skeletonCurrent._2dPoint[7].y;
		rPoint2.x = skeletonCurrent._2dPoint[11].x;
		rPoint2.y = skeletonCurrent._2dPoint[11].y;

		CvRect leftHand;
		CvRect rightHand;

		if (featureFromRGB)
		{
			//Feature extracted from RGB
			handSegmentVideo.kickHandsAll(frameCurrent,depthCurrent
				,lPoint2,rPoint2,posture,leftHand,rightHand);
		}
		else
		{
			//Feature extracted from Depth
			Mat tempMat = retrieveGrayDepth(depthCurrent);
			IplImage depthGray = IplImage(tempMat);
			handSegmentVideo.kickHandsAll(&depthGray,depthCurrent
				,lPoint2,rPoint2,posture,leftHand,rightHand);
		}

		if (saveTempImages)
		{
			CString outputName;
			outputName.Format("..\\output\\images\\%03d_left.jpg", framID);
			cvSaveImage(outputName, posture.leftHandImg);
			outputName.Format("..\\output\\images\\%03d_right.jpg", framID);
			cvSaveImage(outputName, posture.rightHandImg);
		}
		
		vPosture.push_back(posture);
	}
}

Mat ContinuousHMM::retrieveGrayDepth(Mat depthMat)
{
	double maxDisp = -1.f;
	float S = 1.f;
	float V = 1.f;
	Mat disp;
	disp.create( Size(640,480), CV_32FC1);
	disp = cv::Scalar::all(0);
	for( int y = 0; y < disp.rows; y++ )
	{
		for( int x = 0; x < disp.cols; x++ )
		{
			unsigned short curDepth = depthMat.at<unsigned short>(y,x);
			if( curDepth != 0 )
				disp.at<float>(y,x) = (75.0 * 757) / curDepth;
		}
	}
	Mat gray;
	disp.convertTo( gray, CV_8UC1 );
	if( maxDisp <= 0 )
	{
		maxDisp = 0;
		minMaxLoc( gray, 0, &maxDisp );
	}
	Mat _depthColorImage;
	_depthColorImage.create( gray.size(), CV_8UC3 );
	_depthColorImage = Scalar::all(0);
	for( int y = 0; y < gray.rows; y++ )
	{
		for( int x = 0; x < gray.cols; x++ )
		{
			uchar d = gray.at<uchar>(y,x);
			if (d == 0)
				continue;

			unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

			_depthColorImage.at<Point3_<uchar> >(y,x) = Point3_<uchar>(H, H, H);     
		}
	}
	return _depthColorImage;
}


void ContinuousHMM::recognize(char* result)
{
	myFeaExtraction.postureFeature(vPosture,handSegmentVideo);
	myFeaExtraction.SPFeature(vSkeleton);
	myFeaExtraction.PostureSP();


	//CString modelPath = "..\\model\\HmmData_50.dat";
	//loadModel(modelPath);
	run(myFeaExtraction.feature, myFeaExtraction.frameNum, result);


	//Release resource.
	myFeaExtraction.release();
	for (int i=0; i<vPosture.size(); i++)
	{
		cvReleaseImage(&vPosture[i].leftHandImg);
		cvReleaseImage(&vPosture[i].rightHandImg);
	}
	vPosture.clear();
	vSkeleton.clear();
	frameSelect.clear();
}


void ContinuousHMM::patchRun(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, vector<IplImage*> vColorData, 
	char* resWord)
{
	SLR_ST_Skeleton skeletonCurrent;    //The 3 current data.
	Mat             depthCurrent;
	IplImage        *frameCurrent;

	//Decide the frames to be used or not. frameSelect is the mask. 
	int frameSize = vColorData.size();

	//The height constraints is in the readIndata. Hanjie Wang closed the constraint. 
// 	int heightLimit = min(vSkeletonData[0]._2dPoint[7].y,
// 		vSkeletonData[0]._2dPoint[11].y) - 20;
// 	for (int i=0; i<frameSize; i++)
// 	{
// 		frameSelect_inMatch(heightLimit, vSkeletonData[i]._2dPoint[7].y,
// 			vSkeletonData[i]._2dPoint[11].y);
// 	}

	//Read in data and extract the hand postures in each available frame. 
	for (int i=0; i<frameSize; i++)
	{
		skeletonCurrent = vSkeletonData[i];
		depthCurrent    = vDepthData[i];
		frameCurrent    = vColorData[i];
		int framID = i;
		readIndata(skeletonCurrent, depthCurrent, frameCurrent, framID);
	}

	//Extract the SP and hog feature, and recognize
	
	recognize(resWord);

}

void ContinuousHMM::patchRun_continuous_offline(vector<SLR_ST_Skeleton> vSkeletonData, vector<Mat> vDepthData, 
	vector<IplImage*> vColorData,int frameNum, char* result)
{
	//Read in data of one frame to this class
	for (int i=0; i<frameNum; i++)
	{
		readIndata(vSkeletonData[i], vDepthData[i], vColorData[i], i);
	}
	
	//Compute the feature. The size of the frame is only 1.
	myFeaExtraction.postureFeature(vPosture,handSegmentVideo);
	myFeaExtraction.SPFeature(vSkeleton);
	myFeaExtraction.PostureSP();

	m_pRecog->ContinueTestOneSen_Whj(myFeaExtraction.feature, frameNum, result);

	//release
	vPosture.clear();
	vSkeleton.clear();
	frameSelect.clear();

}
void ContinuousHMM::patchRun_continuous(SLR_ST_Skeleton vSkeletonData, Mat vDepthData, 
	IplImage* vColorData,int framID, int rankIndex[], int &rankLength)
{
	//Read in data of one frame to this class
	readIndata(vSkeletonData, vDepthData, vColorData, framID);
	//The vPosture and vSkeleton have been computed 

	//Compute the feature. The size of the frame is only 1.
	myFeaExtraction.postureFeature(vPosture,handSegmentVideo);
	myFeaExtraction.SPFeature(vSkeleton);
	myFeaExtraction.PostureSP();

	//Recognize the frame
	m_pRecog->continuous_loop(myFeaExtraction.feature[0], framID, resWord);

	CString str(resWord);
	int strLength = str.GetLength();
	int nWord = strLength/6;
	for (int i=0; i<nWord; i++)
	{
		CString strTemp = str.Mid(i*6+1, 4);
		int iWord = _ttoi(strTemp);
		rankIndex[i] = iWord;
	}
	rankLength = nWord;

	//Release the resource of the current frame
	myFeaExtraction.release();
	for (int i=0; i<vPosture.size(); i++)
	{
		cvReleaseImage(&vPosture[i].leftHandImg);
		cvReleaseImage(&vPosture[i].rightHandImg);
	}
	vPosture.clear();
	vSkeleton.clear();
	frameSelect.clear();

}

void ContinuousHMM::ReadGallery(CString path)
{
	modelPath = path;
	loadModel(modelPath);
}


void ContinuousHMM::patchRun_release(void)
{
	m_pRecog->continuous_release();
}


void ContinuousHMM::patchRun_initial(void)
{
	m_pRecog->continuous_initial();
	//resWord[0] = 0;
	memset(resWord,0,500);
}
