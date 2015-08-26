// SLR_HMM_ISO.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ContinuousHMM.h"
#include "Readvideo.h"
using namespace std;


int _tmain(int argc, _TCHAR* argv[])
{
	ContinuousHMM myMatching;

	//Read videos.
	bool fileFindFlag;
	CFileFind fileFind;
	CString normFileName;
	normFileName.Format("E:\\continuousDemoSign4test\\p08_05\\*.oni");
	fileFindFlag = true;
	fileFindFlag = fileFind.FindFile(normFileName);


	while (fileFindFlag)
	{
		fileFindFlag = fileFind.FindNextFile();
		CString videoFilePath = fileFind.GetFilePath();
		CString videoFileName = fileFind.GetFileName();
		CString videoFileClass = videoFileName.Mid(4,4);
		int classNo = _ttoi(videoFileClass);
		cout<<"Video: "<<classNo<<endl;

		Readvideo myReadVideo;
		string s = (LPCTSTR)videoFilePath;
		myReadVideo.readvideo(s);
		int frameSize = myReadVideo.vColorData.size();
		cout<<"Total frameSize "<<frameSize<<endl;

		//Interface of off-line-continuous SLR
// 		char* result = new char[100];
// 		myMatching.patchRun_continuous_offline(myReadVideo.vSkeletonData, myReadVideo.vDepthData, 
// 			myReadVideo.vColorData,frameSize, result);
// 		cout<<result<<endl;
// 		delete[] result;

		//Interface of onl-ine-continuous SLR
		int rankIndex[100];
		int rankLength = 0;
		myMatching.patchRun_initial();
		for (int i=0; i<frameSize; i++)
		{
			myMatching.patchRun_continuous(myReadVideo.vSkeletonData[i], 
				myReadVideo.vDepthData[i],
				myReadVideo.vColorData[i],i, rankIndex, rankLength);
			for (int j=0; j<rankLength; j++)
			{
				cout<<rankIndex[j]<<" ";
			}
 			cout<<endl;
		}
		myMatching.patchRun_release();
	}

	cout<<"Done!"<<endl;
	getchar();
	return 0;
}

