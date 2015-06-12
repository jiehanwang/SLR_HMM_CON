// SLR_HMM_ISO.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "IsolateHMM.h"
#include "Readvideo.h"
using namespace std;


int _tmain(int argc, _TCHAR* argv[])
{

	IsolateHMM myMatching;

	//Read videos.
	bool fileFindFlag;
	CFileFind fileFind;
	CString normFileName;
	normFileName.Format("E:\\continuousDemoSign4test\\p08_01\\*.oni");
	fileFindFlag = true;
	fileFindFlag = fileFind.FindFile(normFileName);


	while (fileFindFlag)
	{
		fileFindFlag = fileFind.FindNextFile();
		CString videoFilePath = fileFind.GetFilePath();
		CString videoFileName = fileFind.GetFileName();
		CString videoFileClass = videoFileName.Mid(4,4);
		int classNo = _ttoi(videoFileClass);
		cout<<classNo<<endl;

		Readvideo myReadVideo;
		string  s   =   (LPCTSTR)videoFilePath;
		myReadVideo.readvideo(s);
		int frameSize = myReadVideo.vColorData.size();
		cout<<"Total frameSize "<<frameSize<<endl;

			//Recognize
// 		myMatching.patchRun(myReadVideo.vSkeletonData, 
// 			myReadVideo.vDepthData,
// 			myReadVideo.vColorData,
// 			resWord);
		//vector<int> rankIndex;
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
			//cout<<resWord<<endl;
		}

		
		//Show the result
		//cout<<"-----"<<resWord<<endl;
		myMatching.patchRun_release();
	}

	cout<<"Done!"<<endl;
	getchar();
	return 0;
}

