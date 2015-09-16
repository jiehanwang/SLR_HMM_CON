// SLR_HMM_ISO.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "ContinuousHMM.h"
#include "Readvideo.h"
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	ofstream outfile;
	ContinuousHMM myMatching;

	//Find videos to be recognized.
	bool fileFindFlag;
	CFileFind fileFind;
	CString normFileName;
	normFileName.Format("E:\\sentence_yf\\P05\\*.oni");
	fileFindFlag = true;
	fileFindFlag = fileFind.FindFile(normFileName);

	while (fileFindFlag)
	{
		//Read the file name
		fileFindFlag = fileFind.FindNextFile();
		CString videoFilePath = fileFind.GetFilePath();
		CString videoFileName = fileFind.GetFileName();
		CString videoFileClass = videoFileName.Mid(4,4);
		int classNo = _ttoi(videoFileClass);
		cout<<"Video: "<<classNo<<endl;
		//Read the video
		Readvideo myReadVideo;
		string s = (LPCTSTR)videoFilePath;
		myReadVideo.readvideo(s);
		int frameSize = myReadVideo.vColorData.size();
		cout<<"Total frameSize "<<frameSize<<endl;

		if (!realOnline)  //use offline strategy
		{
			//Interface of the off-line-continuous SLR
			char* result = new char[100]; //To record the results
			myMatching.patchRun_continuous_offline(myReadVideo.vSkeletonData, 
				myReadVideo.vDepthData, myReadVideo.vColorData,frameSize, result);
			cout<<result<<endl;
			//Output the results to the saved files
			outfile.open("..\\output\\result.txt",ios::out | ios::app);
			outfile<<result<<endl;
			outfile.close();
			delete[] result;
		}
		else
		{
			//Interface of the online-continuous SLR
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

			vector<int> result;
			result = myMatching.addLanguageModel(rankIndex,rankLength);

			//Output the results to the saved files
			outfile.open("..\\output\\result.txt",ios::out | ios::app);
			for (int i=0; i<rankLength; i++)
			{
				outfile<<rankIndex[i]<<" ";
			}
			outfile<<'\t';
			for (int i=0; i<result.size(); i++)
			{
				outfile<<result[i]<<" ";
			}
			outfile<<endl;
			outfile.close();

			//Release the resource
			myMatching.patchRun_release();
		}
	}

	cout<<"Done!"<<endl;
	getchar();
	return 0;
}

