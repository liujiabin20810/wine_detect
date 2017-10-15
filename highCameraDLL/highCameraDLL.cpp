// highCameraDLL.cpp : ���� DLL Ӧ�ó���ĵ���������
//

#include "stdafx.h"
#include "highCameraDLL.h"
#include "DetectWines.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include "BpNet.h"



using namespace cv;

// ���ǵ���������һ��ʾ��
HIGHCAMERADLL_API int nhighCameraDLL=0;

// ���ǵ���������һ��ʾ����
HIGHCAMERADLL_API int fnhighCameraDLL(void)
{
	return 42;
}

// �����ѵ�����Ĺ��캯����
// �й��ඨ�����Ϣ������� highCameraDLL.h
ChighCameraDLL::ChighCameraDLL()
{
	_detect_wine = new DetectWines();
	return;
}

bool ChighCameraDLL::yanghe_init(string path)
{
	bool flag;
	//����Ƥ���ָ����
	//������ƿ����ز���
	flag = _detect_wine->init(path);

	cout << "init result: " << flag << endl;
	if (!flag)
		return false;

	flag = _detect_wine->readBpNet(path);

	cout << "read BpNet: " << flag << endl;
	return flag;
}

bool ChighCameraDLL::yanghe_setBackground(cv::Mat background0, cv::Mat background1, cv::Mat background2, cv::Mat background3)
{
	if (!background0.data || !background1.data || !background2.data || !background3.data)
		return false;

	_detect_wine->setBackground(background0, background1, background2, background3);

	return true;
}

void ChighCameraDLL::yanghe_detect(cv::Mat img0, cv::Mat img1, cv::Mat img2, cv::Mat img3)
{

	// ��ƿ��
	_detect_wine->countWines(img0);

	// �������ݿ�
	_detect_wine->updateWineData(img1, img2, img3);

	// ƥ�������
	_detect_wine->processWineData();

}

cv::Mat ChighCameraDLL::yanghe_combineImage(int bottle_id)
{
	
	map<int, Mat> combinImage = _detect_wine->getCombinImage();

	return combinImage[bottle_id];
}



std::map<int, bool> ChighCameraDLL::yanghe_detectResult()
{
	//����ƿ�ӱ���Լ���Ӧ��ʶ����
	std::map<int, bool> result = _detect_wine->getResult();
	
	return result;
}