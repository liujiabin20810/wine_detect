// highCameraDLL.cpp : 定义 DLL 应用程序的导出函数。
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

// 这是导出变量的一个示例
HIGHCAMERADLL_API int nhighCameraDLL=0;

// 这是导出函数的一个示例。
HIGHCAMERADLL_API int fnhighCameraDLL(void)
{
	return 42;
}

// 这是已导出类的构造函数。
// 有关类定义的信息，请参阅 highCameraDLL.h
ChighCameraDLL::ChighCameraDLL()
{
	_detect_wine = new DetectWines();
	return;
}

bool ChighCameraDLL::yanghe_init(string path)
{
	bool flag;
	//设置皮带分割参数
	//设置数瓶子相关参数
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

	// 数瓶子
	_detect_wine->countWines(img0);

	// 更新数据库
	_detect_wine->updateWineData(img1, img2, img3);

	// 匹配与计算
	_detect_wine->processWineData();

}

cv::Mat ChighCameraDLL::yanghe_combineImage(int bottle_id)
{
	
	map<int, Mat> combinImage = _detect_wine->getCombinImage();

	return combinImage[bottle_id];
}



std::map<int, bool> ChighCameraDLL::yanghe_detectResult()
{
	//返回瓶子编号以及对应的识别结果
	std::map<int, bool> result = _detect_wine->getResult();
	
	return result;
}