// highCameraDemo.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "highCameraDLL.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <stdarg.h>

using namespace cv;
using namespace std;

string Int_to_String(int n)
{
	ostringstream stream;
	stream << n;  //n为int类型
	return stream.str();
}

std::string & to_string_format(std::string & _str, const char * _Format, ...) {
	std::string tmp;

	va_list marker = NULL;
	va_start(marker, _Format);

	size_t num_of_chars = _vscprintf(_Format, marker);

	if (num_of_chars > tmp.capacity()) {
		tmp.resize(num_of_chars + 1);
	}

	vsprintf_s((char *)tmp.data(), tmp.capacity(), _Format, marker);

	va_end(marker);

	_str = tmp.c_str();
	return _str;
}


int _tmain(int argc, _TCHAR* argv[])
{

	ChighCameraDLL _detect;

	string path = "E:\\Wopu\\YANGHE\\WineData\\1014_new\\01\\";

	if (!_detect.yanghe_init(""))
	{
		cout << "init : out" << endl;
		return -1;
	}

	Mat background0 = cv::imread("cam0_0000.jpg", 0);
	Mat background1 = cv::imread("cam1_0000.jpg", 0);
	Mat background2 = cv::imread("cam2_0000.jpg", 0);
	Mat background3 = cv::imread("cam3_0000.jpg", 0);

	if (!background0.empty())
	{
		background0(Rect(0, 0, 2592, 570)) = 0;
	}

	if (!_detect.yanghe_setBackground(background0, background1, background2, background3))
	{
		cout << "setBackground : out" << endl;
		return -1;
	}

	for (int fileNum = 72; fileNum <= 220; fileNum++)
	{
		
		string _id;
		to_string_format(_id, "%04d.jpg", fileNum);
		string filename0 = path + "cam0_" + _id;
		string filename1 = path + "cam1_" + _id;
		string filename2 = path + "cam2_" + _id;
		string filename3 = path + "cam3_" + _id;
// 		string filename0 = path + "image0\\pic" + Int_to_String(fileNum) + ".jpg";
// 		string filename1 = path + "image1\\pic" + Int_to_String(fileNum) + ".jpg";
// 		string filename2 = path + "image2\\pic" + Int_to_String(fileNum) + ".jpg";
// 		string filename3 = path + "image3\\pic" + Int_to_String(fileNum) + ".jpg";
		std::cout << "fileNum :" << filename0 << endl;
		Mat img0 = cv::imread(filename0, 0);
		Mat img1 = cv::imread(filename1, 0);
		Mat img2 = cv::imread(filename2, 0);
		Mat img3 = cv::imread(filename3, 0);

		if (!img0.data || (!img1.data && !img2.data && !img3.data))
		{
			std::cout << "读取失败" << endl;
			continue;
		}
		

		if (img1.empty())
			background1.copyTo(img1);

		if (img2.empty())
			background2.copyTo(img2);

		if (img3.empty())
			background3.copyTo(img3);

		double t1 = getTickCount();

		_detect.yanghe_detect(img0, img1, img2, img3);
		
		double t2 = getTickCount();
		t2 = (t2 - t1) * 1000 / getTickFrequency();
		cout << "=============== process time: " << t2 << "ms" << endl;

		map<int, bool> detect_result = _detect.yanghe_detectResult();

		map<int, bool>::iterator iter;
		for (iter = detect_result.begin(); iter != detect_result.end(); iter++)
		{
			int bottle_id = iter->first;

			Mat combinImage = _detect.yanghe_combineImage(bottle_id);

			string imgName = Int_to_String(iter->first) + ".jpg";
			imwrite(imgName, combinImage);
		}
		
		

	}

	return 0;
}

