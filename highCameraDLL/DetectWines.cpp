#include "StdAfx.h"
#include "DetectWines.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <stdarg.h>


using namespace cv;
using namespace std;

float length(Point2f pt1, Point2f pt2)
{
	float dx = pt1.x - pt2.x;
	float dy = pt1.y - pt2.y;

	return sqrtf(dx*dx + dy*dy);
}

Mat getSobel(const Mat src)
{
	Mat gray;
	if (src.channels() == 3)
		cv::cvtColor(src, gray, CV_BGR2GRAY);
	else
		src.copyTo(gray);
	
	//GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//���� grad_x �� grad_y ����
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad, grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// �� X�����ݶ�
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel(gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// ��Y�����ݶ�
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// �ϲ��ݶ�(����)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	return grad;

}

// ���ַ�����
int find_left_edge(Mat mask, int start, int end)
{
	int i = 0;
	while ((start <= end) && i < 12){

		i++;
		int middle = start + ((end - start) >> 1);
		int ch3 = mask.at<uchar>(middle + 2);
		int ch4 = mask.at<uchar>(middle - 2);
		// ������ch4 > ch3 �����
		if (ch3 >= 100 && ch4 < 100)
			return middle;
		else if (ch4 > 100)
			end = middle + 1;
		else
			start = middle - 1;
	}

	return -1;
}

// ���ַ�����
int find_right_edge(Mat mask, int start, int end)
{
	int i = 0;
	while (start <= end && i < 12){
		i++;
		int middle = start + ((end - start) >> 1);
		int ch4 = mask.at<uchar>(middle + 2);
		int ch3 = mask.at<uchar>(middle - 2);
		// ������ch4 > ch3 �����
		if (ch3 >= 100 && ch4 < 100)
			return middle;
		else if (ch4 > 100)
			start = middle + 1;
		else
			end = middle - 1;
	}

	return -1;

}


vector<vector<Point>> getContour(Mat src , Mat mask)
{
	// mask contour
	vector<cv::Point> mask_contour;

	Point last_point;
	for (int i = 0; i < mask.rows; i += 1)
	{
		int ch1 = mask.at<uchar>(i, mask.cols / 2);

		if (ch1 < 100)
			continue;

		int middle = mask.cols / 2;

		int left_flag = find_left_edge(mask.row(i), 0, middle);

		if (left_flag < 0)
			continue;

		mask_contour.push_back(Point(left_flag, i));
	}

	for (int i = mask.rows - 1; i >= 0; i -= 1)
	{
		int ch1 = mask.at<uchar>(i, mask.cols / 2);

		if (ch1 < 100)
			continue;

		int middle = mask.cols / 2;
		int right_flag = find_right_edge(mask.row(i), middle, mask.cols);

		if (right_flag < 0)
			continue;

		mask_contour.push_back(Point(right_flag, i));

	}

	vector<Point> mask_contour_hull;
	convexHull(mask_contour, mask_contour_hull, true);

	vector< vector<cv::Point>> impurities_contours;//��������
	vector< vector<cv::Point>> contours;
	vector<Vec4i> hierarchy;

	// impurites contour
	findContours(src, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE);

#ifdef _DEBUG
	cout << "distance to edge: ";
#endif // _DEBUG


	for (int index = 0; index < contours.size(); index++)
	{
		
		//cout << Suspectedimpurities.size() << endl;
		if (contours.at(index).size()>1000)
			continue;

		double area = cv::contourArea(contours.at(index));
		//cout << "area: " << area << endl;
		if (area >= 8.0 && area <= 250.0)//15С�ڵ� 3500��Ӭ
		{
			Moments mu;
			mu = moments(contours.at(index), false);
			Point2f mc;//����
			mc = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

			double _distance = pointPolygonTest(mask_contour_hull, mc, true);
			
#ifdef DEBUG
			cout << _distance << " ";
#endif // DEBUG
		
			if (fabs(_distance) < 20.0)		// ��ԵĿ�����
				continue;

			impurities_contours.push_back(contours.at(index));
		}
	}
#ifdef _DEBUG
	cout << endl;
#endif // _DEBUG
	

	return impurities_contours;

}

void delete_conveyor(Mat &src, int _row)
{
	int nl = src.rows;
	int nc = src.cols;
	for (int j = 0; j < nl; j++)
	{
		if (j>_row)
		{
			uchar *data = src.ptr<uchar>(j);
			for (int i = 0; i < nc; i++)
				data[i] = 0;

		}
	}
}

Mat getBottle(const Mat &src, const Mat &background, bool showFlag)
{
	Mat foreground;
	cv::absdiff(src, background, foreground);
	cv::threshold(foreground, foreground, 10, 255, 0);
	Mat openElement = getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(foreground, foreground, MORPH_OPEN, openElement);
	
	if (showFlag)
	{
		cv::namedWindow("foreground", 0);
		cv::imshow("foreground", foreground);
	}

	return foreground;
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

Mat combinImage(Mat image1, Mat image2)
{
	if (image1.empty())
		return image2;

	Mat combin = image1.t();
	Mat tmpImg = image2.t();

	combin.push_back(tmpImg);

	return combin.t();
}

// ����ͼ�������о�ƿ�����������꣬����������븨���
vector<int> DetectWines::getMidlines(Mat src, bool showFlag, int camId)
{
	vector<int> orders;
	vector<int> AllSum;

	vector<int> midLines;

	Mat Transposition = src.t();//src��ת��

	int nl = Transposition.rows;
	int nc = Transposition.cols;
	bool begin = true;//��ʼ��־

	float highT, widthT;
	if (camId == 0)
	{
		//highT = 0.05;
		//widthT = 0.07;

		highT  = _detect_bottle_start_highT0;
		widthT = _detect_bottle_widthT0;
	}
	else
	{
		//highT = 0.2;
		//widthT = 0.315;

		highT  = _detect_bottle_start_highT;
		widthT = _detect_bottle_widthT;
	}

	//	cout <<camId<<" "<< highT << " " << widthT << endl;

	Mat showImg = src.clone();
	float bottle_width = 0;
	float bottle_height = 0;
	for (int j = 0; j < nl; ++j)
	{
		uchar *data = Transposition.ptr<uchar>(j);
		int sum = 0;
		sum = countNonZero(Transposition.row(j));


		if (sum / static_cast<float>(nc) > highT)//��
		{
			// ĳ�еķ���ֵ���������趨��ֵʱ����������
			if (begin)
			{
				orders.push_back(j);
				AllSum.push_back(sum);
			}
		}
		else if (static_cast<float>(orders.size()) / static_cast<float>(nc) > widthT)//��
		{
			// ����ֵ����С�����趨��ֵ��
			// �����ۻ���ȴ����趨��ֵ,
			// ���ķ���ֵ�����������˵ĸ���
			// �������ֵΪ��������
			int _count = orders.size();
			int _midpos = accumulate(orders.begin(), orders.end(), 0) / _count;

			if (AllSum[_count/2] > AllSum[0] && AllSum[_count/2] > AllSum[_count - 1])
			{
				midLines.push_back(_midpos);
				orders.clear();
				begin = false;
			}
		}
		else
		{
			orders.clear();
			begin = true;
		}

		bottle_width = static_cast<float>(orders.size()) / static_cast<float>(nc);
		bottle_height = sum / static_cast<float>(nc);
	}

	if (showFlag && !midLines.empty())
	{
		string windowName;
		to_string_format(windowName, "camId:%02d", camId);

		//for (int n = 0; n < midLines.size(); n++)
		{
			cv::line(showImg, cv::Point(midLines[0], 0), cv::Point(midLines[0], showImg.rows), cv::Scalar(100), 5);
		}
		cv::namedWindow(windowName, 0);
		cv::imshow(windowName, showImg);
		waitKey(2);
	}
	else if (showFlag)
	{
		int _count = orders.size();
		if(_count > 10)
		{
			cout<<camId<<" bottle width "<< bottle_width<<"  bottle height "<<bottle_height<<endl;
			cout<<AllSum[_count/2]<<" "<<AllSum[0] <<" "<< AllSum[_count - 1]<<endl;
			cout<<endl;
		}
		string windowName;
		to_string_format(windowName, "camId:%02d", camId);
		cv::namedWindow(windowName, 0);
		cv::imshow(windowName, src);
		waitKey(2);
	}

	return midLines;
}

bool DetectWines::init(std::string path)
{
	string ini_File = path + "params.ini";

	cout << "input file: " << ini_File << endl;
	ifstream fp(ini_File);
	if(!fp.is_open())
	{
		cout<<"�������ļ�ʧ��"<<endl;
		return false;
	}

	fp >> _conveyorEdgeY0 >> _conveyorEdgeY1 ;

	cout <<"Ƥ���ָ�λ�ã� "<<_conveyorEdgeY0 <<" "<<_conveyorEdgeY1<<endl;

	int start_row,start_col;
	int end_row,bottle_width;
	int rows;
	// ��ƿ��������
	fp>> start_row >> end_row;
	fp>> start_col >> bottle_width;

	fp >> rows;

	//���0ƿ�����߼�����
	fp >> _detect_bottle_start_highT0 >> _detect_bottle_widthT0;
	//���1��2��3ƿ�����߼�����
	fp >> _detect_bottle_start_highT >> _detect_bottle_widthT;

	//�����camera1,camera 0����Ч���߷�Χ
	fp >> camera1_midline_lTh >> camera1_midline_hTh ;

	//�����camera2,camera 0����Ч���߷�Χ
	fp >> camera2_midline_lTh >> camera2_midline_hTh  ;

	//�����camera3,camera 0����Ч���߷�Χ
	fp >> camera3_midline_lTh >> camera3_midline_hTh ;
	
	fp.close();

	_wcount.setROI(start_row, end_row, start_col, bottle_width, rows);

	return true;
}

void DetectWines::setBackground(cv::Mat _bg0,cv::Mat _bg1,cv::Mat _bg2,cv::Mat _bg3)
{
	//���е�Ƥ������
	delete_conveyor(_bg0,_conveyorEdgeY0);
	_wcount.setBackground(_bg0);

	image_width = _bg0.cols;
	image_hight = _bg0.rows;

	assert(_bg1.channels() == 1);
	assert(_bg2.channels() == 1);
	assert(_bg3.channels() == 1);

	delete_conveyor(_bg1,_conveyorEdgeY1);
	delete_conveyor(_bg2,_conveyorEdgeY1);
	delete_conveyor(_bg3,_conveyorEdgeY1);

	_bg0.copyTo(_host_backgroud_image);

#ifdef _DEBUG
// 	namedWindow("bg0", 0);
// 	namedWindow("bg1", 0);
// 	namedWindow("bg2", 0);
// 	namedWindow("bg3", 0);
// 
// 	imshow("bg0", _bg0);
// 	imshow("bg1", _bg1);
// 	imshow("bg2", _bg2);
// 	imshow("bg3", _bg3);
// 	waitKey();
#endif // _DEBUG
	_background_images.clear();
	_background_images.push_back(_bg1);
	_background_images.push_back(_bg2);
	_background_images.push_back(_bg3);

}

int DetectWines::cameraMatchWithMidLine(double _position, vector<int> _lines)
{
	int min = INT_MAX;
	int detectMidline = 0;

	for (int index2 = 0; index2 < _lines.size(); index2++)
	{
		int dis = abs(_position - _lines.at(index2));
		if (dis< min && dis <= 100 )
		{
			min = dis;
			detectMidline = _lines.at(index2);
		}
	}
	
	return detectMidline;

	//////////////////////////////////////////////////////////////////////////
}

// ��ȡ������о�ƿ��ROI����
bool DetectWines::getImageROI(std::vector<int> _camlines, BpNet _bp, int _line, Rect& rect)
{
	//_ROI (_x,_y,w,h)
	int w = 1000;
	int h = image_hight/2 + 20;
	int _y = 500;

	double mid_line  = (double)_line/image_width;
	double *r = _bp.recognize(&mid_line);

	int matchedLine = cameraMatchWithMidLine(_bp.result[0]*image_width,_camlines);

#ifdef _DEBUG
		cout << "======= getImageROI: " << _bp.result[0] * image_width << " " << matchedLine << " " << image_width - w / 2 << endl;
#endif // _DEBUG
		
	if ( abs(image_width / 2 - matchedLine) > image_width*0.1 )//���߳����˹涨��Χ ͼ���м� [image_width*0.4,image_width*0.6]
		return false;
	else 
	{
		rect = Rect(matchedLine - w/2,_y,w,h);
		return true;
	}
	
}

// �����������㷨�ĺ��ģ�������ƿ���ݿⲿ�֣���֤��ͬ�ľ�ƿ��ͼ�����ݴ洢��һ��
// ���ұ�֤�½���ľ�ƿ��Ӳ�����ʧ�ľɾ�ƿ�����ܹ���ʱɾ��
// ��ƿ����������������̵ĺ������ݣ��������ʵ�ʶ��ƥ���Լ�����ʶ�����Ļ��Ƶ�

void DetectWines::creatWineData()
{
#ifdef _DEBUG
	cout<<"============creatWineData============== "<<endl;
	cout<<endl;
#endif // _DEBUG

	//�������0�����1��2��3��Ӧ��ϵ
	//�����camera1,camera 0����Ч���߷�Χ
// 	int camera1_midline_lTh  = 200;
// 	int camera1_midline_hTh = 500;
// 
// 	//�����camera2,camera 0����Ч���߷�Χ
// 
// 	int camera2_midline_lTh  = 1130;
// 	int camera2_midline_hTh  = 1480;
// 
// 	//�����camera3,camera 0����Ч���߷�Χ
// 
// 	int camera3_midline_lTh  = 1980;
// 	int camera3_midline_hTh  = 2380;

	int bottle_num = _wine_Id.size();
	for (int i=0; i< bottle_num; i++)
	{
		int _mid_line = _wine_middle_position[i];

#ifdef _DEBUG
		cout<<"���: "<<_wine_Id[i]<<" ����: "<< _mid_line <<endl;
#endif // _DEBUG
		
		int _bottle_id = _wine_Id[i];

		int cameraId;
		Mat wine_image;
		Mat wine_image_fg;
		Rect _roi;
		if(_mid_line > camera1_midline_lTh && 
			_mid_line < camera1_midline_hTh)
		{
			cameraId = 1;

			if (_cam1_wine_mid_lines.empty())
			{
#ifdef _DEBUG
				cout << "===== no line in camera 1 " << endl;
#endif // _DEBUG
				continue;
			}

			if (!getImageROI(_cam1_wine_mid_lines, _cam1_bpNet, _mid_line, _roi))
			{
#ifdef _DEBUG
				cout << "=====no match in camera 1 " << endl;
#endif // _DEBUG
				
				continue; // �����1����ƥ��ƿ�ӣ�ֱ�ӷ���forѭ��
			}
			_cam1Img.copyTo(wine_image);
			_cam1_bottle.copyTo(wine_image_fg);
		}
		else if(_mid_line > camera2_midline_lTh &&
			_mid_line < camera2_midline_hTh)
		{
			cameraId = 2;

			if (_cam2_wine_mid_lines.empty())
			{
#ifdef _DEBUG
				cout << "===== no line in camera 2 " << endl;
#endif // _DEBUG
				continue;
			}

			if(!getImageROI(_cam2_wine_mid_lines,_cam2_bpNet,_mid_line,_roi))
			{		// �����2����ƥ��ƿ�ӣ�ֱ�ӷ���forѭ��
#ifdef _DEBUG
				cout << "=====no match in camera 2 " << endl;
#endif // _DEBUG
				continue;
			}
			
			_cam2Img.copyTo(wine_image);
			_cam2_bottle.copyTo(wine_image_fg);
		}
		else if(_mid_line > camera3_midline_lTh && 
			_mid_line < camera3_midline_hTh)
		{
			cameraId = 3;

			if (_cam3_wine_mid_lines.empty())
			{
#ifdef _DEBUG
				cout << "===== no line in camera 3 " << endl;
#endif // _DEBUG
				continue;
			}

			if (!getImageROI(_cam3_wine_mid_lines, _cam3_bpNet, _mid_line, _roi))
			{
#ifdef _DEBUG
				cout << "=====no match in camera 3 " << endl;
#endif // _DEBUG
				continue;
			}

			_cam3Img.copyTo(wine_image);
			_cam3_bottle.copyTo(wine_image_fg);

		}
		else
		{
#ifdef _DEBUG
			cout << "=====no match in all camera " << endl;
#endif // _DEBUG
			continue; // ���0�е�ƿ�Ӳ������1��2��3���Ƿ�Χ��
		}


		//���ݾ�ƿ�����Ϣ�����µ����ݻ������������
		bool _found_bottle = false;
		int _found_ID ;
		for (int i = 0; i < _wine_data.size(); i++)
		{
			int __uuID = _wine_data[i]._uuID;
			if( _bottle_id == __uuID ) //���ݿ����ҵ���Ӧƿ��
			{
				_found_ID = i;
				_found_bottle = true;
			}
		}
#ifdef _DEBUG
		cout<<"ƥ�����: "<<cameraId<<endl;
		cout<<"���ݿ��ѯ: "<<_found_bottle<<endl;
#endif // _DEBUG
		if(!_found_bottle)		//���ݿ���û�в�ѯ�����´���
		{
			WineData _wd(_bottle_id,true);
			
			_wd._changed_finished = false;// �޸�����ͬ����ʶ

			_wd._camId.push_back(cameraId);
			_wd._wine_image.push_back(wine_image);
			_wd._wine_image_fg.push_back(wine_image_fg);
			_wd._roi_rect.push_back(_roi);
			_wd._number++;
			vector<WineData> vec_wd;
			vec_wd.push_back(_wd);

			detect_impuritie(vec_wd.begin(),0);

			vec_wd[0]._changed_finished = true;	//�޸�����ͬ�����

			_wine_data.push_back(vec_wd[0]);		// �´�����WineData����

			// ���ݿ��ʼ��ʱ����
			_wine_data_added.push_back(false);

			// ��һ�����_draw_result_image
			Mat _roi_img = wine_image(_roi);
//#ifdef _DEBUG
			string str;
			to_string_format(str, "%02d", _wd._number);
			putText(_roi_img, str, Point(_roi.width / 2, _roi.height / 2),
				FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0), 3, CV_AA);
			cout << "rect1: " << _bottle_id<<" "<<_roi << endl;
//#endif // _DEBUG
			Mat _colorImg = getDrawedImage(_roi_img, _bottle_id, cameraId);
			_draw_result_image[_bottle_id] = _colorImg;
			
			//
		}
		else
		{
			//WineData _wd = _wine_data[_found_ID];

			vector<WineData>::iterator _wd = _wine_data.begin() + _found_ID;
			_wd->_changed_finished = false;// �޸�����ͬ����ʶ

			_wd->_camId.push_back(cameraId);
			_wd->_wine_image.push_back(wine_image);
			_wd->_wine_image_fg.push_back(wine_image_fg);
			_wd->_roi_rect.push_back(_roi);
			_wd->_number++;
			// ����ӵ����ݽ������ʼ��
			int _num = _wd->_wine_image.size();
			detect_impuritie(_wd,_num - 1 );

			_wd->_changed_finished = true;	//�޸�����ͬ�����

			// ���ݿ����������޸ģ���Ӧ���_found_ID
			_wine_data_added[_found_ID] = true;

			// �������_draw_result_image
			Mat _roi_img = wine_image(_roi);
			string str;
			to_string_format(str, "%02d", _wd->_number);
			putText(_roi_img, str, Point(_roi.width / 2, _roi.height / 2),
				FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0), 3, CV_AA);
#ifdef _DEBUG		
			cout << "bottle " << _bottle_id << " add image ..." << endl;
			cout << "rect2: " << _roi << endl;
#endif // _DEBUG

			Mat _colorImg = getDrawedImage(_roi_img, _bottle_id, cameraId);
			Mat _combinImage = combinImage(_draw_result_image[_bottle_id],_colorImg);
			_draw_result_image[_bottle_id] = _combinImage;

#ifdef _DEBUG

			cout<<"ԭͼ����: "<<_wine_data[_found_ID]._wine_image.size()<<endl;
			cout<<"ǰ��ͼ����: "<<_wine_data[_found_ID]._wine_image_fg.size()<<endl;
			cout<<"ROI����: "<<_wine_data[_found_ID]._roi_rect.size()<<endl;
			cout<<"������Ŀ: "<<_wine_data[_found_ID]._impurities[_num -1].size()<<endl;

#endif // _DEBUG
		}
	} // ����������

	//ɾ�����ݿ�_wine_data �� �ϳ�ͼƬ
	for (int i = 0; i < _deleted_wine_Id.size(); i++)
	{
		int _bottle_id = _deleted_wine_Id[i];

		// _wine_data
		for (int j = 0; j < _wine_data.size(); j++)
		{
			if( _bottle_id == _wine_data[j]._uuID )
			{
				/*
				// ͼƬ����
				_wine_data[j]._wine_image.clear();
				_wine_data[j]._wine_image_fg.clear();
				_wine_data[j]._roi_rect.clear();

				// ��������
				int _num_imprt = _wine_data[j]._impurities.size();

				for (int k = 0; k < _num_imprt; k++)
					_wine_data[j]._impurities[k].clear();
				
				_wine_data[j]._impurities.clear();
				*/

				// ��ƿ����
				_wine_data.erase(_wine_data.begin() + j);
				_wine_data_added.erase(_wine_data_added.begin() + j);
				j--;
			}
		}// _wine_data


		// _draw_result_image
		auto iter = _draw_result_image.begin();
		while(iter != _draw_result_image.end())
		{
			if( _bottle_id == iter->first )
			{
				_draw_result_image.erase(iter++);
			}
			else
			{
				iter++;
			}
		}	// _draw_result_image	
	}

#ifdef _DEBUG
	cout<<"============creatWineData============== "<<endl;
	cout<<endl;
#endif // _DEBUG

}

// �����0�ľ�ƿ����
void DetectWines::countWines(cv::Mat source)
{
	Mat gray;
	if(source.channels() == 3)
		cvtColor(source,gray,CV_BGR2GRAY);
	else
		source.copyTo(gray);

	delete_conveyor(gray, _conveyorEdgeY0);

	_wcount.setImage(gray);

	_wcount.countWines();

	_wine_Id = _wcount.getWinesId();

	_deleted_wine_Id = _wcount.getDeletedWineId();

	_wine_middle_position = _wcount.getWinesMiddlePosition();

//#ifdef _DEBUG

	cout<<"countWines: "<<_wine_Id.size()<<endl;

	Mat countImage = _wcount.drawResult();
	namedWindow("Count Bottle",0);
	imshow("Count Bottle",countImage);
	waitKey(10);

//#endif

}

#ifdef _DEBUG

Mat drawCameraMiddleLine( Mat gray, vector<int> midLines, int height)
{
	Mat color_img;
	cvtColor(gray,color_img,CV_GRAY2BGR);

	for (int i = 0; i < midLines.size(); i++)
	{
		int _pos = midLines[i];

		line(color_img,Point(_pos,0),Point(_pos,height),Scalar(0,0,255),2);
	}

	return color_img;
}

#endif

// �ֱ���㸨���1,2,3ͼ���о�ƿ���ߣ������������

void DetectWines::calcWineMiddleLine()
{
	Mat gray1,gray2,gray3;

	//�۳����ʹ�����
	if(_cam1Img.channels() == 3)
		cvtColor(_cam1Img,gray1,CV_BGR2GRAY);
	else
		_cam1Img.copyTo(gray1);
	delete_conveyor(gray1, _conveyorEdgeY1);

	if(_cam2Img.channels() == 3)
		cvtColor(_cam2Img,gray2,CV_BGR2GRAY);
	else
		_cam2Img.copyTo(gray2);
	delete_conveyor(gray2, _conveyorEdgeY1);

	if(_cam3Img.channels() == 3)
		cvtColor(_cam3Img,gray3,CV_BGR2GRAY);
	else
		_cam3Img.copyTo(gray3);
	delete_conveyor(gray3, _conveyorEdgeY1);

	//��ֵ��
	_cam1_bottle = getBottle(gray1,_background_images[0],false);
	_cam2_bottle = getBottle(gray2,_background_images[1],false);
	_cam3_bottle = getBottle(gray3,_background_images[2],false);

	_cam1_wine_mid_lines = getMidlines(_cam1_bottle,false,1);
	_cam2_wine_mid_lines = getMidlines(_cam2_bottle,false,2);
	_cam3_wine_mid_lines = getMidlines(_cam3_bottle,false,3);

#ifdef _DEBUG

	Mat drawImg1 = drawCameraMiddleLine(gray1,_cam1_wine_mid_lines,image_hight);
	Mat drawImg2 = drawCameraMiddleLine(gray2,_cam2_wine_mid_lines,image_hight);
	Mat drawImg3 = drawCameraMiddleLine(gray3,_cam3_wine_mid_lines,image_hight);

	namedWindow("mid1",0);
	imshow("mid1",drawImg1);
	namedWindow("mid2",0);
	imshow("mid2",drawImg2);
	namedWindow("mid3",0);
	imshow("mid3",drawImg3);
	waitKey(10);
#endif

}

void DetectWines::setImageDate(cv::Mat img1,cv::Mat img2,cv::Mat img3)
{
	_cam1Img = img1.clone();
	_cam2Img = img2.clone();
	_cam3Img = img3.clone();
}

// ��װ�������ϲ��ֲ���
void DetectWines::updateWineData(cv::Mat cam1Img,cv::Mat cam2Img, cv::Mat cam3Img)
{
	setImageDate(cam1Img,cam2Img,cam3Img);
	// �ֱ���㸨���1,2,3ͼ���о�ƿ����
	calcWineMiddleLine();
	// ���ݸ����1,2,3ͼ���о�ƿ����λ�ã��������ݿ�
	creatWineData();
}

// ��ͬ��ƿ�Ĳ�ͬͼ�����ݽ�������������ƥ��
// ����ʶ�𵽵������Լ���ƥ����

void DetectWines::processWineData()
{
#ifdef _DEBUG
	cout << endl;
	cout << "==============processWineData===============" << endl;
	cout << endl;
#endif // _DEBUG
	int _num = _wine_data.size();
	for (int index = 0; index < _num; index++)
	{
		// ��ǰƿ��û����������
		if (!_wine_data_added[index])
			continue;
		vector<WineData>::iterator _wd = _wine_data.begin() + index;

#ifdef DEBUG
		cout << "bottle " << _wd->_uuID << " add image ..." << endl;
#endif // DEBUG		

		int _image_num = _wd->_wine_image.size();

		if (_image_num < 2) // ����2��ͼƬ�������бȽ�
			continue;

		// ������֡ͼ�񣬼�������ƥ��
		std::vector<Impuritie> _imprt1 = _wd->_impurities[_image_num - 1];
		std::vector<Impuritie> _imprt2 = _wd->_impurities[_image_num - 2];

		map<int, int> _match1 = contoursMatch(_imprt1, _imprt2);
		map<int, int> _match2 = contoursMatch(_imprt2, _imprt1);

		map<int, int> _new_match;
		// ��������ƥ��������������λ��
		for (int i = 0; i < _imprt1.size(); i++)
		{
			int _mId1 = _match1[i];
			if (_mId1 < 0)
				continue;

			int _mId2 = _match2[_mId1];

			// ˫��ƥ��
			if (_mId2 == i)
			{

				_new_match[i] = _mId1;

				vector<Impuritie>::iterator it1 = _imprt1.begin() + i;
				vector<Impuritie>::iterator it2 = _imprt2.begin() + _mId1;

				float _dist = length(it1->_center, it2->_center);
				it1->_dist = it2->_dist + _dist;

				if (it1->_dist > 100) // �����ƶ������ۻ��㹻��
				{
					_wd->_is_good = false;
				}
			}// ˫��ƥ��
		}

		// ����
		drawLines(_wd, _new_match);

		// ������ϣ�����
		_wine_data_added[index] = false;

#ifdef _DEBUG
		cout << " matches = " << _new_match.size() << endl;
		cout << endl;
		cout << "==============processWineData===============" << endl;
		cout << endl;
#endif // _DEBUG
	}
}


const float areaT = 50.0;//�����ֵ
const float distT = 100.0;//������ֵ
const float distM = 4.0 ;//��������ֵ

map<int, int> DetectWines::contoursMatch(std::vector<Impuritie> imprt1, std::vector<Impuritie> imprt2)
{
	map<int,int> _match;

	// imprt2 Ϊ�¼�����ʣ�imprt1Ϊǰһ֡��������
	std::vector<Impuritie>::iterator it1;
	std::vector<Impuritie>::iterator it2;
	int i = 0;

#ifdef _DEBUG

	cout << "contours match: "<<endl;

#endif // _DEBUG

	for (it1 = imprt1.begin(); it1 != imprt1.end(); it1++,i++)
	{
		bool method = false;
		if(it1->_area < areaT ) //�����С���÷���1�������÷���0
			method = true;

		double _min = INT_MAX;
		int minIndex = -1;

		int j = 0;
		for (it2 = imprt2.begin(); it2 != imprt2.end(); it2++,j++)
		{
			bool _method = false;
			if ( it2->_area > areaT )
				_method = true;
			
//			if ( method != _method )	// ���������һ��
//				continue;

			float _distance;
			_distance = length(it1->_center, it2->_center);

#ifdef _DEBUG
			cout << i << " " << j << " " << _distance << " " << abs(it1->_center.y - it2->_center .y)<< endl;
#endif // _DEBUG


			if ( _distance >= distT )
				continue;		
/*
			if(_method)
			{
				_distance = matchShapes( it1->contours, it2->contours, CV_CONTOURS_MATCH_I2, 0.0);
				if (_distance > distM)
					continue;
			}
*/

			if(_min > _distance)
			{
				_min = _distance;
				minIndex = j;
			}

		}

	   // �����ҵ�ƥ��Ķ���ʱ��_match[i] = 1;
	   // �����Ҳ���ƥ��Ķ���ʱ��_match[i] = -1��

		_match[i] = minIndex;

	}

	return _match;
}

Mat DetectWines::getImage(int i,int j)
{
	Mat roi_img = (_wine_data[i]._wine_image[j])( _wine_data[i]._roi_rect[j] );

	Mat color_roi_img;
	if(roi_img.channels() == 1)
		cvtColor(roi_img,color_roi_img,CV_GRAY2BGR);
	else
		roi_img.copyTo(color_roi_img);

	// ƿ�ӱ����������

	int uuid = _wine_data[i]._uuID;
	int camId = _wine_data[i]._camId[j];

	string bottle_id_str;
	to_string_format(bottle_id_str,"%04d",uuid);

	string cam_id_str;
	to_string_format(cam_id_str,"%02d",camId);

	string str1 = "No. :" + bottle_id_str + "/";
	string str2 = "Cam: " + cam_id_str;

	int _w = color_roi_img.cols;
	int _h = 60;
	
	putText(color_roi_img,str1,Point(_w/5,_h),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,0,0),3,CV_AA);
	putText(color_roi_img,str2,Point(3*_w/5,_h),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,0,0),3,CV_AA);

	return color_roi_img;
}

Mat DetectWines::getDrawedImage(Mat& roi_img, int uuid,int camId)
{

	Mat color_roi_img;
	if (roi_img.channels() == 1)
		cvtColor(roi_img, color_roi_img, CV_GRAY2BGR);
	else
		roi_img.copyTo(color_roi_img);

	string bottle_id_str;
	to_string_format(bottle_id_str, "%04d", uuid);

	string cam_id_str;
	to_string_format(cam_id_str, "%02d", camId);

	string str1 = "No. :" + bottle_id_str + "/";
	string str2 = "Cam: " + cam_id_str;

	int _w = color_roi_img.cols;
	int _h = 60;

	putText(color_roi_img, str1, Point(_w / 5, _h), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 0, 0), 3, CV_AA);
	putText(color_roi_img, str2, Point(3 * _w / 5, _h), FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255, 0, 0), 3, CV_AA);

	return color_roi_img;
}

void DetectWines::addImage()
{
// 	if(_wine_data.empty())
// 		return;
// 
// 	if(_draw_result_image.empty())
// 	{
// 		for (int i = 0; i < _wine_data.size(); i++)
// 		{
// 			int uuid = _wine_data[i]._uuID;
// 			Mat _colorImg = getImage(i,0);
// 
// 			_draw_result_image[uuid] = _colorImg;
// 		}
// 	}
// 	else
// 	{
// 		auto iter = _draw_result_image.begin();
// 
// 		while(iter != _draw_result_image.end())
// 		{
// 			int uuid = iter->first;
// 			for (int i = 0 ; i < _wine_data.size(); i++)
// 			{
// 				if( uuid == _wine_data[i]._uuID )
// 				{
// 					int j = _wine_data[i]._wine_image.size();
// #ifdef _DEBUG
// 					cout<<" 2 ================ addImage "<<endl;
// 					cout<<uuid<<" "<<_wine_data[i]._camId[j-1]<<endl;
// 
// #endif // _DEBUG
// 										
// 					Mat _colorImg = getImage(i,j-1);
// 
// 					Mat _resultImg;
// 					
// 					_resultImg = combinImage(iter->second, _colorImg);
// 
// 					_resultImg.copyTo(iter->second);
// 				}
// 			}
// 
// 			iter++;
// 		}
// 	}
}

void DetectWines::drawLines(vector<WineData>::iterator wd, std::map< int,int > matchedId )
{

	int uuid = wd->_uuID;
	Mat _colorImg = _draw_result_image[uuid];

	int image_num = wd->_wine_image.size();

	int __width = wd->_roi_rect[0].width;

	int _startX2 = (image_num - 2)*__width;
	int _startX1 = (image_num - 1)*__width;

#ifdef _DEBUG


#endif // _DEBUG

	vector<Impuritie> imprt1 = wd->_impurities[image_num - 1];
	vector<Impuritie> imprt2 = wd->_impurities[image_num - 2];

	

#ifdef _DEBUG

//	cout<<" X1 = "<<_startX1<<" X2 = "<<_startX2<<endl;
	cout<<"drawLines: impurites1_size = "<<imprt1.size() <<" impurites2_size = "<< imprt2.size() << endl;

#endif // _DEBUG
	vector<Impuritie>::iterator iter1 = imprt1.begin();
	vector<Impuritie>::iterator iter2 = imprt2.begin();
	
	int _radius = 10;
	
	while(iter1 != imprt1.end())
	{
		Point2f center = iter1->_center;

		Point2f point = Point2f(center.x + _startX1,center.y);

		circle(_colorImg, point, _radius, Scalar(0, 0, 255), 1, CV_AA);

		iter1++;
	}

	while(iter2 != imprt2.end())
	{
		Point2f center = iter2->_center;

		Point2f point = Point2f(center.x + _startX2,center.y);

		circle(_colorImg, point, _radius, Scalar(0, 0, 255), 1, CV_AA);

		iter2++;
	}

	auto iter = matchedId.begin();
	while( iter != matchedId.end() )
	{
		int index1 = iter->first;
		int index2 = iter->second;

		Point2f center1 = imprt1[index1]._center;
		Point2f center2 = imprt2[index2]._center;

		Point2f start_point = Point2f(center1.x + _startX1 - _radius , center1.y);
		Point2f end_point = Point2f(center2.x + _startX2 + _radius, center2.y);

		line(_colorImg,start_point,end_point,Scalar(0,0,255),2,CV_AA);

		iter++;
	}


#ifdef _DEBUG
	namedWindow("drawLine", 0);
	imshow("drawLine", _colorImg);
	waitKey(2);
#endif // _DEBUG
}

cv::Mat DetectWines::getCountWineResult()
{
	return _wcount.drawResult();
}

bool DetectWines::readBpNet(string path)
{

	string bpfile1 = path + "wb_Camera1";
	if (!_cam1_bpNet.readtrain(bpfile1))
	{
		return false;
	}

	string bpfile2 = path + "wb_Camera2";
	if(!_cam2_bpNet.readtrain(bpfile2))
		return false;

	string bpfile3 = path + "wb_Camera3";
	if(!_cam3_bpNet.readtrain(bpfile3))
		return false;

	return true;

}

//ͼ��Աȶ�����
int linearExpand(Mat& src,Mat mask, Mat& dst)
{
	if(!src.data){  
		cout<<"Miss Data"<<endl;  
		return -1;  
	}

	if(dst.empty())
		dst = src.clone();


	//���Ա任
	double minV = 255 ,maxV = 0;
	int w = src.cols;
	int h = src.rows;

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int ch =(int)mask.ptr<uchar>(i)[j];

			if(ch > 100)
			{
				ch = (int)src.ptr<uchar>(i)[j];
				if(ch > maxV)
					maxV = ch;

				if(ch < minV)
					minV = ch;
			}
		}
#ifdef _DEBUG

	cout << "linearExpand:  max = " << maxV << " min = " << minV << endl;

#endif // _DEBUG
		
		if(maxV - minV < 1e-2)
			return -1;

		for (int i=0; i<h; i++)
			for(int j=0; j<w;j++)
			{
				int val = src.ptr<uchar>(i)[j];
				dst.ptr<uchar>(i)[j] = saturate_cast<uchar>(255*((val - minV)/(maxV-minV)));
			}

			return 0;
}

// ��ƿ��������
int changeROI(Mat& mask1, Mat& mask2)
{
	int row_shrink = 40;
	int col_cut	= 20;

	mask1.copyTo(mask2);

	int w = mask1.cols;
	int h = mask1.rows;
	mask2(Range(0,col_cut),Range(0,w)) = 0;
	mask2(Range(h-col_cut,h),Range(0,w)) = 0;

	for (int i = 0; i < h; i++)
	{
		//cout<<i<<": ";
		Mat _row1 = mask1.row(i);
		int j = 0;
		while(j < w)
		{
			if (_row1.at<uchar>(j) < 100)
				j++;
			else
				break;
		}

		
		if (j > w/2 )
			continue;

		//cout << j << " ";
		for (int k=0; k<row_shrink; k++)
		{
			int index = j+k;
			mask2.ptr<uchar>(i)[index] = 0;
		}

		j = w-1;

		
		while(j > 0)
		{
			if(_row1.at<uchar>(j) < 100)
				j--;
			else
				break;
		}

		if (j < w / 2)
			continue;

		//cout << j << endl;
		for (int k=0; k<row_shrink; k++)
		{
			int index = j-k;
			mask2.ptr<uchar>(i)[index] = 0;
		}
	}

	return 0 ;
}

// ��ֵ��
Mat getBinaryImage(Mat& src, Mat& mask)
{
	Mat fg;

	//ƿ���������������˱�Ե��ֵ������
	changeROI(mask,fg);

	fg.copyTo(mask);

	Mat foreground;
	// ǰ����ȡ
	bitwise_and(src,fg,foreground);
	// �Աȶ�����
	linearExpand(foreground,fg,foreground); 

	Mat grad, bin0, bin1, bin2;
	//����Ӧ��ֵ�ָ�
	adaptiveThreshold(foreground, bin1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 31, 10);
	cv::erode(bin1, bin1, Mat(),Point(-1,-1),1);
	cv::dilate(bin1, bin1, Mat(),Point(-1,-1),1);

	// Ӳ��ֵ�ָ��ȡ�Ҷ�С��70�ĺڵ������
	threshold(src, bin0, 70, 255, CV_THRESH_BINARY_INV);
	cv::dilate(bin0, bin0, Mat(),Point(-1,-1),1);
	cv::bitwise_and(bin1,bin0,bin2);

	Mat Suspectedimpurities;
	Mat Element = getStructuringElement(MORPH_RECT, Size(5, 5));
	cv::morphologyEx(bin2, Suspectedimpurities, MORPH_CLOSE, Element);

//	cv::bitwise_and(Suspectedimpurities, fg, Suspectedimpurities);

	return Suspectedimpurities;
}

// ��ȡ��������
void DetectWines::detect_impuritie(vector<WineData>::iterator wd ,int i)
{
	/*
	//////////////////////////////////////////////////////////////////////////
	
	Mat Suspectedimpurities;//��������
	if (src_roi.channels() == 3)
	cv::cvtColor(src_roi, Suspectedimpurities, CV_BGR2GRAY);
	else
	Suspectedimpurities = src_roi.clone();

	//cv::bitwise_and(gray,mask,Suspectedimpurities);//��ƿ������
	Mat grad, bin1, bin2;
	adaptiveThreshold(Suspectedimpurities, bin1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 31, 10);//��ʶ����������

	//	grad = getSobel(Suspectedimpurities);

	//	Canny(Suspectedimpurities, bin2, 20, 55);

	dilate(bin1, bin1, Mat());

	erode(bin1, bin1, Mat());

	//	bitwise_or(bin1, bin2, Suspectedimpurities);

	Mat Element = getStructuringElement(MORPH_RECT, Size(3, 3));

	morphologyEx(bin1, Suspectedimpurities, MORPH_OPEN, Element);

	bitwise_and(Suspectedimpurities, foreground_roi, Suspectedimpurities);

	#ifdef _DEBUG
	namedWindow("binary",0);
	imshow("binary",src_roi);
	waitKey(10);
	#endif // _DEBUG

	//////////////////////////////////////////////////////////////////////////
	*/
	//ͼ���ֵ��
	Mat src_roi = (wd->_wine_image[i])(wd->_roi_rect[i]);
	Rect r = wd->_roi_rect[i];
	Mat foreground_roi = (wd->_wine_image_fg[i])(r);

	Mat binary = getBinaryImage(src_roi, foreground_roi);

//	imwrite("src_roi.bmp", src_roi);
//	imwrite("fg_roi.bmp", foreground_roi);
//	imwrite("binary.bmp", binary);

	//��������
	vector<vector<Point>> contours = getContour(binary, foreground_roi);

#ifdef _DEBUG

	Mat showSuspectedimpurities;
	if (src_roi.channels() == 1)
		cv::cvtColor(src_roi, showSuspectedimpurities, CV_GRAY2BGR);
	else
		showSuspectedimpurities = src_roi.clone();

	string cam_id;
	to_string_format(cam_id, "%02d", wd->_camId[i]);

	cout << cam_id <<" "<<(wd->_camId[i]) << endl;

	cv::namedWindow(cam_id, 0);
	drawContours(showSuspectedimpurities, contours, -1, cv::Scalar(0,0,255), 2);
	cv::imshow(cam_id, showSuspectedimpurities);
	cv::waitKey(10);

#endif

	vector<Impuritie> __impurities;
	for (int i = 0 ; i < contours.size(); i++)
	{
		vector<Point> _contour = contours[i];

		Moments mu1;
		mu1 = moments(_contour, false);
		Point2f mc1;//����
		mc1 = Point2f(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);

		float _area = contourArea(_contour);

		Impuritie impuritie(_contour);
		impuritie._center = mc1;
		impuritie._area = _area;

		__impurities.push_back(impuritie);		
	}

	wd->_impurities.push_back(__impurities);  //�����������

}

// ����ʶ������_wd._is_goodΪ�жϵ�ǰ��ƿ�Ƿ������ʵı�ʶ
map<int,bool> DetectWines::getResult()
{
	map<int,bool> _result;

	for (int i = 0; i< _wine_data.size(); i++)
	{
		WineData _wd = _wine_data[i];

		int uuid = _wd._uuID;
		_result[uuid] = _wd._is_good;
	}

	return _result;
}

DetectWines::DetectWines(void)
{
	_wine_data.clear();
	_wine_data_added.clear();
}

DetectWines::~DetectWines(void)
{
}
