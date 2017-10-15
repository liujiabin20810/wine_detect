
#include "WineCount.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define  _DEBUG

using namespace std;

void WineCount::setROI(int start_row,int end_row, int start_col, int bottle_width , int row)
{
	_start_row = start_row;

	_end_row = end_row;

	_start_col = start_col;

	_bottle_width = bottle_width;

	_i_row = row;
}

void WineCount::setBackground(cv::Mat bg)
{
	cv::Mat gray;

	if(bg.channels() == 3)
		cvtColor(_source,gray,CV_BGR2GRAY);
	else
		bg.copyTo(gray);

	gray.copyTo(_bg);
}

void WineCount::setImage(cv::Mat image)
{
	//设置原图
	image.copyTo(_source);

	cv::Mat gray;
	if(_source.channels() == 3)
		cvtColor(_source,gray,CV_BGR2GRAY);
	else
		_source.copyTo(gray);

	//计算二值图
	cv::Mat foreground;
	cv::absdiff(_bg,gray, foreground);

	cv::threshold(foreground, _binary, 10, 255, cv::THRESH_BINARY);

	cv::Mat openElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	cv::morphologyEx(_binary, _binary, cv::MORPH_OPEN, openElement);

	cv::namedWindow("_bg", 0);
	cv::imshow("_bg",_binary);
	cv::waitKey(2);
}

bool WineCount::bottle_online()
{
	int count = 0;

	int len = _end_row - _start_row;

	for (size_t i = _start_row; i < _end_row; i++)
	{
		uchar *p = _binary.ptr<uchar>(i);

		if (p[_start_col] > 0 ) 
			count++;

		if (count >= 0.2*len)
			return true;
	}

	return false;
}

bool WineCount::detect_bound()
{
	int _width = 0;
	int _left_border,_right_border;
	//Left
//	for (int i = _start_row; i <= _end_row; ++i)
	
	int right_border = _start_col, left_border = _start_col;

	uchar *p = _binary.ptr<uchar>(_i_row);

	while (p[left_border] == 255)
	{
		if (left_border == 0)
			break;

		--left_border;
	}

	while (p[right_border] == 255)
	{
		if (right_border == _source.cols - 1)
			break;

		++right_border;
	}

	_width = right_border - left_border;
	_right_border = right_border;
	_left_border = left_border;

//	cout << "======================" << endl;
//	cout << "set i_row : " << _i_row << endl;

	if (_width < _bottle_width || _left_border == 0)
	{
		cout << "width = " << _width << " " << _bottle_width << " " << _i_row << endl;
		cout << "too narrow or error start... " << endl;
		return false;
	}
	else
	{
//		cout<<"left_bd =  "<<_left_border<<", right_bd = "<<_right_border<<endl;

		_wine_bounds.push_back(std::make_pair(_left_border,_right_border));

		_wine_num++;
		_wine_count_num++;
		_wine_Id.push_back(_wine_count_num);
		
	}

	return true;

}

void WineCount::update_bound()
{
	if(_wine_bounds.empty())
		return;

	uchar *p = _binary.ptr<uchar>(_i_row);

	vector<std::pair<int, int>>::iterator iter = _wine_bounds.begin();
	vector<int>::iterator iter_id = _wine_Id.begin();

	while (iter != _wine_bounds.end() )
	{
		bool rightOk = false;

		int left_bound = 0, rigth_bound = 0;
		for (size_t j = MAX(5, iter->first); j < _binary.cols - 6; j++)
		{
			if (p[j] == 255 && p[j - 5] < 255 && p[j + 5] == 255)//Left
			{
				left_bound = j;
				rightOk = true;
				break;
			}
		}
		
		for (size_t j = MAX(5, iter->second-1); j < _binary.cols - 6; j++)
		{
			if (p[j] == 255 && p[j - 5] == 255 && p[j + 5] < 255 && rightOk)
			{
				rigth_bound = j;
				break;
			}
		}

#ifdef _DEBUG
//		cout << ": left_bd =  " << left_bound << ", right_bd = " << rigth_bound << endl;
#endif
 		if ((left_bound == 0 && rigth_bound == 0) || left_bound >= rigth_bound)
		{
#ifdef _DEBUG
			cout << left_bound << " " << rigth_bound << endl;
			cout << *iter_id << "号瓶子未检测到，删除......\n";
#endif

			_delete_wine_Id.push_back(*iter_id);

			iter = _wine_bounds.erase(iter);
			iter_id = _wine_Id.erase(iter_id);

			_wine_num--;
		}
		else
		{
			iter->first = left_bound;
			iter->second = rigth_bound;

			iter_id++;
			iter++;
		}
	}

// 	for (size_t i = 0; i < _wine_num; i++)
// 	{
// 
// 		bool rightOk = false;
// 		
// 		int left_bound = 0,rigth_bound = 0;
// 		
// 		for (size_t j = MAX(5,_wine_bounds[i].first ); j < _binary.cols - 6; j++)
// 		{
// 			if (p[j] == 255 && p[j - 5] < 255 && p[j + 5] == 255 && !rightOk)//Left
// 			{
// 				left_bound = j;
// 				rightOk = true;
// 			}
// 
// 			if (p[j] == 255 && p[j - 5] == 255 && p[j + 5] < 255 && rightOk)
// 			{
// 				rigth_bound = j;
// 				break;
// 			}
// 		}
// 
// 		cout<<i<<": left_bd =  "<<left_bound<<", right_bd = "<<rigth_bound<<endl;
// 		
// 		if (left_bound == 0 || rigth_bound == 0 || left_bound >= rigth_bound)
// 		{
// 			
// 
// 			//瓶子编号是从小到大顺序存储，每次删除的都是_wine_Id[0]
// 
// 			//_deleted_id.push_back(i);
// 			_wine_bounds.erase(_wine_bounds.begin() + i);
// 			_wine_Id.erase(_wine_Id.begin() + i);
// 
// 			i--;
// 			//酒瓶数不能减1
// 			_wine_num--;
// 		}
// 		//更新坐标
// 		else
// 		{
// 			
// 			_wine_bounds.at(i).first = left_bound;
// 			_wine_bounds.at(i).second = rigth_bound;
// 
// 		}
// 	}
	
}

void WineCount::countWines()
{
	if(!_is_online)
	{
		if(bottle_online())
		{
			bool flag = detect_bound();
			if (flag)
				_is_online = true;
#ifdef _DEBUG
			if(!flag)
			{
				cout<<"detect bound failed..."<<endl;
			}
			else
				cout<<_wine_Id[_wine_num-1]<<"号瓶子进入...\n";
#endif
			
		}
	}
	else
	{
		if(!bottle_online())
			_is_online = false;
	}

	//更新坐标信息
	update_bound();

}

cv::Mat WineCount::drawResult()
{
	cv::Mat drawImg ;

	if(_source.channels() != 3)
		cvtColor(_source,drawImg,CV_GRAY2BGR);
	else
		drawImg = _source.clone();

	cv::line(drawImg,cv::Point(0,_start_row),cv::Point(_source.cols,_start_row),cv::Scalar(0,0,255),2);
	cv::line(drawImg,cv::Point(0,_end_row),cv::Point(_source.cols,_end_row),cv::Scalar(0,0,255),2);

	cv::line(drawImg,cv::Point(_start_col,_start_row),cv::Point(_start_col,_end_row),cv::Scalar(0,0,255),2);

	cv::line(drawImg,cv::Point(_start_col,_i_row),cv::Point(_source.cols,_i_row),cv::Scalar(0,0,255),2);

	//绘制瓶子边界以及编号
	for (int i = 0; i < _wine_num; i++)
	{
		int left_bound = _wine_bounds[i].first;
		int right_bound = _wine_bounds[i].second;

		cv::line(drawImg,cv::Point(left_bound,_start_row), cv::Point(left_bound,_end_row),cv::Scalar(255,0,0),2);
		cv::line(drawImg,cv::Point(right_bound,_start_row),cv::Point(right_bound,_end_row),cv::Scalar(255,0,0),2);

		char _id[10];
		sprintf(_id,"%02d",_wine_Id[i]);

		cv::putText(drawImg,string(_id),cv::Point(left_bound,_start_row-10), cv::FONT_ITALIC, 2, cv::Scalar(255,0,0),2);

	}

	cv::Mat resize_drawImg;
	cv::resize(drawImg, resize_drawImg, cv::Size(drawImg.cols/2,drawImg.rows/2));

	return resize_drawImg;
}

WineCount::~WineCount(void)
{
}
