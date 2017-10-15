#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>

class WineCount
{
public:
	
	WineCount(void) :_start_row(0), _end_row(0), _start_col(0), _wine_num(0), _bottle_width(0), _wine_count_num(0), 
		_is_online(false), _new_bottle(false){}

	WineCount(int start_row,int end_row, int start_col):_start_row(start_row),_end_row(end_row),_start_col(start_col)
		, _wine_num(0), _bottle_width(0), _wine_count_num(0), _is_online(false), _new_bottle(false){}

	void setROI(int start_row, int end_row, int start_col, int bottle_width , int row);

	void setBackground(cv::Mat bg);

	void setImage(cv::Mat image) ;

	void countWines();

	cv::Mat drawResult();

	std::vector<int> getWinesId(){ return _wine_Id;};

	int getWineNum() {return _wine_num; };

	std::vector<int> getDeletedWineId() { return _delete_wine_Id; };

	std::vector<int> getWinesMiddlePosition()
	{
		std::vector<int> _position;
		for (int i = 0; i < _wine_num; i++)
		{
			int _mid = (_wine_bounds[i].first + _wine_bounds[i].second) / 2;
			_position.push_back(_mid);
		}

		return _position;
	};

	~WineCount(void);

private:

	bool bottle_online(); //瓶子是否处于标示线

	bool detect_bound();  //检测瓶子边线

	void update_bound();  //更新瓶子边线

private:
	cv::Mat _source;	//原图
	cv::Mat _bg;		//背景图
	cv::Mat _binary;	//二值化图

	int _start_row;		//分割图起始行
	int _end_row;		//分割图终止行
	int _start_col;		//标示线位置

	int _bottle_width;	//瓶子最小宽度阈值

	int _i_row ;		//酒瓶最大宽度所在行号
	bool _is_online;
	bool _new_bottle;

	int _wine_num;		//当前识别瓶子数目
	int _wine_count_num; //当前酒瓶总数
	std::vector<std::pair<int,int>> _wine_bounds;		//酒瓶边界
	std::vector<int> _wine_Id;		//酒瓶编号(从小到大排序)

	std::vector<int> _delete_wine_Id; // 删除的瓶子Id
};

