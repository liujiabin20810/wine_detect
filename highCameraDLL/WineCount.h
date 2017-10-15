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

	bool bottle_online(); //ƿ���Ƿ��ڱ�ʾ��

	bool detect_bound();  //���ƿ�ӱ���

	void update_bound();  //����ƿ�ӱ���

private:
	cv::Mat _source;	//ԭͼ
	cv::Mat _bg;		//����ͼ
	cv::Mat _binary;	//��ֵ��ͼ

	int _start_row;		//�ָ�ͼ��ʼ��
	int _end_row;		//�ָ�ͼ��ֹ��
	int _start_col;		//��ʾ��λ��

	int _bottle_width;	//ƿ����С�����ֵ

	int _i_row ;		//��ƿ����������к�
	bool _is_online;
	bool _new_bottle;

	int _wine_num;		//��ǰʶ��ƿ����Ŀ
	int _wine_count_num; //��ǰ��ƿ����
	std::vector<std::pair<int,int>> _wine_bounds;		//��ƿ�߽�
	std::vector<int> _wine_Id;		//��ƿ���(��С��������)

	std::vector<int> _delete_wine_Id; // ɾ����ƿ��Id
};

