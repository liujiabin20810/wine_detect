#pragma once
#include <vector>
#include <map>
#include <string>

#include "BpNet.h"
#include "WineCount.h"

#include <opencv2/core/core.hpp>

class DetectWines
{
	//杂质数据
	class Impuritie
	{
	public:
		Impuritie(std::vector < cv::Point > contours):_contours(contours),_finish(false),_dist(0.0),_type(0)
		{}

		~Impuritie()
		{
			_contours.clear();
		};
	public:
		std::vector<cv::Point> _contours;	//杂质轮廓
		cv::Point2f _center;		// 杂质中心
		float _area;		// 杂质大小
		bool _finish;	// 匹配结束
		float _dist;		// 杂质移动距离
		int _type;		// 杂质类型
		
	};

	//单个酒瓶数据
	class WineData
	{

		public:
			WineData(void) :_uuID(0), _number(0),_is_good(true){}
			WineData(int id, bool flag) :_uuID(id), _number(0), _is_good(flag){}
			~WineData(){
				_wine_image.clear();
				_wine_image_fg.clear();
				_roi_rect.clear();

				_mLine.clear();

				for (int i=0; i < _impurities.size(); i++)
				{
					_impurities[i].clear();
				}

				_impurities.clear();
			};

		public:
			//酒瓶编号
			unsigned int _uuID;
			//相机编号
			vector<int> _camId;

			//酒瓶中线
			vector<int> _mLine;
			//是否有杂质
			bool _is_good; 

			// 图片数量
			int _number;

			std::vector<cv::Mat> _wine_image;
			std::vector<cv::Mat> _wine_image_fg;
			std::vector<cv::Rect> _roi_rect; 
			
			bool _changed_finished;
			//杂质数据
			std::vector<std::vector<Impuritie>> _impurities;
	};

public:
	DetectWines(void);
	
	//初始化参数配置
	bool init(std::string path);

	//设置相机0，1，2，3背景图
	void setBackground(cv::Mat _bg0,cv::Mat _bg1,cv::Mat _bg2,cv::Mat _bg3);

	// 数酒瓶
	void countWines(cv::Mat source);

	cv::Mat getCountWineResult();

	//读取BP网络参数
	//path后面包含 '/'
	bool readBpNet(string path);

	void updateWineData(cv::Mat cam1Img,cv::Mat cam2Img, cv::Mat cam3Img);

	void processWineData();

	std::map<int,bool> getResult();

	std::map<int ,cv::Mat> getCombinImage()	{
		return _draw_result_image;
	};

	~DetectWines(void);

private:
	//设置图像数据
	void setImageDate(cv::Mat img1,cv::Mat img2,cv::Mat img3);

	//计算相机1，2，3中瓶子中线
	void calcWineMiddleLine();

	//相机中线匹配
	int cameraMatchWithMidLine(double _position, vector<int> _lines);

	void creatWineData();

	void detect_impuritie(std::vector<WineData>::iterator wd, int i);

	//杂质轮廓匹配
	std::map<int, int> contoursMatch(std::vector<Impuritie> imprt1, std::vector<Impuritie> imprt2);

	std::vector<int> getMidlines(cv::Mat src, bool showFlag, int camId);

	bool getImageROI(std::vector<int> _camlines, BpNet _bp, int _line, cv::Rect & rect);

	cv::Mat getDrawedImage(cv::Mat& color_roi_img, int uuid, int camId);

	cv::Mat getImage(int i,int j);

	void addImage();

	void drawLines(std::vector<WineData>::iterator wd ,std::map<int,int> matchedId);


public:
//	std::map<int,cv::Mat> _wine_complex_image;

private:

	int image_width;
	int image_hight;

	BpNet _cam1_bpNet;
	BpNet _cam2_bpNet;
	BpNet _cam3_bpNet;

	//Count Wine
	WineCount _wcount;
	std::vector<int> _wine_Id;
	std::vector<int> _wine_middle_position;

	std::vector<int> _deleted_wine_Id;

	// Image Data
	cv::Mat _cam1Img;
	cv::Mat _cam2Img;
	cv::Mat _cam3Img;

	// _binary
	cv::Mat _cam1_bottle;
	cv::Mat _cam2_bottle;
	cv::Mat _cam3_bottle;
	//Middle line
	std::vector<int> _cam1_wine_mid_lines;
	std::vector<int> _cam2_wine_mid_lines;
	std::vector<int> _cam3_wine_mid_lines;

	//每个相机皮带上边沿的Y坐标
	int _conveyorEdgeY0;
	int _conveyorEdgeY1;
	//相机0瓶子中线检测参数
	float _detect_bottle_start_highT0;
	float _detect_bottle_widthT0;

	//相机1，2，3瓶子中线检测参数
	float _detect_bottle_start_highT;
	float _detect_bottle_widthT;

	//相对于camera1,camera 0的有效中线范围
	int camera1_midline_lTh;
	int camera1_midline_hTh;

	//相对于camera2,camera 0的有效中线范围
	int camera2_midline_lTh;
	int camera2_midline_hTh;

	//相对于camera3,camera 0的有效中线范围
	int camera3_midline_lTh;
	int camera3_midline_hTh;


	std::vector<WineData>  _wine_data;
	std::vector<cv::Mat> _background_images; //3张背景图
	cv::Mat _host_backgroud_image;

	std::vector<bool> _wine_data_added;
	
	std::map<int ,cv::Mat> _draw_result_image;		// 合成图

//	std::map<int, int> _matchedId;  // 当前图杂质与上一帧匹配关系

};

