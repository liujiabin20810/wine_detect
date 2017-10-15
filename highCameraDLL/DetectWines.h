#pragma once
#include <vector>
#include <map>
#include <string>

#include "BpNet.h"
#include "WineCount.h"

#include <opencv2/core/core.hpp>

class DetectWines
{
	//��������
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
		std::vector<cv::Point> _contours;	//��������
		cv::Point2f _center;		// ��������
		float _area;		// ���ʴ�С
		bool _finish;	// ƥ�����
		float _dist;		// �����ƶ�����
		int _type;		// ��������
		
	};

	//������ƿ����
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
			//��ƿ���
			unsigned int _uuID;
			//������
			vector<int> _camId;

			//��ƿ����
			vector<int> _mLine;
			//�Ƿ�������
			bool _is_good; 

			// ͼƬ����
			int _number;

			std::vector<cv::Mat> _wine_image;
			std::vector<cv::Mat> _wine_image_fg;
			std::vector<cv::Rect> _roi_rect; 
			
			bool _changed_finished;
			//��������
			std::vector<std::vector<Impuritie>> _impurities;
	};

public:
	DetectWines(void);
	
	//��ʼ����������
	bool init(std::string path);

	//�������0��1��2��3����ͼ
	void setBackground(cv::Mat _bg0,cv::Mat _bg1,cv::Mat _bg2,cv::Mat _bg3);

	// ����ƿ
	void countWines(cv::Mat source);

	cv::Mat getCountWineResult();

	//��ȡBP�������
	//path������� '/'
	bool readBpNet(string path);

	void updateWineData(cv::Mat cam1Img,cv::Mat cam2Img, cv::Mat cam3Img);

	void processWineData();

	std::map<int,bool> getResult();

	std::map<int ,cv::Mat> getCombinImage()	{
		return _draw_result_image;
	};

	~DetectWines(void);

private:
	//����ͼ������
	void setImageDate(cv::Mat img1,cv::Mat img2,cv::Mat img3);

	//�������1��2��3��ƿ������
	void calcWineMiddleLine();

	//�������ƥ��
	int cameraMatchWithMidLine(double _position, vector<int> _lines);

	void creatWineData();

	void detect_impuritie(std::vector<WineData>::iterator wd, int i);

	//��������ƥ��
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

	//ÿ�����Ƥ���ϱ��ص�Y����
	int _conveyorEdgeY0;
	int _conveyorEdgeY1;
	//���0ƿ�����߼�����
	float _detect_bottle_start_highT0;
	float _detect_bottle_widthT0;

	//���1��2��3ƿ�����߼�����
	float _detect_bottle_start_highT;
	float _detect_bottle_widthT;

	//�����camera1,camera 0����Ч���߷�Χ
	int camera1_midline_lTh;
	int camera1_midline_hTh;

	//�����camera2,camera 0����Ч���߷�Χ
	int camera2_midline_lTh;
	int camera2_midline_hTh;

	//�����camera3,camera 0����Ч���߷�Χ
	int camera3_midline_lTh;
	int camera3_midline_hTh;


	std::vector<WineData>  _wine_data;
	std::vector<cv::Mat> _background_images; //3�ű���ͼ
	cv::Mat _host_backgroud_image;

	std::vector<bool> _wine_data_added;
	
	std::map<int ,cv::Mat> _draw_result_image;		// �ϳ�ͼ

//	std::map<int, int> _matchedId;  // ��ǰͼ��������һ֡ƥ���ϵ

};

