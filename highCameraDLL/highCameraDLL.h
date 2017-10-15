// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� HIGHCAMERADLL_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// HIGHCAMERADLL_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef HIGHCAMERADLL_EXPORTS
#define HIGHCAMERADLL_API __declspec(dllexport)
#else
#define HIGHCAMERADLL_API __declspec(dllimport)
#endif

#include <opencv2/core/core.hpp>
#include <map>
#include <string>

class DetectWines;

// �����Ǵ� highCameraDLL.dll ������
class HIGHCAMERADLL_API ChighCameraDLL {
public:

	//����Ƥ���ָ����
	//������ƿ����ز���
	//path Ϊ�ļ�·���� ��β����'/'
	bool yanghe_init(std::string path); 

	//�������0��1��2��3�ı���ͼ
	bool yanghe_setBackground(cv::Mat background0, cv::Mat background1, cv::Mat background2, cv::Mat background3);

	void yanghe_detect(cv::Mat img0, cv::Mat img1, cv::Mat img2, cv::Mat img3);

	
	ChighCameraDLL(void);

	// (bottle_id , is_good)
	std::map<int, bool> yanghe_detectResult();

	// �������淵�ص�bottle_id����ȡ��Ӧ�ϳ�ͼƬ
	cv::Mat yanghe_combineImage(int bottle_id);


private:
	DetectWines* _detect_wine;

};

extern HIGHCAMERADLL_API int nhighCameraDLL;

HIGHCAMERADLL_API int fnhighCameraDLL(void);
