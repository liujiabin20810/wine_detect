// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 HIGHCAMERADLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// HIGHCAMERADLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef HIGHCAMERADLL_EXPORTS
#define HIGHCAMERADLL_API __declspec(dllexport)
#else
#define HIGHCAMERADLL_API __declspec(dllimport)
#endif

#include <opencv2/core/core.hpp>
#include <map>
#include <string>

class DetectWines;

// 此类是从 highCameraDLL.dll 导出的
class HIGHCAMERADLL_API ChighCameraDLL {
public:

	//设置皮带分割参数
	//设置数瓶子相关参数
	//path 为文件路径， 结尾包含'/'
	bool yanghe_init(std::string path); 

	//设置相机0，1，2，3的背景图
	bool yanghe_setBackground(cv::Mat background0, cv::Mat background1, cv::Mat background2, cv::Mat background3);

	void yanghe_detect(cv::Mat img0, cv::Mat img1, cv::Mat img2, cv::Mat img3);

	
	ChighCameraDLL(void);

	// (bottle_id , is_good)
	std::map<int, bool> yanghe_detectResult();

	// 利用上面返回的bottle_id，获取对应合成图片
	cv::Mat yanghe_combineImage(int bottle_id);


private:
	DetectWines* _detect_wine;

};

extern HIGHCAMERADLL_API int nhighCameraDLL;

HIGHCAMERADLL_API int fnhighCameraDLL(void);
