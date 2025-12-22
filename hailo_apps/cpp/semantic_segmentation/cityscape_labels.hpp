#include <cassert>
#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

class CityScapeLabels
{
private:
    std::array<cv::Vec3f, 19> _colors;

public:
    CityScapeLabels()
    {
        _colors[0]  = cv::Vec3f(128., 64.,  128.); // purple - road 1
        _colors[1]  = cv::Vec3f(244., 35.,  232.); // orange - sidewalk 2
        _colors[2]  = cv::Vec3f(70.,  70.,  70. ); // yellow - building 3
        _colors[3]  = cv::Vec3f(102., 102., 156.); // yellow - wall 4
        _colors[4]  = cv::Vec3f(190., 153., 153.); // yellow - fence 5
        _colors[5]  = cv::Vec3f(153., 153., 153.); // grey   - pole 6
        _colors[6]  = cv::Vec3f(250., 170., 30. ); // purple - trafficliight 7
        _colors[7]  = cv::Vec3f(220., 220., 0.  ); // purple - trafficsign 8
        _colors[8]  = cv::Vec3f(107., 142., 35. ); // orange - vegetation 9
        _colors[9]  = cv::Vec3f(152., 251., 152.); // orange - terrain 10
        _colors[10] = cv::Vec3f(70.,  130., 180.); // blue - sky 11
        _colors[11] = cv::Vec3f(220., 20.,  60. ); // red - person 12
        _colors[12] = cv::Vec3f(255., 0.,   0.  ); // green - rider 13
        _colors[13] = cv::Vec3f(0.,   0.,   142.); // green - car 14
        _colors[14] = cv::Vec3f(0.,   0.,   70. ); // green - truck 15
        _colors[15] = cv::Vec3f(0.,   60.,  100.); // green - bus 16
        _colors[16] = cv::Vec3f(0.,   80.,  100.); // green - train 17
        _colors[17] = cv::Vec3f(0.,   0.,   230.); // green - motorcycle 18
        _colors[18] = cv::Vec3f(119., 11.,  32. ); // green - bicycle 19                                       
    }
    cv::Vec3f id_2_color(int i)
    {
        assert(i >= 0 && i < 19);
        return _colors[i];
    }
};
