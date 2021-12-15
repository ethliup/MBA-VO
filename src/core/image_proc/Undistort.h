#ifndef SLAM_CORE_UNDISTORT
#define SLAM_CORE_UNDISTORT

#include <opencv2/opencv.hpp>
#include "core/sensors/CameraPinhole.h"
#include "core/sensors/CameraUnified.h"
#include "core/measurements/Image.h"

namespace SLAM
{
    namespace Core
    {
        class Undistort
        {
        public:
            Undistort(CameraBase *from_camera, CameraBase *to_camera);
            ~Undistort();

        public:
            cv::Mat
            undistort(const cv::Mat &image);

        protected:
            void computePixelMappings();

        private:
            CameraBase *mFromCamera;
            CameraBase *mToCamera;
            cv::Mat mMapx, mMapy;
        };
    } // namespace Core
} // namespace SLAM

#endif