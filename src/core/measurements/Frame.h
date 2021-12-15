//
// Created by peidong on 2/13/20.
//

#ifndef SLAM_FRAME_H
#define SLAM_FRAME_H

#include "Image.h"
#include "ImagePyramid.h"
#include "core/common/Enums.h"
#include "core/feature_detectors/FeatureDetectorBase.h"
#include "core/sensors/CameraBase.h"
#include <map>

namespace SLAM
{
    namespace Core
    {
        class Frame
        {

        public:
            Frame();
            Frame(int frameId);
            Frame(int frameId, double elapsedTimeSinceProgramStart);
            ~Frame();

        public:
            void copyImageFrom(int cameraId, unsigned char *dataptr, size_t H, size_t W, size_t C);

        public:
            void setCaptureTime(double captureTime);
            void setExposureTime(double exposureTime);

        public:
            Image<unsigned char> *getImagePtr(int cameraId);
            Image<unsigned char> *getImagePtr(int cameraId, int level);
            Image<float> *getGradImagePtr(int cameraId, int level);
            Image<float> *getGradImageMagPtr(int cameraId, int level);

            std::vector<cv::KeyPoint> &getFeaturePoints(int cameraId, int level);
            cv::KeyPoint &getFeaturePoint(int cameraId,
                                          int level,
                                          int featureId);

            std::vector<size_t> getNeighborFeturePointIndices(
                int cameraId,
                int level,
                double x,
                double y,
                double radius);

            size_t getNumFeaturePoints(int level);

            cv::Mat &getFeatureDescriptors(int cameraId, int level);

            std::vector<int> &getFeatureMatchedP3dIdx(int cameraId, int level);

            FeatureDetectorBase *getFeatureDetector(int cameraId);

            int getFrameId();
            double getTimestamp();

            double getCaptureTime();
            double getExposureTime();

        public:
            void computeImagePyramid(int cameraId, int nLevels);
            void computeGradImagePyramid(int cameraId, int nLevels);
            void detectFeatures(int cameraId, Core::FeatureDetectorOptions &options);

        protected:
            // (cameraId, Image) have ownership
            std::map<int, Image<unsigned char> *> m_imagesPtrMap;

            // (cameraId, ImagePyramid) have ownership
            std::map<int, ImagePyramid<unsigned char> *> m_imagesPyramidPtrMap;

            // (cameraId, GradImagePyramid) have ownership
            std::map<int, ImagePyramid<float> *> m_gradImagePyramidPtrMap;

            // (cameraId, GradImageMagPyramid) have ownership
            std::map<int, ImagePyramid<float> *> m_gradImageMagPyramidPtrMap;

            // (cameraId, Features) have ownership
            std::map<int, FeatureDetectorBase *> m_featuresDetectorPtrMap;

            int m_frame_id;

            // in seconds
            double m_elapsedTimeSinceProgramStart;

            //
            double m_exposure_time;
            double m_capture_time;
        };
    } // namespace Core
} // namespace SLAM

#endif //SLAM_FRAME_H
