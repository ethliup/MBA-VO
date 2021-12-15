//
// Created by peidong on 2/13/20.
//

#include "Frame.h"
#include "core/feature_detectors/FeatureDetectorSemiDense.h"
#include "core/feature_detectors/FeatureDetectorSparse.h"
#include "core/image_proc/Gradient.h"

namespace SLAM
{
    namespace Core
    {
        Frame::Frame()
        {
        }

        Frame::Frame(int frameId)
        {
            m_frame_id = frameId;
        }

        Frame::Frame(int frameId, double elapsedTimeSinceProgramStart)
        {
            m_frame_id = frameId;
            m_elapsedTimeSinceProgramStart = elapsedTimeSinceProgramStart;
        }

        Frame::~Frame()
        {
            for (std::map<int, Image<unsigned char> *>::iterator it = m_imagesPtrMap.begin(); it != m_imagesPtrMap.end(); ++it)
                delete it->second;

            for (std::map<int, ImagePyramid<unsigned char> *>::iterator it = m_imagesPyramidPtrMap.begin(); it != m_imagesPyramidPtrMap.end(); ++it)
                delete it->second;

            for (std::map<int, ImagePyramid<float> *>::iterator it = m_gradImagePyramidPtrMap.begin(); it != m_gradImagePyramidPtrMap.end(); ++it)
                delete it->second;

            for (std::map<int, ImagePyramid<float> *>::iterator it = m_gradImageMagPyramidPtrMap.begin(); it != m_gradImageMagPyramidPtrMap.end(); ++it)
                delete it->second;

            for (std::map<int, FeatureDetectorBase *>::iterator it = m_featuresDetectorPtrMap.begin(); it != m_featuresDetectorPtrMap.end(); ++it)
            {
                switch (it->second->getType())
                {
                case ENUM_SEMIDENSE:
                    delete static_cast<FeatureDetectorSemiDense *>(it->second);
                    break;
                case ENUM_SPARSE_ORB:
                case ENUM_SPARSE_SHI_TOMASI:
                    delete static_cast<FeatureDetectorSparse *>(it->second);
                    break;
                }
            }
        }

        void Frame::copyImageFrom(int cameraId, unsigned char *dataptr, size_t H, size_t W, size_t C)
        {
            Image<unsigned char> *imagePtr = new Image<unsigned char>();
            imagePtr->copyFrom(dataptr, H, W, C);
            m_imagesPtrMap.insert(std::pair<int, Image<unsigned char> *>(cameraId, imagePtr));
        }

        void Frame::setCaptureTime(double captureTime)
        {
            m_capture_time = captureTime;
        }

        void Frame::setExposureTime(double exposureTime)
        {
            m_exposure_time = exposureTime;
        }

        double Frame::getCaptureTime()
        {
            return m_capture_time;
        }

        double Frame::getExposureTime()
        {
            return m_exposure_time;
        }

        Image<unsigned char> *Frame::getImagePtr(int cameraId)
        {
            if (m_imagesPtrMap.count(cameraId) == 0)
                return nullptr;
            return m_imagesPtrMap.at(cameraId);
        }

        Image<unsigned char> *Frame::getImagePtr(int cameraId, int level)
        {
            return m_imagesPyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        Image<float> *Frame::getGradImagePtr(int cameraId, int level)
        {
            return m_gradImagePyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        Image<float> *Frame::getGradImageMagPtr(int cameraId, int level)
        {
            return m_gradImageMagPyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        int Frame::getFrameId()
        {
            return m_frame_id;
        }

        double Frame::getTimestamp()
        {
            return m_elapsedTimeSinceProgramStart;
        }

        void Frame::computeImagePyramid(int cameraId, int nLevels)
        {
            ImagePyramid<unsigned char> *imagePyramid = new ImagePyramid<unsigned char>();
            imagePyramid->setNumOfPyramidLevels(nLevels);
            imagePyramid->computePyramid(m_imagesPtrMap.at(cameraId));
            m_imagesPyramidPtrMap.insert(std::pair<int, ImagePyramid<unsigned char> *>(cameraId, imagePyramid));
        }

        void Frame::computeGradImagePyramid(int cameraId, int nLevels)
        {
            ImagePyramid<float> *gradImagePyramid = new ImagePyramid<float>();
            gradImagePyramid->setNumOfPyramidLevels(nLevels);

            ImagePyramid<float> *gradImageMagPyramid = new ImagePyramid<float>();
            gradImageMagPyramid->setNumOfPyramidLevels(nLevels);

            for (int lv = 0; lv < nLevels; ++lv)
            {
                Image<unsigned char> *imagePtr = m_imagesPyramidPtrMap.at(cameraId)->getImagePtr(lv);

                int H = imagePtr->nHeight();
                int W = imagePtr->nWidth();
                int C = imagePtr->nChannels();

                Image<float> gradImage(H, W, 2 * C);
                Image<float> gradImageMag(H, W, 1);

                compute_image_gradients<unsigned char, float>(imagePtr, &gradImage, &gradImageMag);

                gradImagePyramid->copyPyramidFrom(lv, gradImage.getData(), H, W, 2 * C);
                gradImageMagPyramid->copyPyramidFrom(lv, gradImageMag.getData(), H, W, 1);
            }

            m_gradImagePyramidPtrMap.insert(std::pair<int, ImagePyramid<float> *>(cameraId, gradImagePyramid));
            m_gradImageMagPyramidPtrMap.insert(std::pair<int, ImagePyramid<float> *>(cameraId, gradImageMagPyramid));
        }

        void Frame::detectFeatures(int cameraId, Core::FeatureDetectorOptions &options)
        {
            switch (options.detector_type)
            {
            case ENUM_SEMIDENSE:
            {
                FeatureDetectorSemiDense *featureDetector =
                    new FeatureDetectorSemiDense(options);

                featureDetector->detect(
                    m_gradImageMagPyramidPtrMap.at(cameraId));

                m_featuresDetectorPtrMap.insert(
                    std::pair<int, FeatureDetectorSemiDense *>(
                        cameraId, featureDetector));

                break;
            }

            case ENUM_SPARSE_ORB:
            case ENUM_SPARSE_SHI_TOMASI:
            {
                FeatureDetectorSparse *featureDetector =
                    new FeatureDetectorSparse(options);

                featureDetector->detect(m_imagesPtrMap.at(cameraId));

                m_featuresDetectorPtrMap.insert(
                    std::pair<int, FeatureDetectorSparse *>(
                        cameraId, featureDetector));
                break;
            }
            }
        }

        std::vector<cv::KeyPoint> &Frame::getFeaturePoints(int cameraId, int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeaturePoints(level);
        }

        cv::Mat &Frame::getFeatureDescriptors(int cameraId, int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeatureDescriptors(level);
        }

        std::vector<int> &Frame::getFeatureMatchedP3dIdx(int cameraId,
                                                         int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getMatchedP3dIdx(level);
        }

        FeatureDetectorBase *Frame::getFeatureDetector(int cameraId)
        {
            return m_featuresDetectorPtrMap.at(cameraId);
        }

        cv::KeyPoint &Frame::getFeaturePoint(int cameraId, int level, int featureId)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeaturePoints(level).at(featureId);
        }

        std::vector<size_t> Frame::getNeighborFeturePointIndices(int cameraId, int level, double x, double y, double radius)
        {
            return static_cast<FeatureDetectorSparse *>(m_featuresDetectorPtrMap.at(cameraId))->get_neighbor_feturePoint_indices(x, y, radius);
        }

        size_t Frame::getNumFeaturePoints(int level)
        {
            size_t num_keypoints = 0;
            for (auto it = m_featuresDetectorPtrMap.begin(); it != m_featuresDetectorPtrMap.end(); it++)
            {
                num_keypoints += it->second->getFeaturePoints(level).size();
            }
            return num_keypoints;
        }
    } // namespace Core
} // namespace SLAM