/*
 * @Author: Peidong Liu 
 * @Date: 2020-04-01 08:22:58 
 * @Last Modified by: Peidong Liu
 * @Last Modified time: 2020-04-01 08:27:28
 */

#ifndef SLAM_FEATUREDETECTORSPARSE_H
#define SLAM_FEATUREDETECTORSPARSE_H

#include "FeatureDetectorBase.h"
#include "core/kd_tree/KDTree.h"
#include "core/measurements/ImagePyramid.h"

namespace SLAM
{
    namespace Core
    {
        class FeatureDetectorSparse : public FeatureDetectorBase
        {
        public:
            FeatureDetectorSparse(FeatureDetectorOptions &options);
            ~FeatureDetectorSparse();

        public:
            void detect(Image<unsigned char> *grayImagePtr);

            std::vector<size_t> get_neighbor_feturePoint_indices(double x,
                                                                 double y,
                                                                 double radius);

        private:
            void construct_feature_points_kdtree();

        private:
            // void detect_fast(Image<unsigned char>* grayImagePtr, float threshold);
            void detect_orb(Image<unsigned char> *grayImagePtr);
            void compute_orb(Image<unsigned char> *grayImagePtr);

        private:
            cv::Ptr<cv::ORB> m_detector_orb;
            KDTree *m_featurePointsKdtree;
        };
    } // namespace Core
} // namespace SLAM

#endif