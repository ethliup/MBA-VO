//
// Created by peidong on 2/15/20.
//

#ifndef SLAM_FEATUREDETECTORSEMIDENSE_H
#define SLAM_FEATUREDETECTORSEMIDENSE_H

#include "FeatureDetectorBase.h"
#include "core/measurements/ImagePyramid.h"

namespace SLAM
{
    namespace Core
    {
        class FeatureDetectorSemiDense : public FeatureDetectorBase
        {
        public:
            FeatureDetectorSemiDense(FeatureDetectorOptions &options);

        public:
            void detect(ImagePyramid<float> *gradImageMagPyramid);
        };
    } // namespace Core
} // namespace SLAM

#endif //SLAM_FEATUREDETECTORSEMIDENSE_H
