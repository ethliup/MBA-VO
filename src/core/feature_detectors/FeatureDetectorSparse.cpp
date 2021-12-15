#include "FeatureDetectorSparse.h"

namespace SLAM
{
    namespace Core
    {
        FeatureDetectorSparse::FeatureDetectorSparse(FeatureDetectorOptions &options)
            : FeatureDetectorBase(options)
        {
            //     int nfeatures = 500,
            //     float scaleFactor = 1.2f,
            //     int nlevels = 8,
            //     int edgeThreshold = 31,
            //     int firstLevel = 0,
            //     int WTA_K = 2,
            //     int scoreType = ORB::HARRIS_SCORE,
            //     int patchSize = 31,
            //     int fastThreshold = 20
            m_detector_orb = cv::ORB::create();
        }

        FeatureDetectorSparse::~FeatureDetectorSparse()
        {
            delete m_featurePointsKdtree;
        }

        void FeatureDetectorSparse::detect(Image<unsigned char> *grayImagePtr)
        {
            switch (mOptions.detector_type)
            {
            case ENUM_SPARSE_ORB:
            {
                m_detector_orb->setFastThreshold(mOptions.score_threshold);
                m_detector_orb->setMaxFeatures(mOptions.max_num_features_to_detect);
                m_detector_orb->setScoreType(mOptions.score_type);
                detect_orb(grayImagePtr);

                if (mOptions.grid_selection_cell_H > 0 && mOptions.grid_selection_cell_W > 0)
                {
                    this->gridSelection(grayImagePtr->nHeight(),
                                        grayImagePtr->nWidth(),
                                        mOptions.grid_selection_cell_H,
                                        mOptions.grid_selection_cell_W);
                }

                break;
            }
            case ENUM_SPARSE_SHI_TOMASI:
            {
                cv::Mat cvImage(grayImagePtr->nHeight(),
                                grayImagePtr->nWidth(),
                                CV_8UC1,
                                grayImagePtr->getData());

                std::vector<cv::Point2f> points;
                cv::goodFeaturesToTrack(cvImage,
                                        points,
                                        mOptions.max_num_features_to_detect,
                                        0.01,
                                        2);

                cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);

                cv::cornerSubPix(cvImage, points, cv::Size(10, 10), cv::Size(-1, -1), termcrit);

                // convert to internal representation
                std::vector<cv::KeyPoint> cv_keypoints;

                cv_keypoints.resize(points.size());

                for (size_t i = 0; i < cv_keypoints.size(); ++i)
                {
                    cv_keypoints.at(i).pt.x = points.at(i).x;
                    cv_keypoints.at(i).pt.y = points.at(i).y;
                }

                m_featurePointsMap.insert(
                    std::pair<int, std::vector<cv::KeyPoint>>(0, cv_keypoints));

                break;
            }
            default:
            {
                std::cerr << "Unsupported detector type: "
                          << "use ENUM_SPARSE_ORB or ENUM_SPARSE_SHI_TOMASI\n ";
                std::exit(0);
            }
            }

            // compute descriptors
            this->compute_orb(grayImagePtr);

            // construct kd tree is required
            if (mOptions.construct_kd_tree)
            {
                this->construct_feature_points_kdtree();
            }

            // initialize matchedP3dIdx
            m_matchedP3dIdx.insert(
                std::pair<int, std::vector<int>>(
                    0, std::vector<int>(m_featurePointsMap.at(0).size(), -1)));

            // initialize trackedPosition of each keypoint
            if (mOptions.initialize_keypoint_tracked_position)
            {
                std::vector<cv::Point2f> tracked_position;
                cv::KeyPoint::convert(m_featurePointsMap.at(0), tracked_position);

                m_trackedPosition.insert(
                    std::pair<int, std::vector<cv::Point2f>>(
                        0, tracked_position));

                m_hasGoodTrack.insert(
                    std::pair<int, std::vector<bool>>(0, std::vector<bool>(m_featurePointsMap.at(0).size(), true)));
            }
        }

        void FeatureDetectorSparse::detect_orb(Image<unsigned char> *grayImagePtr)
        {
            cv::Mat cvImage(grayImagePtr->nHeight(),
                            grayImagePtr->nWidth(),
                            CV_8UC1,
                            grayImagePtr->getData());
            std::vector<cv::KeyPoint> cvKeypoints;
            m_detector_orb->detect(cvImage, cvKeypoints);
            m_featurePointsMap.insert(std::pair<int, std::vector<cv::KeyPoint>>(0, cvKeypoints));
        }

        void
        FeatureDetectorSparse::compute_orb(Image<unsigned char> *grayImagePtr)
        {
            cv::Mat cvImage(grayImagePtr->nHeight(),
                            grayImagePtr->nWidth(),
                            CV_8UC1,
                            grayImagePtr->getData());

            cv::Mat cvDescriptors;
            m_detector_orb->compute(cvImage, m_featurePointsMap.at(0), cvDescriptors);
            m_descriptorsMap.insert(std::pair<int, cv::Mat>(0, cvDescriptors));
        }

        void FeatureDetectorSparse::construct_feature_points_kdtree()
        {
            std::vector<cv::KeyPoint> &all_keypoints = m_featurePointsMap.at(0);
            pointVec points;
            for (int i = 0; i < all_keypoints.size(); ++i)
            {
                point_t pt;
                pt = {all_keypoints.at(i).pt.x, all_keypoints.at(i).pt.y};
                points.push_back(pt);
            }
            m_featurePointsKdtree = new KDTree(points);
        }

        std::vector<size_t> FeatureDetectorSparse::get_neighbor_feturePoint_indices(double x, double y, double radius)
        {
            if (m_featurePointsKdtree == nullptr)
                return std::vector<size_t>();
            point_t pt = {x, y};
            return m_featurePointsKdtree->neighborhood_indices(pt, radius);
        }
    } // namespace Core
} // namespace SLAM