#ifndef SLAM_VO_BLUR_AWARE_DIRECT_TRACKER_H
#define SLAM_VO_BLUR_AWARE_DIRECT_TRACKER_H

#include "core/common/Spline.h"
#include "core/measurements/Frame.h"
#include "levenberg_marquardt_strategy.h"
#include "spline_update_step.h"
#include "trust_region_step_evaluator.h"
#include <Eigen/Dense>
#include <vector>
namespace SLAM
{
    namespace VO
    {
        struct BlurAwareDirectTrackerOptions
        {
            int num_keypoints[8];
            Core::Vector2d *tmp_keypoints_xy[8];
            double *tmp_keypoints_z[8];

            //
            int camera_id;
            Core::VectorX<double, 4> intrinsics;
            Core::VectorX<int, 2> im_size_HW;

            //
            int num_pyramid_levels;
            int num_virtual_poses_per_frame[8];
            int patch_size[8];
            int *local_patch_pattern_xy[8];

            // huber robust function
            double huber_k;

            // solver options
            int max_consecutive_nonmonotonic_steps = 5;
            int max_num_iterations = 50;
            double min_step_quality = 0.5;
            double min_abs_cost_decrease = 0.001;
            std::string solver_type = "SVD_JACOBI"; // "LDLT"

            // options for shared storages
            int max_num_frames = 16;
            int max_num_virtual_sharp_imgs_per_frame = 64;
            int max_num_keypoints = 500;
            int max_patch_size = 128;
            int max_num_ctrl_knots = 16;

            // options for spline trajectory
            int spline_deg_k = 2;
            double dt_frame;
            double dt_ctrl_knot;

            // chi-square threshold for outlier rejection
            double max_chi_square_error;

            // params for keyframe creation
            double keyframe_max_flow_mag0;
            double keyframe_max_flow_mag1;
            double keyframe_max_flow_mag2;
            double keyframe_max_blur_kernel_mag;

            // params for imshow visualization
            bool with_gui;
            int frame_id_start_to_pause_imshow;
            int line_thickness;
        };

        struct SolverSummary
        {
            int num_iteration = 0;
            double abs_cost_decrease = 1e10;
        };

        class BlurAwareDirectTracker
        {
        public:
            BlurAwareDirectTracker(BlurAwareDirectTrackerOptions &options);
            ~BlurAwareDirectTracker();

        public:
            void setKeyframe(Core::Frame *keyframe);
            void setCurrentFrames(const std::vector<Core::Frame *> &currentFrames);
            void insertCurrentFrame(Core::Frame *frame);
            void setCamera(Core::CameraBase *camera);

        public:
            Core::Frame *getKeyframe();
            BlurAwareDirectTrackerOptions &getOptions();
            double getFinalEnergy();

        public:
            Core::SplineSE3 *getSplineTrajectory();

        public:
            Core::Transformation trackFrame(Core::Frame *sharp_frame, Core::Frame *blur_frame, Core::CameraBase *camera, std::string &datasetType, std::string &pathToDepthMap);
            void optimizeTrajectory();

        public:
            bool isKeyframe(Core::CameraBase *camera);
            bool isKeyframe(Core::CameraBase *camera, double &avg_t_flow_mag, double &avg_Rt_flow_mag, double &avg_blur_kernel);

        private:
            void optimizePyramidLevel(int pyra_level);
            void uploadDataToGpu();
            void uploadDataToGpu(int pyra_level);
            void evaluateCostGradientAndHessian(int pyra_level);
            bool computeTrustRegionStep();
            void computeCandidatePointAndEvaluateCost(int pyra_level);
            void handleInvalidStep();
            bool isStepSuccessful();
            void handleSuccessfulStep(int pyra_level);
            void handleUnsuccessfulStep();
            bool finalizeIterationAndCheckIfMinimizerCanContinue(SolverSummary &summary);
            void detectOutliersAndUploadToGpu(std::vector<uchar> &outlier_flags);

        public:
            void tmpProcessKeyframe(Core::Frame *keyframe, Core::CameraBase *camera, std::string &datasetType, std::string &pathToDepthMap);
            void drawKeyframeKeypoints(int pyra_level);
            void drawCurrFrameKeypoints(int pyrd_lv, Core::CameraBase *camera, const std::string &winText);
            void drawErrorResiduals(int pyrd_lv, std::vector<uchar> &outlier_flags);

        private:
            Core::Frame *mKeyframe;
            std::vector<Core::Frame *> mCurrentFrames;
            Core::SplineSE3 *mSplineTrajectory;
            CudaSharedStorages mCudaSharedStorages;
            BlurAwareDirectTrackerOptions mOptions;

            LevenbergMarquardtStrategy *mLmStrategy;
            TrustRegionStepEvaluator *mStepEvaluator;
            double mEvaluationPointCost;
            double mCandidatePointCost;
            double mQuadraticApproximatedCost;
            double mStepQuality;

            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mHessian;
            Eigen::Matrix<double, Eigen::Dynamic, 1> mGradient;
            Eigen::Matrix<double, Eigen::Dynamic, 1> mTrustRegionStep;
            double *mCandidatePoint_t;
            double *mCandidatePoint_R;

            std::vector<int> mFramesCtrlKnotSegStartIndices;

            //
            bool mIsFirstFrame;
            double mPrevTimestamp;
            Core::Transformation mTprevB2W;
            Core::Transformation mdTneighFrames;
            Core::Transformation mdTspline;
            Eigen::Matrix<double, 6, 1> mSplineVelocity;
            Eigen::Matrix<double, 6, 1> mNeighFrameVelocity;

            double mAvgKernelLength;

            //
            std::vector<Core::Vector2d> mTmpCpuKptXy[8];
            std::vector<double> mTmpCpuKptZ[8];
            Core::CameraBase *mCamera;

            //
            Core::Transformation mTKeyframe;
        };
    } // namespace VO
} // namespace SLAM

#endif