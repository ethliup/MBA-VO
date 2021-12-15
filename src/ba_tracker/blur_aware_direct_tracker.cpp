#include "blur_aware_direct_tracker.h"
#include "core/common/Time.h"
#include "core/feature_detectors/FeatureDetectorSemiDense.h"
#include "solve_normal_equation.h"
#include "utils/Geometry.h"
#include "utils/ImShow.h"
#include "utils/InputOutput.h"
#include "utils/ScalarToColorMap.h"

namespace SLAM
{
    namespace VO
    {
        BlurAwareDirectTracker::BlurAwareDirectTracker(BlurAwareDirectTrackerOptions &options)
            : mKeyframe(nullptr),
              mSplineTrajectory(new Core::SplineSE3()),
              mOptions(options),
              mLmStrategy(new LevenbergMarquardtStrategy()),
              mStepEvaluator(new TrustRegionStepEvaluator(options.max_consecutive_nonmonotonic_steps)),
              mIsFirstFrame(true),
              mAvgKernelLength(1e3)
        {
            initialize_shared_cuda_storages(mOptions.max_num_frames,
                                            mOptions.max_num_virtual_sharp_imgs_per_frame,
                                            mOptions.max_num_keypoints,
                                            mOptions.max_patch_size,
                                            mOptions.max_num_ctrl_knots,
                                            mOptions.spline_deg_k,
                                            mCudaSharedStorages);

            mSplineTrajectory->setSamplingFreq(mOptions.dt_ctrl_knot);

            mSplineVelocity << 0, 0, 0, 0, 0, 0;
            mNeighFrameVelocity << 0, 0, 0, 0, 0, 0;
        }

        BlurAwareDirectTracker::~BlurAwareDirectTracker()
        {
            delete mKeyframe;
            delete mSplineTrajectory;
            delete mLmStrategy;
            delete mStepEvaluator;
            free_shared_cuda_storages(mCudaSharedStorages);
        }

        void BlurAwareDirectTracker::setKeyframe(Core::Frame *keyframe)
        {
            mKeyframe = keyframe;
            mCurrentFrames.clear();
        }

        void BlurAwareDirectTracker::setCurrentFrames(const std::vector<Core::Frame *> &currentFrames)
        {
            mCurrentFrames = currentFrames;
        }

        void BlurAwareDirectTracker::insertCurrentFrame(Core::Frame *frame)
        {
            mCurrentFrames.clear();
            mCurrentFrames.push_back(frame);
        }

        void BlurAwareDirectTracker::setCamera(Core::CameraBase *camera)
        {
            mCamera = camera;
        }

        Core::Frame *BlurAwareDirectTracker::getKeyframe()
        {
            return mKeyframe;
        }

        BlurAwareDirectTrackerOptions &BlurAwareDirectTracker::getOptions()
        {
            return mOptions;
        }

        double BlurAwareDirectTracker::getFinalEnergy()
        {
            return mEvaluationPointCost;
        }

        Core::SplineSE3 *BlurAwareDirectTracker::getSplineTrajectory()
        {
            return mSplineTrajectory;
        }

        Core::Transformation BlurAwareDirectTracker::trackFrame(Core::Frame *sharp_frame, Core::Frame *blur_frame, Core::CameraBase *camera, std::string &datasetType, std::string &pathToDepthMap)
        {
            mCamera = camera;
            if (mIsFirstFrame)
            {
                mIsFirstFrame = false;
                this->setKeyframe(sharp_frame);
                this->tmpProcessKeyframe(sharp_frame, camera, datasetType, pathToDepthMap);
                mPrevTimestamp = sharp_frame->getCaptureTime();

                // initialize spline trajectory to be identity
                if (mSplineTrajectory->get_num_knots() == 0)
                {
                    mSplineTrajectory->setSamplingFreq(mOptions.dt_frame);
                    mSplineTrajectory->setSplineDegK(mOptions.spline_deg_k);
                    mSplineTrajectory->setStartTime(sharp_frame->getCaptureTime());
                    mSplineTrajectory->InsertControlKnot(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
                    mSplineTrajectory->InsertControlKnot(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
                }

                return mTKeyframe;
            }
            else
            {
                blur_frame->computeImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);
                for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
                {
                    blur_frame->getImagePtr(mOptions.camera_id, lv)->uploadToGpu();
                }
                this->insertCurrentFrame(blur_frame);

                // update spline trajectory
                double dt_frame = (blur_frame->getCaptureTime() - mPrevTimestamp);
                // mSplineVelocity *= dt_frame;
                mNeighFrameVelocity *= dt_frame;

                // // check if we need to revert motion direction
                // double t_angle = mSplineVelocity.head(3).dot(mNeighFrameVelocity.head(3));
                // double R_angle = mSplineVelocity.tail(3).dot(mNeighFrameVelocity.tail(3));

                // if (R_angle + t_angle < 0)
                // {
                //     mSplineVelocity = -mSplineVelocity;
                // }

                // if (mAvgKernelLength < 100)
                // {
                mSplineVelocity = mNeighFrameVelocity;
                // }
                mdTspline = Core::Transformation::exp(mSplineVelocity);

                // std::cout << "dt: " << blur_frame->getCaptureTime() - mPrevTimestamp << "\n";
                // std::cout << "tR_angle: " << t_angle << " " << R_angle << "\n";
                // std::cout << "vel: " << mSplineVelocity.transpose() << "\n";
                // std::cout << "vel_:" << mNeighFrameVelocity.transpose() << "\n";

                mSplineTrajectory->setStartTime(blur_frame->getCaptureTime() - 0.5 * blur_frame->getExposureTime());
                mSplineTrajectory->TransformByRight(mdTspline.getRotation(), mdTspline.getTranslation());

                // optimize trajectory
                // this->drawCurrFrameKeypoints(camera, "beforeOpt");

                this->optimizeTrajectory();

                // check if keyframe
                bool is_keyframe = this->isKeyframe(camera);

                // compute dT from two neighbouring frames
                Eigen::Vector3d t_b2w;
                Eigen::Quaterniond R_b2w;
                mSplineTrajectory->GetPose(blur_frame->getCaptureTime(), R_b2w, t_b2w);
                Core::Transformation T_b2w(R_b2w, t_b2w);
                mdTneighFrames = mTprevB2W.inverse() * T_b2w;
                mNeighFrameVelocity = Core::Transformation::log(mdTneighFrames) / dt_frame;
                mTprevB2W = T_b2w;

                /**
                 *  initialize trajectory
                 * */
                // compute velocity from spline trajectory, which is estimated from previous frame
                // Eigen::Vector3d t_b2w_start, t_b2w_end;
                // Eigen::Quaterniond R_b2w_start, R_b2w_end;
                // mSplineTrajectory->GetPose(mSplineTrajectory->getStartTime(), R_b2w_start, t_b2w_start);
                // mSplineTrajectory->GetPose(mSplineTrajectory->getStartTime() + blur_frame->getExposureTime(), R_b2w_end, t_b2w_end);

                // Core::Transformation T_b2w_start(R_b2w_start, t_b2w_start);
                // Core::Transformation T_b2w_end(R_b2w_end, t_b2w_end);

                // Core::Transformation dT = T_b2w_start.inverse() * T_b2w_end;
                // mSplineVelocity = Core::Transformation::log(dT);
                // mSplineVelocity = mSplineVelocity / blur_frame->getExposureTime();

                //
                // this->drawCurrFrameKeypoints(camera, "afterOpt");

                // process keyframe
                if (is_keyframe)
                {
                    this->setKeyframe(sharp_frame);
                    this->tmpProcessKeyframe(sharp_frame, camera, datasetType, pathToDepthMap);

                    mSplineTrajectory->GetPose(blur_frame->getCaptureTime(), R_b2w, t_b2w);
                    mTKeyframe = mTKeyframe * Core::Transformation(R_b2w, t_b2w);

                    R_b2w.setIdentity();
                    t_b2w.setZero();
                    mSplineTrajectory->TransformTo(blur_frame->getCaptureTime(), R_b2w, t_b2w);
                    mTprevB2W = Core::Transformation();
                }
                mPrevTimestamp = blur_frame->getCaptureTime();

                //
                mSplineTrajectory->GetPose(blur_frame->getCaptureTime(), R_b2w, t_b2w);
                return mTKeyframe * Core::Transformation(R_b2w, t_b2w);
            }
        }

        bool BlurAwareDirectTracker::isKeyframe(Core::CameraBase *camera)
        {
            double flow_squared_norm_sum = 0;
            double kernel_squared_norm_sum = 0;
            for (int j = 0; j < mOptions.num_keypoints[0]; j++)
            {
                Core::Vector2d kpt_ = mOptions.tmp_keypoints_xy[0][j];
                double z = mOptions.tmp_keypoints_z[0][j];
                Eigen::Vector2d kpt(kpt_(0), kpt_(1));
                Eigen::Vector3d P3d_ref;
                camera->unproject(kpt, z, P3d_ref);

                //
                Core::Transformation T_cur2ref;
                Eigen::Vector3d t_cur2ref;
                Eigen::Quaterniond R_cur2ref;
                Eigen::Vector2d kpt_cur, kpt_cur1;
                Eigen::Vector3d P3d_cur;

                double t0 = mCurrentFrames[0]->getCaptureTime();
                double dt = mCurrentFrames[0]->getExposureTime();

                mSplineTrajectory->GetPose(t0, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(P3d_cur, kpt_cur);
                flow_squared_norm_sum += (kpt_cur - kpt).squaredNorm();

                mSplineTrajectory->GetPose(t0 - 0.5 * dt, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(P3d_cur, kpt_cur);

                mSplineTrajectory->GetPose(t0 + 0.5 * dt, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(P3d_cur, kpt_cur1);

                kernel_squared_norm_sum += (kpt_cur - kpt_cur1).squaredNorm();
            }

            double avg_flow_norm = sqrtf(flow_squared_norm_sum / mOptions.num_keypoints[0]);
            mAvgKernelLength = sqrtf(kernel_squared_norm_sum / mOptions.num_keypoints[0]);

            // printf("avg_flow: %f; avg_kernel_len: %f\n", avg_flow_norm, mAvgKernelLength);

            if (avg_flow_norm > mOptions.keyframe_max_flow_mag0 && mAvgKernelLength < mOptions.keyframe_max_blur_kernel_mag)
            {
                // std::cout << "Is Keyframe...\n";
                return true;
            }

            if (avg_flow_norm > mOptions.keyframe_max_flow_mag1)
            {
                return true;
            }

            return false;
        }

        bool BlurAwareDirectTracker::isKeyframe(Core::CameraBase *camera, double &avg_t_flow_mag, double &avg_Rt_flow_mag, double &avg_kernel_len)
        {
            Eigen::Vector3d t_cur2ref;
            Eigen::Quaterniond R_cur2ref;
            Core::Transformation T_ref2cur_mid, T_ref2cur_begin, T_ref2cur_end;

            double t0 = mCurrentFrames[0]->getCaptureTime();
            double dt = mCurrentFrames[0]->getExposureTime();

            mSplineTrajectory->GetPose(t0, R_cur2ref, t_cur2ref);
            T_ref2cur_mid = Core::Transformation(R_cur2ref, t_cur2ref).inverse();

            mSplineTrajectory->GetPose(t0 - 0.5 * dt, R_cur2ref, t_cur2ref);
            T_ref2cur_begin = Core::Transformation(R_cur2ref, t_cur2ref).inverse();

            mSplineTrajectory->GetPose(t0 + 0.5 * dt, R_cur2ref, t_cur2ref);
            T_ref2cur_end = Core::Transformation(R_cur2ref, t_cur2ref).inverse();

            Eigen::Vector2d kpt_cur_mid, kpt_cur_begin, kpt_cur_end;
            Eigen::Vector3d P3d;

            double sum_squared_flow_t = 0.0;
            double sum_squared_flow_Rt = 0.0;
            double sum_squared_kernel_len = 0.0;

            for (int i = 0; i < mOptions.num_keypoints[0]; i++)
            {
                // get keypoint position & depth
                Core::Vector2d kpt_ = mOptions.tmp_keypoints_xy[0][i];
                double z = mOptions.tmp_keypoints_z[0][i];

                //
                Eigen::Vector2d kpt(kpt_(0), kpt_(1));
                Eigen::Vector3d P3d_ref;
                camera->unproject(kpt, z, P3d_ref);

                // compute flow induced by translation
                P3d = T_ref2cur_mid.getTranslation() + P3d_ref;
                camera->project(P3d, kpt_cur_mid);
                sum_squared_flow_t += (kpt_cur_mid - kpt).squaredNorm();

                // compute flow induced by rotation & translation
                P3d = T_ref2cur_mid * P3d_ref;
                camera->project(P3d, kpt_cur_mid);
                sum_squared_flow_Rt += (kpt_cur_mid - kpt).squaredNorm();

                // compute blur kernel length
                P3d = T_ref2cur_begin * P3d_ref;
                camera->project(P3d, kpt_cur_begin);

                P3d = T_ref2cur_end * P3d_ref;
                camera->project(P3d, kpt_cur_end);

                sum_squared_kernel_len += (kpt_cur_begin - kpt_cur_end).squaredNorm();
            }

            double avg_squared_flow_t = sum_squared_flow_Rt / (mOptions.num_keypoints[0] + 0.1);
            double avg_squared_flow_Rt = sum_squared_flow_Rt / (mOptions.num_keypoints[0] + 0.1);
            double avg_squared_kernel_len = sum_squared_kernel_len / (mOptions.num_keypoints[0] + 0.1);

            //
            avg_t_flow_mag = sqrtf(avg_squared_flow_t);
            avg_Rt_flow_mag = sqrtf(avg_squared_flow_Rt);
            avg_kernel_len = sqrtf(avg_squared_kernel_len);

            // check if current frame is keyframe
            if (avg_Rt_flow_mag > mOptions.keyframe_max_flow_mag0 && avg_t_flow_mag > mOptions.keyframe_max_flow_mag2 && avg_kernel_len < mOptions.keyframe_max_blur_kernel_mag)
            {
                return true;
            }

            if (avg_Rt_flow_mag > mOptions.keyframe_max_flow_mag1)
            {
                return true;
            }

            return false;
        }

        void BlurAwareDirectTracker::tmpProcessKeyframe(Core::Frame *keyframe, Core::CameraBase *camera, std::string &datasetType, std::string &pathToDepthMap)
        {
            keyframe->computeImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);
            keyframe->computeGradImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);

            for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
            {
                keyframe->getImagePtr(mOptions.camera_id, lv)->uploadToGpu();
                keyframe->getGradImagePtr(mOptions.camera_id, lv)->uploadToGpu();
            }

            Core::FeatureDetectorOptions feat_options;
            feat_options.detector_type = ENUM_SEMIDENSE;
            feat_options.score_threshold = 25;
            feat_options.grid_selection_cell_H = 30;
            feat_options.grid_selection_cell_W = 30;
            keyframe->detectFeatures(mOptions.camera_id, feat_options);

            // Utils::imshow(keyframe->getImagePtr(mOptions.camera_id),
            //               keyframe->getFeaturePoints(mOptions.camera_id, 0),
            //               2,
            //               "ref_frame",
            //               0);

            // load in gt depth map
            const int H = keyframe->getImagePtr(mOptions.camera_id)->nHeight();
            const int W = keyframe->getImagePtr(mOptions.camera_id)->nWidth();

            Core::Image<float> depth_ray_z(H, W, 1);
            if (datasetType == "unreal")
            {
                Core::Image<float> depth_ray_d(H, W, 1);
                Utils::load_depthMap(pathToDepthMap, &depth_ray_d);
                Utils::convert_ray_d_to_z(camera, &depth_ray_d, &depth_ray_z);
            }
            else if (datasetType == "eth3d")
            {
                cv::Mat depth = cv::imread(pathToDepthMap, CV_LOAD_IMAGE_ANYDEPTH);
                cv::Mat depthf;
                depth.convertTo(depthf, CV_32FC1);
                depthf = depthf / 5000;
                depth_ray_z.copyFrom((float *)depthf.data, H, W, 1);
            }

            //
            for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
            {
                double scale = pow(2, lv);
                std::vector<cv::KeyPoint> &cvKeypoints = keyframe->getFeaturePoints(mOptions.camera_id, lv);

                mTmpCpuKptXy[lv].clear();
                mTmpCpuKptZ[lv].clear();

                for (auto &kpt : cvKeypoints)
                {
                    int x = kpt.pt.x * scale + 0.5;
                    int y = kpt.pt.y * scale + 0.5;
                    float z = depth_ray_z.getData(y, x)[0];

                    if (z < 1e-2)
                    {
                        continue;
                    }

                    mTmpCpuKptXy[lv].push_back(Core::Vector2d(kpt.pt.x, kpt.pt.y));
                    mTmpCpuKptZ[lv].push_back(z);
                }

                mOptions.num_keypoints[lv] = mTmpCpuKptXy[lv].size();
                mOptions.tmp_keypoints_xy[lv] = mTmpCpuKptXy[lv].data();
                mOptions.tmp_keypoints_z[lv] = mTmpCpuKptZ[lv].data();
            }
        }

        void BlurAwareDirectTracker::drawKeyframeKeypoints(int pyra_level)
        {
            std::vector<Eigen::Vector2d> keypoints;
            for (int j = 0; j < mOptions.num_keypoints[pyra_level]; j++)
            {
                Core::Vector2d kpt_ = mOptions.tmp_keypoints_xy[pyra_level][j];
                Eigen::Vector2d kpt(kpt_.values[0], kpt_.values[1]);
                keypoints.push_back(kpt);
            }

            int waitKey = 2;
            Utils::imshow<unsigned char>(mKeyframe->getImagePtr(mOptions.camera_id, pyra_level),
                                         keypoints,
                                         2,
                                         "Keyframe",
                                         waitKey);
        } // namespace VO

        void BlurAwareDirectTracker::drawCurrFrameKeypoints(int pyra_lv, Core::CameraBase *camera, const std::string &winText)
        {
            std::vector<std::vector<Eigen::Vector2d>> all_line_segments;
            for (int j = 0; j < mOptions.num_keypoints[pyra_lv]; j++)
            {
                std::vector<Eigen::Vector2d> line;

                Core::Vector2d kpt_ = mOptions.tmp_keypoints_xy[pyra_lv][j];
                double z = mOptions.tmp_keypoints_z[pyra_lv][j];
                Eigen::Vector2d kpt(kpt_(0), kpt_(1));
                Eigen::Vector3d P3d_ref;
                camera->unproject(pyra_lv, kpt, z, P3d_ref);

                //
                Core::Transformation T_cur2ref;
                Eigen::Vector3d t_cur2ref;
                Eigen::Quaterniond R_cur2ref;
                Eigen::Vector2d kpt_cur;
                Eigen::Vector3d P3d_cur;

                double t0 = mCurrentFrames[0]->getCaptureTime();
                double dt = mCurrentFrames[0]->getExposureTime();

                mSplineTrajectory->GetPose(t0 - 0.5 * dt, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(pyra_lv, P3d_cur, kpt_cur);
                line.push_back(kpt_cur);

                mSplineTrajectory->GetPose(t0, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(pyra_lv, P3d_cur, kpt_cur);
                line.push_back(kpt_cur);

                mSplineTrajectory->GetPose(t0 + 0.5 * dt, R_cur2ref, t_cur2ref);
                T_cur2ref = Core::Transformation(R_cur2ref, t_cur2ref);
                P3d_cur = T_cur2ref.inverse() * P3d_ref;
                camera->project(pyra_lv, P3d_cur, kpt_cur);
                line.push_back(kpt_cur);

                all_line_segments.push_back(line);
            }

            int waitKey = 0;
            if (mCurrentFrames[0]->getFrameId() < mOptions.frame_id_start_to_pause_imshow)
            {
                waitKey = 2;
            }

            Utils::imshow<unsigned char>(mCurrentFrames[0]->getImagePtr(mOptions.camera_id, pyra_lv),
                                         all_line_segments,
                                         mOptions.line_thickness,
                                         winText,
                                         waitKey);
        }

        void BlurAwareDirectTracker::drawErrorResiduals(int pyrd_lv, std::vector<uchar> &outlier_flags)
        {
            int num_patches = mOptions.num_keypoints[pyrd_lv];
            int nelems = mOptions.spline_deg_k * 6 + 1;
            nelems = (1 + nelems) * nelems / 2;

            double *cpu_patch_cost_gradient_hessian = new double[num_patches * nelems];
            cudaMemcpy(cpu_patch_cost_gradient_hessian, mCudaSharedStorages.cuda_patch_cost_gradient_hessian_tR, sizeof(double) * num_patches * nelems, cudaMemcpyDeviceToHost);

            double min_cost = 1e6;
            double max_cost = 0;
            for (int i = 0; i < num_patches; i++)
            {
                double cost = cpu_patch_cost_gradient_hessian[i * nelems];
                if (cost < 1e-8)
                {
                    continue;
                }
                if (cost < min_cost)
                {
                    min_cost = cost;
                }
                if (cost > max_cost)
                {
                    max_cost = cost;
                }
            }

            //
            cv::Mat keyframe_image = Utils::convert_to_cvImage(mKeyframe->getImagePtr(mOptions.camera_id, pyrd_lv));
            for (int i = 0; i < num_patches; i++)
            {
                Core::Vector2d kpt_ = mOptions.tmp_keypoints_xy[pyrd_lv][i];
                cv::Point2f kpt(kpt_(0), kpt_(1));

                double cost = cpu_patch_cost_gradient_hessian[i * nelems];
                Eigen::Vector3d color_ = Utils::ScalarToColorMap(cost, min_cost, max_cost);
                cv::circle(keyframe_image, kpt, 3, cv::Scalar(color_(0), color_(1), color_(2)), CV_FILLED);

                if (outlier_flags[i] == 1 || cost < 1e-6)
                {
                    cv::circle(keyframe_image, kpt, 5, cv::Scalar(0, 0, 255));
                }
            }

            delete cpu_patch_cost_gradient_hessian;

            cv::imshow("error_residual", keyframe_image);
            cv::waitKey(2);
        }

        void BlurAwareDirectTracker::optimizeTrajectory()
        {
            // create cuda memory & upload data to gpu
            this->uploadDataToGpu();

            mFramesCtrlKnotSegStartIndices.clear();
            for (auto &frame : mCurrentFrames)
            {
                int idx;
                double u;
                Core::SplineSegmentStartKnotIdxAndNormalizedU(frame->getCaptureTime(),
                                                              mSplineTrajectory->getStartTime(),
                                                              mSplineTrajectory->getSamplingFreq(),
                                                              idx,
                                                              u);
                mFramesCtrlKnotSegStartIndices.push_back(idx);
            }

            //
            const int num_knots = mSplineTrajectory->get_num_knots();
            const int sz = num_knots * 6;
            mHessian.resize(sz, sz);
            mGradient.resize(sz, 1);
            mTrustRegionStep.resize(sz, 1);
            mCandidatePoint_t = new double[num_knots * 3];
            mCandidatePoint_R = new double[num_knots * 4];
            // Core::WallTime start_time = Core::currentTime();
            for (int i = 0; i < mOptions.num_pyramid_levels; i++)
            {
                const int pyra_level = mOptions.num_pyramid_levels - i - 1;
                this->optimizePyramidLevel(pyra_level);
            }
            // std::cout << "Consumes [" << Core::elapsedTimeInMilliSeconds(start_time) << "] ms to optimize ["
            //           << mOptions.num_keypoints[0] << "] keypoints...\n";

            if (mOptions.with_gui)
            {
                // drawErrorResiduals(pyra_level, outlier_flags);
                drawKeyframeKeypoints(0);
                drawCurrFrameKeypoints(0, mCamera, "estimatedKernel");
            }

            delete mCandidatePoint_t;
            delete mCandidatePoint_R;
        }

        void BlurAwareDirectTracker::optimizePyramidLevel(int pyra_level)
        {
            SolverSummary summary;

            // reset
            this->uploadDataToGpu(pyra_level);
            mEvaluationPointCost = 0;
            mCandidatePointCost = 0;
            mQuadraticApproximatedCost = 0;
            mStepQuality = 0;
            mCudaSharedStorages.num_bad_keypoints = 0;
            cudaMemset(mCudaSharedStorages.cuda_keypoints_outlier_flags, 0, sizeof(unsigned char) * mOptions.max_num_keypoints);

            // Iteration 0
            this->evaluateCostGradientAndHessian(pyra_level);
            mLmStrategy->reset();
            mStepEvaluator->reset(mEvaluationPointCost);

            std::vector<uchar> outlier_flags(mOptions.num_keypoints[pyra_level], 0);
            while (finalizeIterationAndCheckIfMinimizerCanContinue(summary))
            {
                // printf("pyra_lvl: %d iter: %2d eval_cost: %f\n",
                //        pyra_level,
                //        summary.num_iteration,
                //        mEvaluationPointCost);

                if (!computeTrustRegionStep())
                {
                    handleInvalidStep();
                    continue;
                }

                computeCandidatePointAndEvaluateCost(pyra_level);

                summary.abs_cost_decrease = mEvaluationPointCost - mCandidatePointCost;
                if (isStepSuccessful())
                {
                    detectOutliersAndUploadToGpu(outlier_flags);
                    handleSuccessfulStep(pyra_level);

                    // drawErrorResiduals(pyra_level, outlier_flags);
                    // drawKeyframeKeypoints(pyra_level);
                    // drawCurrFrameKeypoints(pyra_level, mCamera, "estimatedKernel");
                    continue;
                }
                handleUnsuccessfulStep();
            }
        }

        void BlurAwareDirectTracker::detectOutliersAndUploadToGpu(std::vector<uchar> &outlier_flags)
        {
            // TODO: this function only works for tracking "one"" frame with respect to keyframe
            int num_patches = outlier_flags.size();
            int nelems = mOptions.spline_deg_k * 6 + 1;
            nelems = (1 + nelems) * nelems / 2;

            double *cpu_patch_cost_gradient_hessian = new double[num_patches * nelems];
            cudaMemcpy(cpu_patch_cost_gradient_hessian, mCudaSharedStorages.cuda_patch_cost_gradient_hessian_tR, sizeof(double) * num_patches * nelems, cudaMemcpyDeviceToHost);

            double min_cost = 1e6;
            double max_cost = 0;
            double sum = 0;
            std::vector<double> all_costs;
            for (int i = 0; i < num_patches; i++)
            {
                double cost = cpu_patch_cost_gradient_hessian[i * nelems];
                if (cost < 1e-8)
                {
                    continue;
                }
                if (cost < min_cost)
                {
                    min_cost = cost;
                }
                if (cost > max_cost)
                {
                    max_cost = cost;
                }
                all_costs.push_back(cost);
                sum += cost;
            }

            // compute mean & variance
            double mu = sum / all_costs.size();
            double var = 0;
            for (int i = 0; i < all_costs.size(); i++)
            {
                var += (all_costs[i] - mu) * (all_costs[i] - mu);
            }
            var = var / all_costs.size();

            //
            int nOutliers = 0;
            for (int i = 0; i < num_patches; i++)
            {
                double cost = cpu_patch_cost_gradient_hessian[i * nelems];
                if (std::fabs(cost - mu) > mOptions.max_chi_square_error * sqrtf(var))
                {
                    outlier_flags[i] = 1;
                    nOutliers++;
                }
            }

            delete cpu_patch_cost_gradient_hessian;

            // upload to CUDA
            cudaMemcpy(mCudaSharedStorages.cuda_keypoints_outlier_flags, outlier_flags.data(), sizeof(unsigned char) * outlier_flags.size(), cudaMemcpyHostToDevice);

            mCudaSharedStorages.num_bad_keypoints = nOutliers;
        }

        void BlurAwareDirectTracker::uploadDataToGpu()
        {
            //
            std::vector<double> cpu_cap_time;
            std::vector<double> cpu_exp_time;
            for (auto &frame : mCurrentFrames)
            {
                cpu_cap_time.push_back(frame->getCaptureTime());
                cpu_exp_time.push_back(frame->getExposureTime());
            }
            cudaMemcpy(mCudaSharedStorages.cuda_img_cap_time,
                       cpu_cap_time.data(),
                       sizeof(double) * cpu_cap_time.size(),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(mCudaSharedStorages.cuda_img_exp_time,
                       cpu_exp_time.data(),
                       sizeof(double) * cpu_exp_time.size(),
                       cudaMemcpyHostToDevice);
        }

        void BlurAwareDirectTracker::uploadDataToGpu(int pyra_level)
        {
            std::vector<unsigned char *> cpu_cur_images;
            for (auto &frame : mCurrentFrames)
            {
                cpu_cur_images.push_back(frame->getImagePtr(mOptions.camera_id, pyra_level)->getGpuData());
            }

            cudaMemcpy(mCudaSharedStorages.cuda_cur_images,
                       cpu_cur_images.data(),
                       sizeof(void *) * mCurrentFrames.size(),
                       cudaMemcpyHostToDevice);

            //
            const int num_keypoints = mOptions.num_keypoints[pyra_level];
            cudaMemcpy(mCudaSharedStorages.cuda_keypoint_xy,
                       mOptions.tmp_keypoints_xy[pyra_level],
                       sizeof(Core::Vector2d) * num_keypoints,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(mCudaSharedStorages.cuda_keypoint_depth_z,
                       mOptions.tmp_keypoints_z[pyra_level],
                       sizeof(double) * num_keypoints,
                       cudaMemcpyHostToDevice);

            //
            cudaMemcpy(mCudaSharedStorages.cuda_local_patch_pattern_xy,
                       mOptions.local_patch_pattern_xy[pyra_level],
                       sizeof(int) * mOptions.patch_size[pyra_level] * 2,
                       cudaMemcpyHostToDevice);
        }

        void BlurAwareDirectTracker::evaluateCostGradientAndHessian(int pyra_level)
        {
            cudaMemcpy(mCudaSharedStorages.cuda_spline_ctrl_knots_data_t,
                       mSplineTrajectory->get_knot_data_t(),
                       sizeof(double) * 3 * mSplineTrajectory->get_num_knots(),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(mCudaSharedStorages.cuda_spline_ctrl_knots_data_R,
                       mSplineTrajectory->get_knot_data_R(),
                       sizeof(double) * 4 * mSplineTrajectory->get_num_knots(),
                       cudaMemcpyHostToDevice);

            //
            int scale = std::pow(2, pyra_level);

            Core::VectorX<double, 4> intrinsics = mOptions.intrinsics;
            intrinsics.values[0] /= scale;
            intrinsics.values[1] /= scale;
            intrinsics.values[2] /= scale;
            intrinsics.values[3] /= scale;

            Core::VectorX<int, 2> im_size_HW = mOptions.im_size_HW;
            im_size_HW.values[0] /= scale;
            im_size_HW.values[1] /= scale;

            //
            evaluate_cost_hessian_gradient(mOptions.num_virtual_poses_per_frame[pyra_level],
                                           mCurrentFrames.size(),
                                           mKeyframe->getImagePtr(mOptions.camera_id, pyra_level)->getGpuData(),
                                           mKeyframe->getGradImagePtr(mOptions.camera_id, pyra_level)->getGpuData(),
                                           mOptions.num_keypoints[pyra_level],
                                           mOptions.patch_size[pyra_level],
                                           intrinsics,
                                           im_size_HW,
                                           mOptions.spline_deg_k,
                                           mSplineTrajectory->getStartTime(),
                                           mSplineTrajectory->getSamplingFreq(),
                                           mFramesCtrlKnotSegStartIndices.data(),
                                           mSplineTrajectory->get_num_knots(),
                                           mCudaSharedStorages,
                                           mOptions.huber_k,
                                           &mEvaluationPointCost,
                                           mHessian.data(),
                                           mGradient.data());
        }

        bool BlurAwareDirectTracker::computeTrustRegionStep()
        {
            double radius = mLmStrategy->get_radius();
            double iradius = 1. / radius;
            mHessian.diagonal() += mHessian.diagonal() * iradius;

            if (mOptions.solver_type.compare("SVD_JACOBI") == 0)
            {
                solve_normal_equation(mHessian, mGradient, 0, mTrustRegionStep);
            }
            else if (mOptions.solver_type.compare("LDLT") == 0)
            {
                solve_normal_equation(mHessian, mGradient, 1, mTrustRegionStep);
            }
            else
            {
                std::cerr << "Unsupported linear solver " << mOptions.solver_type << "...\n";
                std::cerr << "Please select either \"SVD_JACOBI\" or \"LDLT\"... quit... \n";
                std::exit(0);
            }

            // compute quadratic approximated cost change
            mQuadraticApproximatedCost = mGradient.transpose() * mTrustRegionStep;
            mQuadraticApproximatedCost += 0.5 * mTrustRegionStep.transpose() * mHessian * mTrustRegionStep;
            mQuadraticApproximatedCost = -mQuadraticApproximatedCost;

            // printf("mQuadraticApproximatedCost: %f\n", mQuadraticApproximatedCost);
            if (mQuadraticApproximatedCost < 0)
            {
                return false;
            }
            return true;
        }

        void BlurAwareDirectTracker::computeCandidatePointAndEvaluateCost(int pyra_level)
        {
            mSplineTrajectory->Plus_t(mTrustRegionStep.data(), mCandidatePoint_t);
            mSplineTrajectory->Plus_R(mTrustRegionStep.data() + mSplineTrajectory->get_num_knots() * 3, mCandidatePoint_R);

            cudaMemcpy(mCudaSharedStorages.cuda_spline_ctrl_knots_data_t,
                       mCandidatePoint_t,
                       sizeof(double) * 3 * mSplineTrajectory->get_num_knots(),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(mCudaSharedStorages.cuda_spline_ctrl_knots_data_R,
                       mCandidatePoint_R,
                       sizeof(double) * 4 * mSplineTrajectory->get_num_knots(),
                       cudaMemcpyHostToDevice);

            //
            int scale = std::pow(2, pyra_level);

            Core::VectorX<double, 4> intrinsics = mOptions.intrinsics;
            intrinsics.values[0] /= scale;
            intrinsics.values[1] /= scale;
            intrinsics.values[2] /= scale;
            intrinsics.values[3] /= scale;

            Core::VectorX<int, 2> im_size_HW = mOptions.im_size_HW;
            im_size_HW.values[0] /= scale;
            im_size_HW.values[1] /= scale;

            //

            evaluate_cost_hessian_gradient(mOptions.num_virtual_poses_per_frame[pyra_level],
                                           mCurrentFrames.size(),
                                           mKeyframe->getImagePtr(mOptions.camera_id, pyra_level)->getGpuData(),
                                           mKeyframe->getGradImagePtr(mOptions.camera_id, pyra_level)->getGpuData(),
                                           mOptions.num_keypoints[pyra_level],
                                           mOptions.patch_size[pyra_level],
                                           intrinsics,
                                           im_size_HW,
                                           mOptions.spline_deg_k,
                                           mSplineTrajectory->getStartTime(),
                                           mSplineTrajectory->getSamplingFreq(),
                                           mFramesCtrlKnotSegStartIndices.data(),
                                           mSplineTrajectory->get_num_knots(),
                                           mCudaSharedStorages,
                                           mOptions.huber_k,
                                           &mCandidatePointCost,
                                           nullptr,
                                           nullptr);

            // printf("mCandidatePointCost: %f\n", mCandidatePointCost);
        }

        void BlurAwareDirectTracker::handleInvalidStep()
        {
            mLmStrategy->step_rejected();
        }

        bool BlurAwareDirectTracker::isStepSuccessful()
        {
            mStepQuality = mStepEvaluator->StepQuality(mCandidatePointCost, mQuadraticApproximatedCost);
            return mStepQuality > mOptions.min_step_quality && mCandidatePointCost < mEvaluationPointCost;
        }

        void BlurAwareDirectTracker::handleSuccessfulStep(int pyra_level)
        {
            mSplineTrajectory->InvalidParameter(mCandidatePoint_t, mCandidatePoint_R);
            this->evaluateCostGradientAndHessian(pyra_level);
            mLmStrategy->step_accepted(mStepQuality);
            // TODO: should we move this before evaluateCostGradientAndHessian()?
            mStepEvaluator->StepAccepted(mEvaluationPointCost, mQuadraticApproximatedCost);
        }

        void BlurAwareDirectTracker::handleUnsuccessfulStep()
        {
            mLmStrategy->step_rejected();
        }

        bool BlurAwareDirectTracker::finalizeIterationAndCheckIfMinimizerCanContinue(SolverSummary &summary)
        {
            summary.num_iteration++;
            if (summary.num_iteration > mOptions.max_num_iterations)
            {
                return false;
            }

            if (summary.abs_cost_decrease < mOptions.min_abs_cost_decrease)
            {
                return false;
            }

            return true;
        }
    } // namespace VO
} // namespace SLAM