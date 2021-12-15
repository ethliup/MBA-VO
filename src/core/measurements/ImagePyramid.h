//
// Created by peidong on 2/16/20.
//

#ifndef SLAM_IMAGEPYRAMID_H
#define SLAM_IMAGEPYRAMID_H

#include "Image.h"
#include <map>
#include <math.h>

namespace SLAM
{
    namespace Core
    {
        template<class T>
        class ImagePyramid {
        public:
            ImagePyramid();
            ~ImagePyramid();

        public:
            void setNumOfPyramidLevels(int nLevels);
            void computePyramid(Image<T>* imagePtr);
            void copyPyramidFrom(int level, T* dataptr, int H, int W, int C);

        public:
            Image<T>* getImagePtr(int level);
            int getNumofPyramidLevels();

        private:
            int m_nLevels;
            std::map<int, Image<T>* > m_imagePyramidPtrMap; // (level, image): has ownership
        };

        /* ===================================================================================*
         *                            Class implementations                                   *
         * ===================================================================================*/
        template<class T>
        ImagePyramid<T>::ImagePyramid()
        {
            m_nLevels = 0;
        }

        template<class T>
        ImagePyramid<T>::~ImagePyramid()
        {
            for (typename std::map<int,Image<T>* >::iterator it=m_imagePyramidPtrMap.begin(); it!=m_imagePyramidPtrMap.end(); ++it)
                delete it->second;
        }

        template<class T>
        void ImagePyramid<T>::setNumOfPyramidLevels(int nLevels)
        {
            m_nLevels = nLevels;
        }

        template<class T>
        void ImagePyramid<T>::computePyramid(Image<T>* imagePtr)
        {
            int H0 = imagePtr->nHeight();
            int W0 = imagePtr->nWidth();
            int C = imagePtr->nChannels();
            this->copyPyramidFrom(0, imagePtr->getData(), H0, W0, C);

            for(int lv = 1; lv < m_nLevels; lv++)
            {
                T* dataPtr = m_imagePyramidPtrMap.at(lv-1)->getData();
                int Wlv_1 = m_imagePyramidPtrMap.at(lv-1)->nWidth();

                int Hlv = H0 / int(pow(2, lv));
                int Wlv = W0 / int(pow(2, lv));
                Image<T>* image_lv = new Image<T>(Hlv, Wlv, C);
                T* dataPtr_lv = image_lv->getData();

                // compute image
                for(int h = 0; h < Hlv; ++h)
                {
                    for(int w = 0; w < Wlv; ++w)
                    {
                        T* dataPtr00 = dataPtr + 2*h*Wlv_1*C +2*w*C;
                        T* dataPtr01 = dataPtr + 2*h*Wlv_1*C +(2*w+1)*C;
                        T* dataPtr10 = dataPtr + (2*h+1)*Wlv_1*C +2*w*C;
                        T* dataPtr11 = dataPtr + (2*h+1)*Wlv_1*C +(2*w+1)*C;

                        for(int c=0; c < C; ++c, ++dataPtr_lv)
                        {
                            dataPtr_lv[0] = T(0.25*(float(dataPtr00[c])+
                                                    float(dataPtr01[c])+
                                                    float(dataPtr10[c])+
                                                    float(dataPtr11[c])));
                        }
                    }
                }

                //
                m_imagePyramidPtrMap.insert(std::pair<int, Image<T>* >(lv, image_lv));
            }
        }

        template<class T>
        void ImagePyramid<T>::copyPyramidFrom(int level, T *dataptr, int H, int W, int C)
        {
            Image<T>* image = new Image<T>(H, W, C);
            image->copyFrom(dataptr, H, W, C);
            m_imagePyramidPtrMap.insert(std::pair<int, Image<T>* > (level, image));
        }

        template<class T>
        Image<T>* ImagePyramid<T>::getImagePtr(int level)
        {
            return m_imagePyramidPtrMap.at(level);
        }

        template<class T>
        int ImagePyramid<T>::getNumofPyramidLevels()
        {
            return m_nLevels;
        }
    }
}

#endif //SLAM_IMAGEPYRAMID_H
