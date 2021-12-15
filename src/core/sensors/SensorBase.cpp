#include "SensorBase.h"

namespace SLAM
{
    namespace Core
    {
        SensorBase::SensorBase()
        {}

        SensorBase::SensorBase(Transformation& T_body_to_sensor)
        {
            m_T_body_to_sensor = T_body_to_sensor;
        }

        Transformation& SensorBase::getSensorExtrinsics()
        {
            return m_T_body_to_sensor;
        }

    } //namespace Core
} // namespace SLAM