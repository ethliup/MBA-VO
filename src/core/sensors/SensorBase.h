#ifndef __SRC_CORE_SENSORBASE_H_
#define __SRC_CORE_SENSORBASE_H_

#include "core/states/Transformation.h"

namespace SLAM
{
	namespace Core
	{
		class SensorBase
		{
		public:
			SensorBase();
			SensorBase(Transformation& T_body_to_sensor);

			Transformation& getSensorExtrinsics();

		private:
			Transformation m_T_body_to_sensor;
		};
	} // namespace Core
} // namespace SLAM



#endif /* SRC_COMMON_SENSOR_SENSORBASE_H_ */
