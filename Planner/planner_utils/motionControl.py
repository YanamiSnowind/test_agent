from typing import Dict,List,Any,Optional
class motionControl:
    def _calculate_motion_parameters(self, parsed_info: Dict) -> Dict[str, Any]:
        """计算运动参数"""
        motion_char = parsed_info.get("motion_characteristics", {})
        action = parsed_info.get("action", "")

        # 从知识库获取基础参数
        base_params = self.knowledge_base["motion_types"].get(action, {
            "motion": "rotation",
            "axis": [0, 1, 0]
        })

        return {
            "rotation_axis": base_params.get("axis", [0, 1, 0]),
            "rotation_angle": self._map_amplitude_to_angle(motion_char.get("amplitude", "medium")),
            "translation": [0, 0, 0],  # 大多数情况下不需要平移
            "motion_type": base_params.get("motion", "rotation"),
            "easing_function": "ease_in_out"
        }

    def _map_amplitude_to_angle(self, amplitude: str) -> float:
        """将幅度描述映射为角度"""
        amplitude_map = {
            "small": 15.0,
            "medium": 30.0,
            "large": 60.0,
            "extreme": 90.0
        }
        return amplitude_map.get(amplitude, 30.0)

    def _calculate_frame_count(self, parsed_info: Dict) -> int:
        """计算动画帧数"""
        frequency = parsed_info.get("motion_characteristics", {}).get("frequency", "low")
        frequency_map = {
            "low": 20,
            "medium": 30,
            "high": 60
        }
        return frequency_map.get(frequency, 30)