import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os


class ABRAnalyzerFolder:
    """ABR 数据处理与分析类库（仅支持 hearlab XML 格式）"""

    def __init__(self):
        self.time: np.ndarray = np.array([])
        self.voltage: np.ndarray = np.array([])
        self.sample_rate: float = 0.0

    def load_from_folder(self, folder_path: str, plot_each: bool = True) -> List[Dict[str, np.ndarray]]:
        """
        批量加载文件夹内的 hearlab 格式 XML 文件，并可选择是否绘制每个文件的波形

        :param folder_path: 包含 XML 文件的文件夹路径
        :param plot_each: 是否对每个文件单独绘制波形（默认 True）
        :return: 成功加载的文件数据列表 [{"time": ndarray, "voltage": ndarray, "sample_rate": float}]
        """
        loaded_data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".xml"):
                try:
                    with open(os.path.join(folder_path, filename), 'r') as f:
                        xml_content = f.read()
                    self.load_from_xml(xml_content)

                    # 存储当前文件数据
                    file_data = {
                        "filename": filename,
                        "time": self.time.copy(),
                        "voltage": self.voltage.copy(),
                        "sample_rate": self.sample_rate
                        "label":float(filename.split("_")[0])
                    }
                    loaded_data.append(file_data)

                    # 如果启用 plot_each，绘制当前文件的波形
                    if plot_each:
                        v_waves = self.detect_v_waves()
                        peaks = self.label_peaks()
                        self.plot(v_waves=v_waves, peaks=peaks, title=f"ABR Analysis: {filename}")

                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        return loaded_data

    def load_from_xml(self, xml_content: str) -> None:
        """解析 hearlab 格式 XML 数据"""
        root = ET.fromstring(xml_content)
        points_str = root.find(".//PointsValue").text
        points = [tuple(map(float, pt.strip("()").split(","))) for pt in points_str.split("),(")]
        self.time = np.array([pt[0] for pt in points])
        self.voltage = np.array([pt[1] for pt in points])
        self.sample_rate = 1 / (self.time[1] - self.time[0]) * 1000  # 计算采样率（Hz）

    def detect_v_waves(self, min_prominence: float = 10.0) -> List[Tuple[float, float]]:
        """检测 V 型波形（波谷+两侧波峰）"""
        valleys, _ = find_peaks(-self.voltage, prominence=min_prominence)
        peaks, _ = find_peaks(self.voltage, prominence=min_prominence)

        v_waves = []
        for v in valleys:
            left_peak = peaks[peaks < v][-1] if any(peaks < v) else None
            right_peak = peaks[peaks > v][0] if any(peaks > v) else None

            if left_peak and right_peak:
                left_height = self.voltage[left_peak] - self.voltage[v]
                right_height = self.voltage[right_peak] - self.voltage[v]
                if left_height > min_prominence and right_height > min_prominence:
                    v_waves.append((self.time[v], self.voltage[v]))
        return v_waves

    def label_peaks(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """标注 ABR 关键波（I、III、V）"""
        peaks, _ = find_peaks(self.voltage, prominence=10, distance=int(0.5 * self.sample_rate))
        peaks = sorted(peaks, key=lambda x: self.time[x])

        labels = {}
        if len(peaks) >= 1:
            labels["wave_I"] = (self.time[peaks[0]], self.voltage[peaks[0]])
        if len(peaks) >= 3:
            labels["wave_III"] = (self.time[peaks[1]], self.voltage[peaks[1]])
            labels["wave_V"] = (self.time[peaks[2]], self.voltage[peaks[2]])
        return labels

    def plot(self, v_waves: List[Tuple[float, float]] = None,
             peaks: Dict[str, Tuple[float, float]] = None,
             title: str = "ABR Analysis") -> None:
        """绘制 ABR 信号和标注"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time, self.voltage, label="ABR Signal")

        if v_waves:
            v_times, v_voltages = zip(*v_waves)
            plt.scatter(v_times, v_voltages, color="red", s=100, label="V-Wave")

        if peaks:
            for name, (t, v) in peaks.items():
                plt.scatter(t, v, marker="*", s=200, label=name)

        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (µV)")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()


# 使用示例
if __name__ == "__main__":
    analyzer = ABRAnalyzerFolder()

    # 加载文件夹并自动绘制每个文件的波形
    loaded_data = analyzer.load_from_folder(r"C:\ABR\data\hearlab\ABRI02R", plot_each=True)

    # 也可以选择不自动绘图，后续手动处理
    # loaded_data = analyzer.load_from_folder("path/to/hearlab_xmls", plot_each=False)
    # for data in loaded_data:
    #     analyzer.time = data["time"]
    #     analyzer.voltage = data["voltage"]
    #     analyzer.sample_rate = data["sample_rate"]
    #     v_waves = analyzer.detect_v_waves()
    #     peaks = analyzer.label_peaks()
    #     analyzer.plot(v_waves=v_waves, peaks=peaks, title=f"ABR Analysis: {data['filename']}")
