# DataCollector 数据采集应用

这是整理后的独立项目目录，用于从摄像头或视频流采集原始帧、ROI 裁剪图和目标检测裁剪图，并支持 YOLO 标注。

## 运行

```powershell
cd D:\zyyProject\DataCollectorApp
python -m pip install -r requirements.txt
python src\data_collector_app.py
```

摄像头配置位于 `config\cameras.json`，默认模型位于 `models\yolov8n.pt`，采集结果默认写入 `output`。这些路径均可在应用界面中修改。

## 打包

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

打包结果位于 `dist\DataCollector`。`build` 是 PyInstaller 的中间目录，`DataCollector.spec` 是打包配置。

## 目录说明

- `src`：当前数据采集桌面应用源码。
- `config`：摄像头配置。
- `models`：应用默认使用的 YOLO 模型。
- `scripts`：构建脚本。
- `assets`：界面或说明图片。
- `legacy`：早期采集、抽帧和标注脚本，保留供参考。
- `build`：已有的 PyInstaller 构建中间产物。
- `dist`：已有的可执行程序及发布压缩包。
- `output`：程序运行后生成的采集数据，不纳入源码管理。

## 说明

`legacy` 中部分旧 `.py` 文件带有 `%TSD-Header-###%` 文件头，内容不是普通 Python 明文；本次整理只归档、不修改这些文件。
