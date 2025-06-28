import cv2
import imageio


def mp4_to_gif(mp4_path, gif_path, fps=10):
    """
    将 MP4 文件转换为 GIF 文件
    :param mp4_path: MP4 文件的路径
    :param gif_path: 生成的 GIF 文件的路径
    :param fps: GIF 的帧率，默认为 10
    """
    # 打开 MP4 视频文件
    cap = cv2.VideoCapture(mp4_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # OpenCV 默认以 BGR 格式读取帧，需要转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break

    # 释放视频捕获对象
    cap.release()

    if frames:
        # 使用 imageio 保存为 GIF 文件，并设置循环参数
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"成功将 {mp4_path} 转换为 {gif_path}。")
    else:
        print(f"未从 {mp4_path} 中读取到有效帧。")


# 使用示例
mp4_file = r'result.mp4'
gif_file = r'result.gif'

# 调用函数进行转换
mp4_to_gif(mp4_file, gif_file, fps=15)
