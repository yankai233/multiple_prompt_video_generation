import os
import zipfile
import tarfile
from pyunpack import Archive
import subprocess


class FileCompressorExtractor:
    def __init__(self):
        pass

    def extract_file(self, file_path, extract_dir):
        """
        解压单个文件的函数
        :param file_path: 要解压的文件路径
        :param extract_dir: 解压到的目标目录
        """
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"成功解压 {file_path} 到 {extract_dir}")
        elif file_path.endswith(('.tar', '.tar.gz', '.tar.bz2')):
            with tarfile.open(file_path) as tar:
                tar.extractall(path=extract_dir)
            print(f"成功解压 {file_path} 到 {extract_dir}")
        elif file_path.endswith(('.rar', '.7z')):
            Archive(file_path).extractall(extract_dir)
            print(f"成功解压 {file_path} 到 {extract_dir}")
        else:
            print(f"不支持的文件格式: {file_path}")

    def extract_folder(self, folder_path):
        """
        解压文件夹中所有压缩文件的函数
        :param folder_path: 包含压缩文件的文件夹路径
        """
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                extract_dir = os.path.join(root, os.path.splitext(file)[0])
                if not os.path.exists(extract_dir):
                    os.makedirs(extract_dir)
                self.extract_file(file_path, extract_dir)

    def compress_files(self, file_paths, output_file):
        """
        压缩文件的函数
        :param file_paths: 要压缩的文件路径列表
        :param output_file: 压缩后的输出文件路径
        """
        if output_file.endswith('.zip'):
            with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in file_paths:
                    if os.path.isfile(file):
                        arcname = os.path.relpath(file, os.path.dirname(os.path.commonprefix(file_paths)))
                        zipf.write(file, arcname)
            print(f"成功压缩文件到 {output_file}")
        elif output_file.endswith('.tar'):
            with tarfile.open(output_file, 'w') as tar:
                for file in file_paths:
                    if os.path.isfile(file):
                        arcname = os.path.relpath(file, os.path.dirname(os.path.commonprefix(file_paths)))
                        tar.add(file, arcname=arcname)
            print(f"成功压缩文件到 {output_file}")
        elif output_file.endswith('.tar.gz'):
            with tarfile.open(output_file, 'w:gz') as tar:
                for file in file_paths:
                    if os.path.isfile(file):
                        arcname = os.path.relpath(file, os.path.dirname(os.path.commonprefix(file_paths)))
                        tar.add(file, arcname=arcname)
            print(f"成功压缩文件到 {output_file}")
        elif output_file.endswith('.tar.bz2'):
            with tarfile.open(output_file, 'w:bz2') as tar:
                for file in file_paths:
                    if os.path.isfile(file):
                        arcname = os.path.relpath(file, os.path.dirname(os.path.commonprefix(file_paths)))
                        tar.add(file, arcname=arcname)
            print(f"成功压缩文件到 {output_file}")
        elif output_file.endswith('.7z'):
            command = f'7z a "{output_file}" {" ".join(file_paths)}'
            subprocess.run(command, shell=True)
            print(f"成功压缩文件到 {output_file}")
        elif output_file.endswith('.rar'):
            command = f'rar a "{output_file}" {" ".join(file_paths)}'
            subprocess.run(command, shell=True)
            print(f"成功压缩文件到 {output_file}")
        else:
            print(f"不支持的压缩文件格式: {output_file}")

    def compress_floder(self, floder_path, output_file):
        self.compress_files(self.get_all_files_in_folder(floder_path), output_file)

    def get_all_files_in_folder(self, folder_path):
        """
        获取文件夹中所有文件的路径
        :param folder_path: 文件夹路径
        :return: 文件路径列表
        """
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths


if __name__ == "__main__":
    root_path = r'E:\Data\datasets\Video_Datasets\MovieNet\OpenDataLab___MovieNet'
    folder_path = os.path.join(root_path, r'raw\movie1K_keyframes_240p\240P')
    compressor_extractor = FileCompressorExtractor()
    compressor_extractor.extract_folder(folder_path)
    compressor_extractor.extract_file(os.path.join(root_path, r'raw\MovieNet.tar.gz'), os.path.join(root_path, 'raw'))
    compressor_extractor.extract_folder(os.path.join(root_path, 'raw/files'))